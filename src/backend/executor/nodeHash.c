/*-------------------------------------------------------------------------
 *
 * CSI 3130
 * 2024 FALL - UNIVERSITY OF OTTAWA
 * DATABASES II
 * JOHN SURETTE
 * 300307306
 * nodehash.c
 * Routines to hash relations for hashjoin
 *
 * IDENTIFICATION
 *	  src/backend/executor/nodeHash.c
 *
 *-------------------------------------------------------------------------
 */
/*
 * INTERFACE ROUTINES
 *		multi_exec_hash	- generate an in-memory hash table of the relation
 *		ExecInitHash	- initialize node and subnodes
 *		ExecEndHash		- shutdown node and subnodes
 */

#include "postgres.h"

#include <math.h>
#include <limits.h>
#include "commands/tablespace.h"
#include "executor/execdebug.h"
#include "miscadmin.h"
#include "utils/dynahash.h"

#include "utils/lsyscache.h"
#include "utils/syscache.h"

#include "access/htup_details.h"
#include "catalog/pg_statistic.h"
#include "utils/memutils.h"

#include "executor/hashjoin.h"
#include "executor/nodeHash.h"
#include "executor/nodeHashjoin.h"



static void hash_table_increase_batch_count(HashJoinTable hashtable);
static void hash_table_increase_bucket_count(HashJoinTable hashtable);
static void build_skew_hash_table(HashJoinTable hashtable, Hash *node,
					  int mcvsToUse);
static void insert_into_skew_table(HashJoinTable hashtable,
						TupleTableSlot *slot,
						uint32 hashvalue,
						int bucketNumber);
static void remove_next_skew_bucket(HashJoinTable hashtable);

static void *dense_alloc(HashJoinTable hashtable, Size size);

/* ----------------------------------------------------------------
 *		exec_hash

 *
 *		stub for pro forma compliance
 * ----------------------------------------------------------------
 */
static TupleTableSlot *
exec_hash(PlanState *pstate)
{
	elog(ERROR, "Hash node does not support ExecProcNode call convention");
	return NULL;
}
/* -------------------------------------------------------------------------
 * exec_multi_hash
 * 
 * Builds the hash table for the symmetric hash join, performing partitioning if
 * more than one batch is required, and executing in a pipelined manner.
 * ------------------------------------------------------------------------- */

Node *
exec_multi_hash(HashState *hash_state)
{
    HashJoinTable hash_table;
    TupleTableSlot *slot;

    PlanState *outer_node;
    ExprContext *expr_context;
    uint32 hash_value;
    List *hash_keys;

    /* Instrumentation support */
    if (hash_state->ps.instrument)
        InstrStartNode(hash_state->ps.instrument);

    /* Get state information from hash_state */
    outer_node = outerPlanState(hash_state);
    hash_table = hash_state->hashtable;
    hash_keys = hash_state->hashkeys;
    expr_context = hash_state->ps.ps_ExprContext;
    
    /* Process tuples in a pipelined manner */
    for (;;)
    {
        slot = ExecProcNode(outer_node);
        if (TupIsNull(slot))
            break;

        /* Compute the hash value */
        expr_context->ecxt_innertuple = slot;
        if (exec_hash_get_hash_value(hash_table, expr_context, hash_keys,
                                    false, hash_table->keepNulls, &hash_value))
        {
            int bucket_number = exec_hash_get_skew_bucket(hash_table, hash_value);
            if (bucket_number != INVALID_SKEW_BUCKET_NO)
            {
                insert_into_skew_table(hash_table, slot, hash_value, bucket_number);
                hash_table->skewTuples += 1;
            }
            else
            {	
				/* Log that no skew bucket was found for the given hash value */
				#ifdef HJDEBUG
                printf("Hashjoin %p: No skew bucket found for hash value %u\n", hash_table, hash_value);
				#endif
                exec_hash_table_insert(hash_table, slot, hash_value);
            }
            hash_table->totalTuples += 1;
        }

        /* 
         * As this is a pipelined execution model, we are continuously probing both relations.
         * Trigger bi-directional probing here.
         */
        exec_probe_both_sides();
    }

    /* Resize the hash table if needed (NTUP_PER_BUCKET exceeded) */
    if (hash_table->nbuckets != hash_table->nbuckets_optimal)
        hash_table_increase_bucket_count(hash_table);

    /* Account for the buckets in space usage (reported in EXPLAIN ANALYZE) */
    hash_table->spaceUsed += hash_table->nbuckets * sizeof(HashJoinTuple);
    if (hash_table->spaceUsed > hash_table->spacePeak)
        hash_table->spacePeak = hash_table->spaceUsed;

    /* Instrumentation support */
    if (hash_state->ps.instrument)
        InstrStopNode(hash_state->ps.instrument, hash_table->totalTuples);

    return NULL;
}

/* -------------------------------------------------------------------------
 * exec_init_hash
 *
 * Initializes the HashState for the hash node, setting up the state structure,
 * child expressions, child nodes, and tuple types.
 * ------------------------------------------------------------------------- */

HashState *
exec_init_hash(Hash *hash_node, EState *execution_state, int execution_flags)
{
    HashState *hash_state;

    /* Check for unsupported flags */
    Assert(!(execution_flags & (EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK)));

    /* Create state structure */
    hash_state = makeNode(HashState);
    hash_state->ps.plan = (Plan *) hash_node;
    hash_state->ps.state = execution_state;
    hash_state->ps.ExecProcNode = exec_hash;
    hash_state->hashtable = NULL;
    hash_state->hashkeys = NIL; /* Will be set by the parent HashJoin */

    /* Miscellaneous initialization: create expression context for node */
    ExecAssignExprContext(execution_state, &hash_state->ps);

    /* Initialize result tuple slot */
    ExecInitResultTupleSlot(execution_state, &hash_state->ps);

    /* Initialize child expressions */
    hash_state->ps.qual = ExecInitQual(hash_node->plan.qual, (PlanState *) hash_state);

    /* Initialize child nodes */
    outerPlanState(hash_state) = ExecInitNode(outerPlan(hash_node), execution_state, execution_flags);

    /* Initialize tuple type. No need to initialize projection info because this node doesn't perform projections */
    ExecAssignResultTypeFromTL(&hash_state->ps);
    hash_state->ps.ps_ProjInfo = NULL;

    return hash_state;
}


/* -------------------------------------------------------------------------
 * exec_end_hash
 * 
 * Cleans up resources used by the hash node, freeing contexts and shutting down
 * any child subplans.
 * ------------------------------------------------------------------------- */
void
exec_end_hash(HashState *hash_state)
{
    PlanState *outer_plan;

    /* Free expression context */
    ExecFreeExprContext(&hash_state->ps);

    /* Shut down the subplan */
    outer_plan = outerPlanState(hash_state);
    ExecEndNode(outer_plan);
}


/* -------------------------------------------------------------------------
 * exec_hash_table_create
 *
 * Creates an empty hash table data structure for symmetric hash join.
 * ------------------------------------------------------------------------- */
HashJoinTable
exec_hash_table_create(Hash *node, List *hash_operators, bool keep_nulls)
{
    HashJoinTable hash_table;
    Plan *outer_node;
    int num_buckets;
    int num_batches;
    int num_skew_mcvs;
    int log2_num_buckets;
    int num_keys;
    int i;
    ListCell *hash_operator;
    MemoryContext old_context;

    /*
     * Get information about the size of the relation to be hashed (it's the
     * "outer" subtree of this node, but the inner relation of the hash join).
     * Compute the appropriate size of the hash table.
     */
    outer_node = outerPlan(node);

    exec_choose_hash_table_size(outer_node->plan_rows, outer_node->plan_width,
                                OidIsValid(node->skewTable),
                                &num_buckets, &num_batches, &num_skew_mcvs);

    /* num_buckets must be a power of 2 */
    log2_num_buckets = my_log2(num_buckets);
    Assert(num_buckets == (1 << log2_num_buckets));

    /*
     * Initialize the hash table control block.
     *
     * The hash table control block is just allocated from the executor's
     * per-query memory context.
     */
    hash_table = (HashJoinTable) palloc(sizeof(HashJoinTableData));
    hash_table->num_buckets = num_buckets;
    hash_table->num_buckets_original = num_buckets;
    hash_table->num_buckets_optimal = num_buckets;
    hash_table->log2_num_buckets = log2_num_buckets;
    hash_table->log2_num_buckets_optimal = log2_num_buckets;
    hash_table->buckets = NULL;
    hash_table->keep_nulls = keep_nulls;
    hash_table->skew_enabled = false;
    hash_table->skew_bucket = NULL;
    hash_table->skew_bucket_length = 0;
    hash_table->num_skew_buckets = 0;
    hash_table->skew_bucket_numbers = NULL;
    hash_table->num_batches = num_batches;
    hash_table->current_batch = 0;
    hash_table->num_batches_original = num_batches;
    hash_table->num_batches_outstart = num_batches;
    hash_table->growth_enabled = true;
    hash_table->total_tuples = 0;
    hash_table->skew_tuples = 0;
    hash_table->inner_batch_file = NULL;
    hash_table->outer_batch_file = NULL;
    hash_table->space_used = 0;
    hash_table->space_peak = 0;
    hash_table->space_allowed = work_mem * 1024L;
    hash_table->space_used_skew = 0;
    hash_table->space_allowed_skew = hash_table->space_allowed * SKEW_WORK_MEM_PERCENT / 100;
    hash_table->chunks = NULL;

#ifdef HJDEBUG
    printf("Hashjoin %p: initial num_batches = %d, num_buckets = %d\n",
           hash_table, num_batches, num_buckets);
#endif

    /*
     * Get info about the hash functions to be used for each hash key. Also
     * remember whether the join operators are strict.
     */
    num_keys = list_length(hash_operators);
    hash_table->outer_hash_functions = (FmgrInfo *) palloc(num_keys * sizeof(FmgrInfo));
    hash_table->inner_hash_functions = (FmgrInfo *) palloc(num_keys * sizeof(FmgrInfo));
    hash_table->hash_strict = (bool *) palloc(num_keys * sizeof(bool));
    i = 0;
    foreach(hash_operator, hash_operators)
    {
        Oid hash_op = lfirst_oid(hash_operator);
        Oid left_hash_fn;
        Oid right_hash_fn;

        if (!get_op_hash_functions(hash_op, &left_hash_fn, &right_hash_fn))
            elog(ERROR, "could not find hash function for hash operator %u", hash_op);
        fmgr_info(left_hash_fn, &hash_table->outer_hash_functions[i]);
        fmgr_info(right_hash_fn, &hash_table->inner_hash_functions[i]);
        hash_table->hash_strict[i] = op_strict(hash_op);
        i++;
    }

    /*
     * Create temporary memory contexts in which to keep the hash table working
     * storage.  See notes in executor/hashjoin.h.
     */
    hash_table->hash_context = AllocSetContextCreate(CurrentMemoryContext,
                                                     "HashTableContext",
                                                     ALLOCSET_DEFAULT_SIZES);

    hash_table->batch_context = AllocSetContextCreate(hash_table->hash_context,
                                                      "HashBatchContext",
                                                      ALLOCSET_DEFAULT_SIZES);

    /* Allocate data that will live for the life of the hash join */

    old_context = MemoryContextSwitchTo(hash_table->hash_context);

    if (num_batches > 1)
    {
        /*
         * Allocate and initialize the file arrays in hash_context
         */
        hash_table->inner_batch_file = (BufFile **)
            palloc0(num_batches * sizeof(BufFile *));
        hash_table->outer_batch_file = (BufFile **)
            palloc0(num_batches * sizeof(BufFile *));
        /* The files will not be opened until needed... */
        /* ... but make sure we have temp tablespaces established for them */
        prepare_temp_tablespaces();
    }

    /*
     * Prepare context for the first-scan space allocations; allocate the
     * hash bucket array therein, and set each bucket "empty".
     */
    MemoryContextSwitchTo(hash_table->batch_context);

    hash_table->buckets = (HashJoinTuple *)
        palloc0(num_buckets * sizeof(HashJoinTuple));

    /*
     * Set up for skew optimization, if possible and there's a need for more
     * than one batch.  (In a one-batch join, there's no point in it.)
     */
    if (num_batches > 1)
        build_skew_hash_table(hash_table, node, num_skew_mcvs);

    MemoryContextSwitchTo(old_context);

    return hash_table;
}


/* -------------------------------------------------------------------------
 * exec_choose_hash_table_size
 *
 * Computes the optimal size for the hash table based on the estimated number of 
 * tuples and their average width. This ensures efficient memory usage and reduces
 * potential overflows during the join process.
 *
 * Parameters:
 *     ntuples (double): Estimated number of tuples.
 *     tupwidth (int): Average width of each tuple.
 *     useskew (bool): Whether skew optimization is used.
 *     numbuckets (int*): Pointer to store the calculated number of buckets.
 *     numbatches (int*): Pointer to store the calculated number of batches.
 *     num_skew_mcvs (int*): Pointer to store the number of most common values (MCVs) for skew.
 * ------------------------------------------------------------------------- */

/* Target bucket loading (tuples per bucket) */
#define NTUP_PER_BUCKET 1

void
exec_choose_hash_table_size(double ntuples, int tupwidth, bool useskew,
                            int *numbuckets,
                            int *numbatches,
                            int *num_skew_mcvs)
{
    int tupsize;
    double inner_rel_bytes;
    long bucket_bytes;
    long hash_table_bytes;
    long skew_table_bytes;
    long max_pointers;
    long mppow2;
    int nbatch = 1;
    int nbuckets;
    double dbuckets;

    /* Force a plausible relation size if no info is available */
    if (ntuples <= 0.0)
        ntuples = 1000.0;

    /*
     * Estimate tuple size based on the footprint of the tuple in the hash table.
     * Note: This estimate does not account for any palloc overhead.
     * The manipulations of spaceUsed don't count palloc overhead either.
     */
    tupsize = HJTUPLE_OVERHEAD +
              MAXALIGN(SizeofMinimalTupleHeader) +
              MAXALIGN(tupwidth);
    inner_rel_bytes = ntuples * tupsize;

    /*
     * Target in-memory hash table size is work_mem kilobytes.
     */
    hash_table_bytes = work_mem * 1024L;

    /*
     * If skew optimization is possible, estimate the number of skew buckets
     * that will fit in the memory allowed, and decrement the assumed space
     * available for the main hash table accordingly.
     *
     * We optimistically assume each skew bucket will contain one inner-relation tuple.
     * If that turns out to be low, we will recover at runtime by reducing the number of skew buckets.
     */
    if (useskew)
    {
        skew_table_bytes = hash_table_bytes * SKEW_WORK_MEM_PERCENT / 100;

        /*
         * Divisor is:
         * size of a hash tuple +
         * worst-case size of skewBucket[] per MCV +
         * size of skewBucketNums[] entry +
         * size of skew bucket struct itself.
         */
        *num_skew_mcvs = skew_table_bytes / (tupsize +
                                             (8 * sizeof(HashSkewBucket *)) +
                                             sizeof(int) +
                                             SKEW_BUCKET_OVERHEAD);
        if (*num_skew_mcvs > 0)
            hash_table_bytes -= skew_table_bytes;
    }
    else
    {
        *num_skew_mcvs = 0;
    }

    /*
     * Set nbuckets to achieve an average bucket load of NTUP_PER_BUCKET when
     * memory is filled, assuming a single batch; but limit the value so that
     * the pointer arrays we try to allocate do not exceed work_mem or MaxAllocSize.
     *
     * Note that both nbuckets and nbatch must be powers of 2 to make
     * exec_hash_get_bucket_and_batch fast.
     */
    max_pointers = (work_mem * 1024L) / sizeof(HashJoinTuple);
    max_pointers = Min(max_pointers, MaxAllocSize / sizeof(HashJoinTuple));

    /* If max_pointers isn't a power of 2, round it down to the nearest power of 2 */
    mppow2 = 1L << my_log2(max_pointers);
    if (max_pointers != mppow2)
        max_pointers = mppow2 / 2;

    /* Also ensure we avoid integer overflow in nbatch and nbuckets */
    max_pointers = Min(max_pointers, INT_MAX / 2);

    dbuckets = ceil(ntuples / NTUP_PER_BUCKET);
    dbuckets = Min(dbuckets, max_pointers);
    nbuckets = (int)dbuckets;

    /* Ensure nbuckets is not too small */
    nbuckets = Max(nbuckets, 1024);
    /* Force nbuckets to be a power of 2 */
    nbuckets = 1 << my_log2(nbuckets);

    /*
     * If there's not enough space to store the projected number of tuples and
     * the required bucket headers, we will need multiple batches.
     */
    bucket_bytes = sizeof(HashJoinTuple) * nbuckets;
    if (inner_rel_bytes + bucket_bytes > hash_table_bytes)
    {
        /* Multiple batches are required */
        long lbuckets;
        double dbatch;
        int minbatch;
        long bucket_size;

        /*
         * Estimate the number of buckets we want when work_mem is entirely full.
         * Each bucket will contain a bucket pointer plus NTUP_PER_BUCKET tuples.
         */
        bucket_size = (tupsize * NTUP_PER_BUCKET + sizeof(HashJoinTuple));
        lbuckets = 1L << my_log2(hash_table_bytes / bucket_size);
        lbuckets = Min(lbuckets, max_pointers);
        nbuckets = (int)lbuckets;
        nbuckets = 1 << my_log2(nbuckets);
        bucket_bytes = nbuckets * sizeof(HashJoinTuple);

        /*
         * Buckets are simple pointers to hash join tuples, while tupsize
         * includes the pointer, hash code, and MinimalTupleData. 
         * Ensure buckets do not exceed 50% of work_mem.
         */
        Assert(bucket_bytes <= hash_table_bytes / 2);

        /* Calculate the required number of batches */
        dbatch = ceil(inner_rel_bytes / (hash_table_bytes - bucket_bytes));
        dbatch = Min(dbatch, max_pointers);
        minbatch = (int)dbatch;
        nbatch = 2;
        while (nbatch < minbatch)
            nbatch <<= 1;
    }

    Assert(nbuckets > 0);
    Assert(nbatch > 0);

    *numbuckets = nbuckets;
    *numbatches = nbatch;
}



/* ----------------------------------------------------------------
 *		exec_hash
TableDestroy
 *
 *		destroy a hash table
 * ----------------------------------------------------------------
 */
void
exec_hashTableDestroy(HashJoinTable hashtable)
{
	int i;

	/*
	 * Make sure all the temp files are closed.  We skip batch 0, since it
	 * can't have any temp files (and the arrays might not even exist if
	 * nbatch is only 1).
	 */
	for (i = 1; i < hashtable->nbatch; i++)
	{
		if (hashtable->innerBatchFile[i])
			BufFileClose(hashtable->innerBatchFile[i]);
		if (hashtable->outerBatchFile[i])
			BufFileClose(hashtable->outerBatchFile[i]);
	}

	/* Release working memory (batchCxt is a child, so it goes away too) */
	MemoryContextDelete(hashtable->hashCxt);

	/* And drop the control block */
	pfree(hashtable);
}

/* --------------------------------------------------------------
 * hash_table_increase_batch_count
 *		increase the original number of batches in order to reduce
 *		current memory consumption
 * -------------------------------------------------------------- */
static void
hash_table_increase_batch_count(HashJoinTable hashtable)
{
    int oldnbatch = hashtable->nbatch;
    int curbatch = hashtable->curbatch;
    int nbatch;
    MemoryContext oldcxt;
    long ninmemory;
    long nfreed;
    HashMemoryChunk oldchunks;

    /* do nothing if we've decided to shut off growth */
    if (!hashtable->growEnabled)
        return;

    /* safety check to avoid overflow */
    if (oldnbatch > Min(INT_MAX / 2, MaxAllocSize / (sizeof(void *) * 2)))
        return;

    nbatch = oldnbatch * 2;
    Assert(nbatch > 1);

#ifdef HJDEBUG
    printf("Hashjoin %p: increasing nbatch to %d because space = %zu\n",
           hashtable, nbatch, hashtable->spaceUsed);
#endif

    oldcxt = MemoryContextSwitchTo(hashtable->hashCxt);

    if (hashtable->innerBatchFile == NULL)
    {
        /* we had no file arrays before */
        hashtable->innerBatchFile = (BufFile **)
            palloc0(nbatch * sizeof(BufFile *));
        hashtable->outerBatchFile = (BufFile **)
            palloc0(nbatch * sizeof(BufFile *));
        /* time to establish the temp tablespaces, too */
        PrepareTempTablespaces();
    }
    else
    {
        /* enlarge arrays and zero out added entries */
        hashtable->innerBatchFile = (BufFile **)
            repalloc(hashtable->innerBatchFile, nbatch * sizeof(BufFile *));
        hashtable->outerBatchFile = (BufFile **)
            repalloc(hashtable->outerBatchFile, nbatch * sizeof(BufFile *));
        MemSet(hashtable->innerBatchFile + oldnbatch, 0,
               (nbatch - oldnbatch) * sizeof(BufFile *));
        MemSet(hashtable->outerBatchFile + oldnbatch, 0,
               (nbatch - oldnbatch) * sizeof(BufFile *));
    }

    MemoryContextSwitchTo(oldcxt);

    hashtable->nbatch = nbatch;

    /*
     * Scan through the existing hash table entries and dump out any that are
     * no longer of the current batch.
     */
    ninmemory = nfreed = 0;

    /* If know we need to resize nbuckets, we can do it while rebatching. */
    if (hashtable->nbuckets_optimal != hashtable->nbuckets)
    {
        /* we never decrease the number of buckets */
        Assert(hashtable->nbuckets_optimal > hashtable->nbuckets);

        hashtable->nbuckets = hashtable->nbuckets_optimal;
        hashtable->log2_nbuckets = hashtable->log2_nbuckets_optimal;

        hashtable->buckets = repalloc(hashtable->buckets,
                                      sizeof(HashJoinTuple) * hashtable->nbuckets);
    }

    /*
     * We will scan through the chunks directly, so that we can reset the
     * buckets now and not have to keep track which tuples in the buckets have
     * already been processed. We will free the old chunks as we go.
     */
    memset(hashtable->buckets, 0, sizeof(HashJoinTuple) * hashtable->nbuckets);
    oldchunks = hashtable->chunks;
    hashtable->chunks = NULL;

    /* so, let's scan through the old chunks, and all tuples in each chunk */
    while (oldchunks != NULL)
    {
        HashMemoryChunk nextchunk = oldchunks->next;

        /* position within the buffer (up to oldchunks->used) */
        size_t idx = 0;

        /* process all tuples stored in this chunk (and then free it) */
        while (idx < oldchunks->used)
        {
            HashJoinTuple hashTuple = (HashJoinTuple) (oldchunks->data + idx);
            MinimalTuple tuple = HJTUPLE_MINTUPLE(hashTuple);
            int hashTupleSize = (HJTUPLE_OVERHEAD + tuple->t_len);
            int bucketno;
            int batchno;

            ninmemory++;
            exec_hashGetBucketAndBatch(hashtable, hashTuple->hashvalue,
                                      &bucketno, &batchno);

            if (batchno == curbatch)
            {
                /* keep tuple in memory - copy it into the new chunk */
                HashJoinTuple copyTuple;

                copyTuple = (HashJoinTuple) dense_alloc(hashtable, hashTupleSize);
                memcpy(copyTuple, hashTuple, hashTupleSize);

                /* and add it back to the appropriate bucket */
                copyTuple->next = hashtable->buckets[bucketno];
                hashtable->buckets[bucketno] = copyTuple;
            }
            else
            {
                /* dump it out */
                Assert(batchno > curbatch);
                exec_hashJoinSaveTuple(HJTUPLE_MINTUPLE(hashTuple),
                                      hashTuple->hashvalue,
                                      &hashtable->innerBatchFile[batchno]);

                hashtable->spaceUsed -= hashTupleSize;
                nfreed++;
            }

            /* next tuple in this chunk */
            idx += MAXALIGN(hashTupleSize);

            /* allow this loop to be cancellable */
            CHECK_FOR_INTERRUPTS();
        }

        /* we're done with this chunk - free it and proceed to the next one */
        pfree(oldchunks);
        oldchunks = nextchunk;
    }

#ifdef HJDEBUG
    printf("Hashjoin %p: freed %ld of %ld tuples, space now %zu\n",
           hashtable, nfreed, ninmemory, hashtable->spaceUsed);
#endif

    /*
     * If we dumped out either all or none of the tuples in the table, disable
     * further expansion of nbatch.  This situation implies that we have
     * enough tuples of identical hashvalues to overflow spaceAllowed.
     * Increasing nbatch will not fix it since there's no way to subdivide the
     * group any more finely. We have to just gut it out and hope the server
     * has enough RAM.
     */
    if (nfreed == 0 || nfreed == ninmemory)
    {
        hashtable->growEnabled = false;
#ifdef HJDEBUG
        printf("Hashjoin %p: disabling further increase of nbatch\n",
               hashtable);
#endif
    }
}

/*
 * hash_table_increase_bucket_count
 *		increase the original number of buckets in order to reduce
 *		number of tuples per bucket
 */
static void
hash_table_increase_bucket_count(HashJoinTable hashtable)
{
	HashMemoryChunk chunk;

	/* do nothing if not an increase (it's called increase for a reason) */
	if (hashtable->nbuckets >= hashtable->nbuckets_optimal)
		return;

#ifdef HJDEBUG
	printf("Hashjoin %p: increasing nbuckets %d => %d\n",
		   hashtable, hashtable->nbuckets, hashtable->nbuckets_optimal);
#endif

	hashtable->nbuckets = hashtable->nbuckets_optimal;
	hashtable->log2_nbuckets = hashtable->log2_nbuckets_optimal;

	Assert(hashtable->nbuckets > 1);
	Assert(hashtable->nbuckets <= (INT_MAX / 2));
	Assert(hashtable->nbuckets == (1 << hashtable->log2_nbuckets));

	/*
	 * Just reallocate the proper number of buckets - we don't need to walk
	 * through them - we can walk the dense-allocated chunks (just like in
	 * hash_table_increase_batch_count, but without all the copying into new
	 * chunks)
	 */
	hashtable->buckets =
		(HashJoinTuple *) repalloc(hashtable->buckets,
								   hashtable->nbuckets * sizeof(HashJoinTuple));

	memset(hashtable->buckets, 0, hashtable->nbuckets * sizeof(HashJoinTuple));

	/* scan through all tuples in all chunks to rebuild the hash table */
	for (chunk = hashtable->chunks; chunk != NULL; chunk = chunk->next)
	{
		/* process all tuples stored in this chunk */
		size_t		idx = 0;

		while (idx < chunk->used)
		{
			HashJoinTuple hashTuple = (HashJoinTuple) (chunk->data + idx);
			int			bucketno;
			int			batchno;

			exec_hash
		GetBucketAndBatch(hashtable, hashTuple->hashvalue,
									  &bucketno, &batchno);

			/* add the tuple to the proper bucket */
			hashTuple->next = hashtable->buckets[bucketno];
			hashtable->buckets[bucketno] = hashTuple;

			/* advance index past the tuple */
			idx += MAXALIGN(HJTUPLE_OVERHEAD +
							HJTUPLE_MINTUPLE(hashTuple)->t_len);
		}

		/* allow this loop to be cancellable */
		CHECK_FOR_INTERRUPTS();
	}
}


/*
 * exec_hash
TableInsert
 *		insert a tuple into the hash table depending on the hash value
 *		it may just go to a temp file for later batches
 *
 * Note: the passed TupleTableSlot may contain a regular, minimal, or virtual
 * tuple; the minimal case in particular is certain to happen while reloading
 * tuples from batch files.  We could save some cycles in the regular-tuple
 * case by not forcing the slot contents into minimal form; not clear if it's
 * worth the messiness required.
 */
void
exec_hashTableInsert(HashJoinTable hashtable,
					TupleTableSlot *slot,
					uint32 hashvalue)
{
	MinimalTuple tuple = ExecFetchSlotMinimalTuple(slot);
	int			bucketno;
	int			batchno;

	exec_hash
GetBucketAndBatch(hashtable, hashvalue,
							  &bucketno, &batchno);

	/*
	 * decide whether to put the tuple in the hash table or a temp file
	 */
	if (batchno == hashtable->curbatch)
	{
		/*
		 * put the tuple in hash table
		 */
		HashJoinTuple hashTuple;
		int			hashTupleSize;
		double		ntuples = (hashtable->totalTuples - hashtable->skewTuples);

		/* Create the HashJoinTuple */
		hashTupleSize = HJTUPLE_OVERHEAD + tuple->t_len;
		hashTuple = (HashJoinTuple) dense_alloc(hashtable, hashTupleSize);

		hashTuple->hashvalue = hashvalue;
		memcpy(HJTUPLE_MINTUPLE(hashTuple), tuple, tuple->t_len);

		/*
		 * We always reset the tuple-matched flag on insertion.  This is okay
		 * even when reloading a tuple from a batch file, since the tuple
		 * could not possibly have been matched to an outer tuple before it
		 * went into the batch file.
		 */
		HeapTupleHeaderClearMatch(HJTUPLE_MINTUPLE(hashTuple));

		/* Push it onto the front of the bucket's list */
		hashTuple->next = hashtable->buckets[bucketno];
		hashtable->buckets[bucketno] = hashTuple;

		/*
		 * Increase the (optimal) number of buckets if we just exceeded the
		 * NTUP_PER_BUCKET threshold, but only when there's still a single
		 * batch.
		 */
		if (hashtable->nbatch == 1 &&
			ntuples > (hashtable->nbuckets_optimal * NTUP_PER_BUCKET))
		{
			/* Guard against integer overflow and alloc size overflow */
			if (hashtable->nbuckets_optimal <= INT_MAX / 2 &&
				hashtable->nbuckets_optimal * 2 <= MaxAllocSize / sizeof(HashJoinTuple))
			{
				hashtable->nbuckets_optimal *= 2;
				hashtable->log2_nbuckets_optimal += 1;
			}
		}

		/* Account for space used, and back off if we've used too much */
		hashtable->spaceUsed += hashTupleSize;
		if (hashtable->spaceUsed > hashtable->spacePeak)
			hashtable->spacePeak = hashtable->spaceUsed;
		if (hashtable->spaceUsed +
			hashtable->nbuckets_optimal * sizeof(HashJoinTuple)
			> hashtable->spaceAllowed)
			hash_table_increase_batch_count(hashtable);
	}
	else
	{
		/*
		 * put the tuple into a temp file for later batches
		 */
		Assert(batchno > hashtable->curbatch);
		exec_hash
	JoinSaveTuple(tuple,
							  hashvalue,
							  &hashtable->innerBatchFile[batchno]);
	}
}

/*
 * exec_hash
GetHashValue
 *		Compute the hash value for a tuple
 *
 * The tuple to be tested must be in either econtext->ecxt_outertuple or
 * econtext->ecxt_innertuple.  Vars in the hashkeys expressions should have
 * varno either OUTER_VAR or INNER_VAR.
 *
 * A true result means the tuple's hash value has been successfully computed
 * and stored at *hashvalue.  A false result means the tuple cannot match
 * because it contains a null attribute, and hence it should be discarded
 * immediately.  (If keep_nulls is true then false is never returned.)
 */
bool
exec_hashGetHashValue(HashJoinTable hashtable,
					 ExprContext *econtext,
					 List *hashkeys,
					 bool outer_tuple,
					 bool keep_nulls,
					 uint32 *hashvalue)
{
	uint32		hashkey = 0;
	FmgrInfo   *hashfunctions;
	ListCell   *hk;
	int			i = 0;
	MemoryContext oldContext;

	/*
	 * We reset the eval context each time to reclaim any memory leaked in the
	 * hashkey expressions.
	 */
	ResetExprContext(econtext);

	oldContext = MemoryContextSwitchTo(econtext->ecxt_per_tuple_memory);

	if (outer_tuple)
		hashfunctions = hashtable->outer_hashfunctions;
	else
		hashfunctions = hashtable->inner_hashfunctions;

	foreach(hk, hashkeys)
	{
		ExprState  *keyexpr = (ExprState *) lfirst(hk);
		Datum		keyval;
		bool		isNull;

		/* rotate hashkey left 1 bit at each step */
		hashkey = (hashkey << 1) | ((hashkey & 0x80000000) ? 1 : 0);

		/*
		 * Get the join attribute value of the tuple
		 */
		keyval = ExecEvalExpr(keyexpr, econtext, &isNull);

		/*
		 * If the attribute is NULL, and the join operator is strict, then
		 * this tuple cannot pass the join qual so we can reject it
		 * immediately (unless we're scanning the outside of an outer join, in
		 * which case we must not reject it).  Otherwise we act like the
		 * hashcode of NULL is zero (this will support operators that act like
		 * IS NOT DISTINCT, though not any more-random behavior).  We treat
		 * the hash support function as strict even if the operator is not.
		 *
		 * Note: currently, all hashjoinable operators must be strict since
		 * the hash index AM assumes that.  However, it takes so little extra
		 * code here to allow non-strict that we may as well do it.
		 */
		if (isNull)
		{
			if (hashtable->hashStrict[i] && !keep_nulls)
			{
				MemoryContextSwitchTo(oldContext);
				return false;	/* cannot match */
			}
			/* else, leave hashkey unmodified, equivalent to hashcode 0 */
		}
		else
		{
			/* Compute the hash function */
			uint32		hkey;

			hkey = DatumGetUInt32(FunctionCall1(&hashfunctions[i], keyval));
			hashkey ^= hkey;
		}

		i++;
	}

	MemoryContextSwitchTo(oldContext);

	*hashvalue = hashkey;
	return true;
}

/*
 * exec_hash
GetBucketAndBatch
 *		Determine the bucket number and batch number for a hash value
 *
 * Note: on-the-fly increases of nbatch must not change the bucket number
 * for a given hash code (since we don't move tuples to different hash
 * chains), and must only cause the batch number to remain the same or
 * increase.  Our algorithm is
 *		bucketno = hashvalue MOD nbuckets
 *		batchno = (hashvalue DIV nbuckets) MOD nbatch
 * where nbuckets and nbatch are both expected to be powers of 2, so we can
 * do the computations by shifting and masking.  (This assumes that all hash
 * functions are good about randomizing all their output bits, else we are
 * likely to have very skewed bucket or batch occupancy.)
 *
 * nbuckets and log2_nbuckets may change while nbatch == 1 because of dynamic
 * bucket count growth.  Once we start batching, the value is fixed and does
 * not change over the course of the join (making it possible to compute batch
 * number the way we do here).
 *
 * nbatch is always a power of 2; we increase it only by doubling it.  This
 * effectively adds one more bit to the top of the batchno.
 */
void
exec_hashGetBucketAndBatch(HashJoinTable hashtable,
						  uint32 hashvalue,
						  int *bucketno,
						  int *batchno)
{
	uint32		nbuckets = (uint32) hashtable->nbuckets;
	uint32		nbatch = (uint32) hashtable->nbatch;

	if (nbatch > 1)
	{
		/* we can do MOD by masking, DIV by shifting */
		*bucketno = hashvalue & (nbuckets - 1);
		*batchno = (hashvalue >> hashtable->log2_nbuckets) & (nbatch - 1);
	}
	else
	{
		*bucketno = hashvalue & (nbuckets - 1);
		*batchno = 0;
	}
}
/* -------------------------------------------------------------------------
 * exec_scan_hash_bucket
 *
 * Scans a hash bucket for matches to the current outer tuple.
 *
 * The current outer tuple must be stored in econtext->ecxt_outertuple.
 * On success, the inner tuple is stored into hjstate->hj_CurTuple and
 * econtext->ecxt_innertuple, using hjstate->hj_HashTupleSlot as the slot
 * for the latter.
 *
 * Parameters:
 *     hjstate (HashJoinState*): State information for the hash join operation.
 *     econtext (ExprContext*): Expression context holding outer and inner tuples.
 *
 * Returns:
 *     bool: True if a match is found, otherwise false.
 * ------------------------------------------------------------------------- */
bool
exec_scan_hash_bucket(HashJoinState *hjstate,
                      ExprContext *econtext)
{
    ExprState *hjclauses = hjstate->hashclauses;
    HashJoinTable hashtable = hjstate->inner_hj_HashTable;
    HashJoinTuple hashTuple = hjstate->inner_hj_CurTuple;
    uint32 hashvalue = hjstate->inner_hj_CurHashValue;

    /*
     * hj_CurTuple is the address of the tuple last returned from the current
     * bucket, or NULL if it's time to start scanning a new bucket.
     *
     * If the tuple hashed to a skew bucket then scan the skew bucket
     * otherwise scan the standard hashtable bucket.
     */
    if (hashTuple != NULL)
        hashTuple = hashTuple->next;
    else if (hjstate->outer_hj_CurBucketNo != INVALID_SKEW_BUCKET_NO)
        hashTuple = hashtable->skewBucket[hjstate->outer_hj_CurBucketNo]->tuples;
    else
        hashTuple = hashtable->buckets[hjstate->outer_hj_CurBucketNo];

    while (hashTuple != NULL)
    {
        if (hashTuple->hashvalue == hashvalue)
        {
            TupleTableSlot *inntuple;

            /* Insert hashtable's tuple into exec slot so ExecQual sees it */
            inntuple = ExecStoreMinimalTuple(HJTUPLE_MINTUPLE(hashTuple),
                                             hjstate->hj_OuterTupleSlot,
                                             false); /* Do not pfree */
            econtext->ecxt_innertuple = inntuple;

            /* Reset temp memory each time to avoid leaks from qual expr */
            ResetExprContext(econtext);

            if (ExecQual(hjclauses, econtext))
            {
                hjstate->inner_hj_CurTuple = hashTuple;
                return true;
            }
        }

        hashTuple = hashTuple->next;
    }

    /* No match found */
    return false;
}

/* -------------------------------------------------------------------------
 * exec_prep_hash_table_for_unmatched
 *
 * Prepares for a series of exec_scan_hash_table_for_unmatched calls by resetting
 * state to the beginning of the hash table and skew buckets.
 *
 * Parameters:
 *     hjstate (HashJoinState*): State information for the hash join operation.
 * ------------------------------------------------------------------------- */
void
exec_prep_hash_table_for_unmatched(HashJoinState *hjstate)
{
    /*----------
     * During this scan we use the HashJoinState fields as follows:
     *
     * hj_CurBucketNo: next regular bucket to scan
     * hj_CurSkewBucketNo: next skew bucket (an index into skewBucketNums)
     * hj_CurTuple: last tuple returned, or NULL to start next bucket
     *----------
     */
    hjstate->outer_hj_CurBucketNo = 0;
    hjstate->outer_hj_CurBucketNo = 0;
    hjstate->inner_hj_CurTuple = NULL;
}

/* -------------------------------------------------------------------------
 * exec_scan_hash_table_for_unmatched
 *
 * Scans the hash table for unmatched inner tuples.
 * On success, the inner tuple is stored into hjstate->hj_CurTuple and
 * econtext->ecxt_innertuple, using hjstate->hj_HashTupleSlot as the slot
 * for the latter.
 *
 * Parameters:
 *     hjstate (HashJoinState*): State information for the hash join operation.
 *     econtext (ExprContext*): Expression context holding outer and inner tuples.
 *
 * Returns:
 *     bool: True if an unmatched tuple is found, otherwise false.
 * ------------------------------------------------------------------------- */
bool
exec_scan_hash_table_for_unmatched(HashJoinState *hjstate, ExprContext *econtext)
{
    HashJoinTable hashtable = hjstate->inner_hj_HashTable;
    HashJoinTuple hashTuple = hjstate->inner_hj_CurTuple;

    for (;;)
    {
        /*
         * hj_CurTuple is the address of the tuple last returned from the
         * current bucket, or NULL if it's time to start scanning a new
         * bucket.
         */
        if (hashTuple != NULL)
            hashTuple = hashTuple->next;
        else if (hjstate->outer_hj_CurBucketNo < hashtable->nbuckets)
        {
            hashTuple = hashtable->buckets[hjstate->outer_hj_CurBucketNo];
            hjstate->outer_hj_CurBucketNo++;
        }
        else if (hjstate->outer_hj_CurBucketNo < hashtable->nSkewBuckets)
        {
            int j = hashtable->skewBucketNums[hjstate->outer_hj_CurBucketNo];
            hashTuple = hashtable->skewBucket[j]->tuples;
            hjstate->outer_hj_CurBucketNo++;
        }
        else
            break; /* Finished all buckets */

        while (hashTuple != NULL)
        {
            if (!HeapTupleHeaderHasMatch(HJTUPLE_MINTUPLE(hashTuple)))
            {
                TupleTableSlot *inntuple;

                /* Insert hashtable's tuple into exec slot */
                inntuple = ExecStoreMinimalTuple(HJTUPLE_MINTUPLE(hashTuple),
                                                 hjstate->hj_OuterTupleSlot,
                                                 false); /* Do not pfree */
                econtext->ecxt_innertuple = inntuple;

                /*
                 * Reset temp memory each time; although this function doesn't
                 * do any qual eval, the caller will, so let's keep it
                 * parallel to exec_scan_hash_bucket.
                 */
                ResetExprContext(econtext);

                hjstate->inner_hj_CurTuple = hashTuple;
                return true;
            }

            hashTuple = hashTuple->next;
        }

        /* Allow this loop to be cancellable */
        CHECK_FOR_INTERRUPTS();
    }

    /* No more unmatched tuples */
    return false;
}

/*
 * exec_hash
TableReset
 *
 *		reset hash table header for new batch
 */
void
exec_hashTableReset(HashJoinTable hashtable)
{
	MemoryContext oldcxt;
	int			nbuckets = hashtable->nbuckets;

	/*
	 * Release all the hash buckets and tuples acquired in the prior pass, and
	 * reinitialize the context for a new pass.
	 */
	MemoryContextReset(hashtable->batchCxt);
	oldcxt = MemoryContextSwitchTo(hashtable->batchCxt);

	/* Reallocate and reinitialize the hash bucket headers. */
	hashtable->buckets = (HashJoinTuple *)
		palloc0(nbuckets * sizeof(HashJoinTuple));

	hashtable->spaceUsed = 0;

	MemoryContextSwitchTo(oldcxt);

	/* Forget the chunks (the memory was freed by the context reset above). */
	hashtable->chunks = NULL;
}

/*
 * exec_hash
TableResetMatchFlags
 *		Clear all the HeapTupleHeaderHasMatch flags in the table
 */
void
exec_hashTableResetMatchFlags(HashJoinTable hashtable)
{
	HashJoinTuple tuple;
	int			i;

	/* Reset all flags in the main table ... */
	for (i = 0; i < hashtable->nbuckets; i++)
	{
		for (tuple = hashtable->buckets[i]; tuple != NULL; tuple = tuple->next)
			HeapTupleHeaderClearMatch(HJTUPLE_MINTUPLE(tuple));
	}

	/* ... and the same for the skew buckets, if any */
	for (i = 0; i < hashtable->nSkewBuckets; i++)
	{
		int			j = hashtable->skewBucketNums[i];
		HashSkewBucket *skewBucket = hashtable->skewBucket[j];

		for (tuple = skewBucket->tuples; tuple != NULL; tuple = tuple->next)
			HeapTupleHeaderClearMatch(HJTUPLE_MINTUPLE(tuple));
	}
}


/* -------------------------------------------------------------------------
 * exec_rescan_hash
 *
 * Rescans the hash table.
 *
 * If the chgParam of the subnode is not NULL, the plan will be re-scanned by
 * calling ExecProcNode.
 *
 * Parameters:
 *     node (HashState*): State information for the hash operation.
 * ------------------------------------------------------------------------- */
void
exec_rescan_hash(HashState *node)
{
    /*
     * If chgParam of subnode is not null, then plan will be re-scanned by
     * the first ExecProcNode.
     */
    if (node->ps.lefttree->chgParam == NULL)
        ExecReScan(node->ps.lefttree);
}

/* -------------------------------------------------------------------------
 * build_skew_hash_table
 *
 * Sets up skew optimization by identifying the most common values (MCVs) of the
 * outer relation's join key. It creates a skew hash bucket for each MCV hash value,
 * subject to the number of available slots based on memory.
 *
 * Parameters:
 *     hashtable (HashJoinTable): Hash join table where skew buckets will be stored.
 *     node (Hash*): Hash node containing information about the join key.
 *     mcvsToUse (int): Number of most common values to use for skew optimization.
 * ------------------------------------------------------------------------- */
static void
build_skew_hash_table(HashJoinTable hashtable, Hash *node, int mcvsToUse)
{
    HeapTupleData *statsTuple;
    AttStatsSlot sslot;

    /* Do nothing if planner didn't identify the outer relation's join key */
    if (!OidIsValid(node->skewTable))
        return;
    /* Also, do nothing if we don't have room for at least one skew bucket */
    if (mcvsToUse <= 0)
        return;

    /*
     * Try to find the MCV statistics for the outer relation's join key.
     */
    statsTuple = SearchSysCache3(STATRELATTINH,
                                 ObjectIdGetDatum(node->skewTable),
                                 Int16GetDatum(node->skewColumn),
                                 BoolGetDatum(node->skewInherit));
    if (!HeapTupleIsValid(statsTuple))
        return;

    if (get_attstatsslot(&sslot, statsTuple,
                         STATISTIC_KIND_MCV, InvalidOid,
                         ATTSTATSSLOT_VALUES | ATTSTATSSLOT_NUMBERS))
    {
        double frac;
        int nbuckets;
        FmgrInfo *hashfunctions;
        int i;

        if (mcvsToUse > sslot.nvalues)
            mcvsToUse = sslot.nvalues;

        /*
         * Calculate the expected fraction of the outer relation that will
         * participate in the skew optimization. If this fraction isn't at least
         * SKEW_MIN_OUTER_FRACTION, we won't use skew optimization.
         */
        frac = 0;
        for (i = 0; i < mcvsToUse; i++)
            frac += sslot.numbers[i];
        if (frac < SKEW_MIN_OUTER_FRACTION)
        {
            free_attstatsslot(&sslot);
            ReleaseSysCache(statsTuple);
            return;
        }

        /*
         * Set up the skew hashtable.
         * skewBucket[] is an open addressing hashtable with a power of 2 size
         * that is greater than the number of MCV values. This ensures there
         * will be at least one null entry, so searches will always terminate.
         */
        nbuckets = 2;
        while (nbuckets <= mcvsToUse)
            nbuckets <<= 1;
        /* Use two more bits to help avoid collisions */
        nbuckets <<= 2;

        hashtable->skewEnabled = true;
        hashtable->skewBucketLen = nbuckets;

        /*
         * Allocate the bucket memory in the hashtable's batch context. This
         * memory is only needed during the first batch, and this ensures it
         * will be automatically released once the first batch is done.
         */
        hashtable->skewBucket = (HashSkewBucket **)
            MemoryContextAllocZero(hashtable->batchCxt,
                                   nbuckets * sizeof(HashSkewBucket *));
        hashtable->skewBucketNums = (int *)
            MemoryContextAllocZero(hashtable->batchCxt,
                                   mcvsToUse * sizeof(int));

        hashtable->spaceUsed += nbuckets * sizeof(HashSkewBucket *) +
                                mcvsToUse * sizeof(int);
        hashtable->spaceUsedSkew += nbuckets * sizeof(HashSkewBucket *) +
                                    mcvsToUse * sizeof(int);
        if (hashtable->spaceUsed > hashtable->spacePeak)
            hashtable->spacePeak = hashtable->spaceUsed;

        /*
         * Create a skew bucket for each MCV hash value.
         *
         * Note: Buckets must be created in order of decreasing MCV frequency.
         * If we have to remove some buckets, they must be removed in reverse
         * order of creation, with the least common MCVs removed first.
         */
        hashfunctions = hashtable->outer_hashfunctions;

        for (i = 0; i < mcvsToUse; i++)
        {
            uint32 hashvalue;
            int bucket;

            hashvalue = DatumGetUInt32(FunctionCall1(&hashfunctions[0],
                                                     sslot.values[i]));

            /*
             * If we encounter a collision, try the next bucket location until
             * we find an empty slot or the desired bucket. This code must match
             * exec_hash_get_skew_bucket.
             */
            bucket = hashvalue & (nbuckets - 1);
            while (hashtable->skewBucket[bucket] != NULL &&
                   hashtable->skewBucket[bucket]->hashvalue != hashvalue)
                bucket = (bucket + 1) & (nbuckets - 1);

            /*
             * If we found an existing bucket with the same hashvalue, leave it
             * alone. It is acceptable for multiple MCVs to share a hashvalue.
             */
            if (hashtable->skewBucket[bucket] != NULL)
                continue;

            /* Create a new skew bucket for this hashvalue. */
            hashtable->skewBucket[bucket] = (HashSkewBucket *)
                MemoryContextAlloc(hashtable->batchCxt,
                                   sizeof(HashSkewBucket));
            hashtable->skewBucket[bucket]->hashvalue = hashvalue;
            hashtable->skewBucket[bucket]->tuples = NULL;
            hashtable->skewBucketNums[hashtable->nSkewBuckets] = bucket;
            hashtable->nSkewBuckets++;
            hashtable->spaceUsed += SKEW_BUCKET_OVERHEAD;
            hashtable->spaceUsedSkew += SKEW_BUCKET_OVERHEAD;
            if (hashtable->spaceUsed > hashtable->spacePeak)
                hashtable->spacePeak = hashtable->spaceUsed;
        }

        free_attstatsslot(&sslot);
    }

    ReleaseSysCache(statsTuple);
}

/* -------------------------------------------------------------------------
 * exec_hash_get_skew_bucket
 *
 * Returns the index of the skew bucket for the given hash value,
 * or INVALID_SKEW_BUCKET_NO if the hash value is not associated with any
 * active skew bucket.
 *
 * Parameters:
 *     hashtable (HashJoinTable): The hash join table containing skew buckets.
 *     hashvalue (uint32): The hash value for which to find the skew bucket.
 *
 * Returns:
 *     int: Index of the skew bucket, or INVALID_SKEW_BUCKET_NO if not found.
 * ------------------------------------------------------------------------- */
int
exec_hash_get_skew_bucket(HashJoinTable hashtable, uint32 hashvalue)
{
    int bucket;

    /*
     * Return INVALID_SKEW_BUCKET_NO if skew optimization is disabled.
     * This typically occurs after the initial batch is complete.
     */
    if (!hashtable->skewEnabled)
        return INVALID_SKEW_BUCKET_NO;

    /*
     * Since skewBucketLen is a power of 2, use bitwise AND for modulo.
     */
    bucket = hashvalue & (hashtable->skewBucketLen - 1);

    /*
     * If there's a collision, try the next bucket location until an empty
     * slot or the desired bucket is found.
     */
    while (hashtable->skewBucket[bucket] != NULL &&
           hashtable->skewBucket[bucket]->hashvalue != hashvalue)
        bucket = (bucket + 1) & (hashtable->skewBucketLen - 1);

    /*
     * If the desired bucket is found, return its index.
     */
    if (hashtable->skewBucket[bucket] != NULL)
        return bucket;

    /*
     * No matching skew bucket found for this hash value.
     */
    return INVALID_SKEW_BUCKET_NO;
}

/* -------------------------------------------------------------------------
 * insert_into_skew_table
 *
 *     Inserts a tuple into the skew hashtable for most common value (MCV) 
 *     optimization. If the space used exceeds the allowed threshold, it 
 *     triggers removal of the least valuable skew buckets to keep within limits.
 *
 * Parameters:
 *     hashtable (HashJoinTable): The hash join table containing skew buckets.
 *     slot (TupleTableSlot*): The slot containing the tuple to insert.
 *     hashvalue (uint32): The hash value of the tuple.
 *     bucketNumber (int): The skew bucket to insert the tuple into.
 * ------------------------------------------------------------------------- */
static void
insert_into_skew_table(HashJoinTable hashtable,
                        TupleTableSlot *slot,
                        uint32 hashvalue,
                        int bucketNumber)
{
    MinimalTuple tuple = ExecFetchSlotMinimalTuple(slot);
    HashJoinTuple hashTuple;
    int hashTupleSize;

    /* Create the HashJoinTuple */
    hashTupleSize = MAXALIGN(HJTUPLE_OVERHEAD + tuple->t_len);
    hashTuple = (HashJoinTuple) MemoryContextAlloc(hashtable->batchCxt, hashTupleSize);
    hashTuple->hashvalue = hashvalue;
    memcpy(HJTUPLE_MINTUPLE(hashTuple), tuple, tuple->t_len);
    HeapTupleHeaderClearMatch(HJTUPLE_MINTUPLE(hashTuple));

    /* Push it onto the front of the skew bucket's list */
    hashTuple->next = hashtable->skewBucket[bucketNumber]->tuples;
    hashtable->skewBucket[bucketNumber]->tuples = hashTuple;

    /* Account for space used, and back off if we've used too much */
    hashtable->spaceUsed += hashTupleSize;
    hashtable->spaceUsedSkew += hashTupleSize;
    if (hashtable->spaceUsed > hashtable->spacePeak)
        hashtable->spacePeak = hashtable->spaceUsed;

    /* Log insertion into skew table */
#ifdef HJDEBUG
    printf("Hashjoin %p: inserted tuple into skew bucket %d, hash value %u\n", 
           hashtable, bucketNumber, hashvalue);
#endif

    /* Remove skew buckets if space used for skew exceeds allowed threshold */
    while (hashtable->spaceUsedSkew > hashtable->spaceAllowedSkew)
        remove_next_skew_bucket(hashtable);

    /* Check if overall space used exceeds the allowed threshold */
    if (hashtable->spaceUsed > hashtable->spaceAllowed)
        hash_table_increase_batch_count(hashtable);
}

/* -------------------------------------------------------------------------
 * remove_next_skew_bucket
 *
 *     Removes the least valuable skew bucket by moving its tuples into the
 *     main hash table. This ensures we do not exceed the memory limits set
 *     for skew buckets while retaining the tuples for further processing.
 *
 * Parameters:
 *     hashtable (HashJoinTable): The hash join table containing skew buckets.
 * ------------------------------------------------------------------------- */
static void
remove_next_skew_bucket(HashJoinTable hashtable)
{
    int bucketToRemove;
    HashSkewBucket *bucket;
    uint32 hashvalue;
    int bucketno;
    int batchno;
    HashJoinTuple hashTuple;

    /* Locate the bucket to remove */
    bucketToRemove = hashtable->skewBucketNums[hashtable->nSkewBuckets - 1];
    bucket = hashtable->skewBucket[bucketToRemove];

    /* Calculate the appropriate bucket and batch for the tuples in the main hash table */
    hashvalue = bucket->hashvalue;
    exec_hash_get_bucket_and_batch(hashtable, hashvalue, &bucketno, &batchno);

    /* Process all tuples in the bucket */
    hashTuple = bucket->tuples;
    while (hashTuple != NULL)
    {
        HashJoinTuple nextHashTuple = hashTuple->next;
        MinimalTuple tuple;
        Size tupleSize;

        /* Extract the minimal tuple and determine its size */
        tuple = HJTUPLE_MINTUPLE(hashTuple);
        tupleSize = HJTUPLE_OVERHEAD + tuple->t_len;

        /* Move the tuple either to the main hash table or to a temp file */
        if (batchno == hashtable->curbatch)
        {
            /* Move the tuple to the main hash table */
            HashJoinTuple copyTuple;

            /* Copy the tuple into the dense storage for further use */
            copyTuple = (HashJoinTuple) dense_alloc(hashtable, tupleSize);
            memcpy(copyTuple, hashTuple, tupleSize);
            pfree(hashTuple);

            copyTuple->next = hashtable->buckets[bucketno];
            hashtable->buckets[bucketno] = copyTuple;

            /* Update skew space usage, but overall space remains unchanged */
            hashtable->spaceUsedSkew -= tupleSize;
        }
        else
        {
            /* Place the tuple in a temp file for future batches */
            Assert(batchno > hashtable->curbatch);
            exec_hash_join_save_tuple(tuple, hashvalue, &hashtable->innerBatchFile[batchno]);
            pfree(hashTuple);
            hashtable->spaceUsed -= tupleSize;
            hashtable->spaceUsedSkew -= tupleSize;
        }

        hashTuple = nextHashTuple;

        /* Allow this loop to be cancellable */
        CHECK_FOR_INTERRUPTS();
    }

    /* Free the bucket struct itself and reset the hashtable entry to NULL */
    hashtable->skewBucket[bucketToRemove] = NULL;
    hashtable->nSkewBuckets--;
    pfree(bucket);
    hashtable->spaceUsed -= SKEW_BUCKET_OVERHEAD;
    hashtable->spaceUsedSkew -= SKEW_BUCKET_OVERHEAD;

    /* If all skew buckets have been removed, disable skew optimization */
    if (hashtable->nSkewBuckets == 0)
    {
        hashtable->skewEnabled = false;
        pfree(hashtable->skewBucket);
        pfree(hashtable->skewBucketNums);
        hashtable->skewBucket = NULL;
        hashtable->skewBucketNums = NULL;
        hashtable->spaceUsed -= hashtable->spaceUsedSkew;
        hashtable->spaceUsedSkew = 0;
    }
}

/* ------------------------------------------------------------------------- 
 * dense_alloc
 *
 * Allocates memory for hash join tuples from the active memory chunk or creates a new chunk if necessary.
 * This ensures efficient memory usage while minimizing fragmentation by allocating in contiguous blocks.
 *
 * Parameters:
 *     hashtable (HashJoinTable): The hash join table that manages memory chunks.
 *     size (Size): Number of bytes to allocate, properly aligned.
 *
 * Returns:
 *     void*: Pointer to the allocated memory within a hash memory chunk.
 * ------------------------------------------------------------------------- */
static void *
dense_alloc(HashJoinTable hashtable, Size size)
{
    HashMemoryChunk new_Chunk;
    char *ptr;

    /* Ensure the size is properly aligned */
    size = MAXALIGN(size);

    /*
     * If tuple size is larger than 1/4 of the chunk size, allocate a separate chunk.
     * This helps manage large tuples efficiently without occupying excessive space in regular chunks.
     */
    if (size > HASH_CHUNK_THRESHOLD)
    {
        /* Allocate a new chunk and put it at the beginning of the list */
        new_Chunk = (HashMemoryChunk) MemoryContextAlloc(hashtable->batchCxt,
                                                        offsetof(HashMemoryChunkData, data) + size);
        new_Chunk->maxlen = size;
        new_Chunk->used = 0;
        new_Chunk->ntuples = 0;

        /*
         * Add this chunk to the list after the first existing chunk, preserving the space in the "current" chunk.
         */
       if (hashtable->chunks != NULL)
		{
			new_Chunk->next = hashtable->chunks;
			hashtable->chunks = new_Chunk;
		}
		else
		{
			new_Chunk->next = NULL;
			hashtable->chunks = new_Chunk;
		}


        new_Chunk->used += size;
        new_Chunk->ntuples += 1;

        return new_Chunk->data;
    }

    /*
     * If the current chunk doesn't have enough space, allocate a new chunk.
     * Allocate a standard-sized chunk and add it to the beginning of the list.
     */
    if ((hashtable->chunks == NULL) ||
        (hashtable->chunks->maxlen - hashtable->chunks->used) < size)
    {
        new_Chunk = (HashMemoryChunk) MemoryContextAlloc(hashtable->batchCxt,
                                                        offsetof(HashMemoryChunkData, data) + HASH_CHUNK_SIZE);

        new_Chunk->maxlen = HASH_CHUNK_SIZE;
        new_Chunk->used = size;
        new_Chunk->ntuples = 1;

        new_Chunk->next = hashtable->chunks;
        hashtable->chunks = new_Chunk;

        return new_Chunk->data;
    }

    /*
     * If there's enough space in the current chunk, allocate memory from it.
     */
    ptr = hashtable->chunks->data + hashtable->chunks->used;
    hashtable->chunks->used += size;
    hashtable->chunks->ntuples += 1;

    /* Return pointer to the start of the allocated memory */
    return ptr;
}

