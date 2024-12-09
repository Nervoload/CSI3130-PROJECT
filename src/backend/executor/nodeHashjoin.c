/*-------------------------------------------------------------------------
 * 
 * CSI 3130
 * 2024 FALL - UNIVERSITY OF OTTAWA
 * DATABASES II
 * JOHN SURETTE
 * 300307306
 * 
 * nodehashjoin.c
 * New implementation to handle Symmetric Hash Join (SHJ)
 *
 * This implementation supports bi-directional probing, which means that
 * both inner and outer hash tables are built and probed concurrently.
 *-------------------------------------------------------------------------*/

#include "postgres.h"
#include "access/htup_details.h"
#include "executor/executor.h"
#include "executor/hashjoin.h"
#include "executor/nodehash.h"
#include "executor/nodehashjoin.h"
#include "miscadmin.h"
#include "utils/memutils.h"

/* ----------------------------------------------------------------
 *  exec_hash_join 
 *
 *  This function implements the Symmetric Hash Join algorithm, which
 *  builds hash tables for both inner and outer relations and supports
 *  pipelined processing and bi-directional probing.
 * ----------------------------------------------------------------*/

/*
 * States of the exec_hash_join state machine for symmetric hash join
 */
#define HJ_BUILD_HASHTABLES    1
#define HJ_NEED_NEW_TUPLE      2
#define HJ_PROBE_OUTER         3
#define HJ_PROBE_INNER         4
#define HJ_NEED_NEW_BATCH      5

static TupleTableSlot *exec_hash_join_get_next_tuple(HashJoinState *hash_join_state, PlanState *plan_node, uint32 *hash_value);
static bool exec_hash_join_new_batch(HashJoinState *hash_join_state);

/*
 * exec_hash_join Function - The main function for executing Symmetric Hash Join
 */

static TupleTableSlot *
exec_hash_join(PlanState *plan_state) {
    HashJoinState *hash_join_state = castNode(HashJoinState, plan_state);
    PlanState *outer_node;
    PlanState *inner_node;
    ExprState *join_qual;
    ExprState *other_qual;
    ExprContext *expr_context;
    HashJoinTable outer_hash_table;
    HashJoinTable inner_hash_table;
    TupleTableSlot *outer_tuple_slot;
    TupleTableSlot *inner_tuple_slot;
    uint32 hash_value_outer, hash_value_inner;
    int current_probe_side;

    /* Get information from the HashJoin node */
    join_qual = hash_join_state->join_state.join_qual;
    other_qual = hash_join_state->join_state.plan_state.qual;
    outer_node = outer_plan_state(hash_join_state);
    inner_node = inner_plan_state(hash_join_state);
    outer_hash_table = hash_join_state->outer_hash_table;
    inner_hash_table = hash_join_state->inner_hash_table;
    expr_context = hash_join_state->join_state.plan_state.expr_context;
    current_probe_side = hash_join_state->probe_side;

    /* Reset per-tuple memory context to free any expression evaluation storage allocated in the previous tuple cycle */
    ResetExprContext(expr_context);

    /* Run the hash join state machine */
    for (;;) {
        CHECK_FOR_INTERRUPTS();

        switch (hash_join_state->join_state_machine) {
            case HJ_BUILD_HASHTABLES:
                /*
                 * First time through: build hash tables for both relations.
                 */
                Assert(outer_hash_table == NULL && inner_hash_table == NULL);

                /* Build hash tables for both inner and outer relations */
                inner_hash_table = exec_hash_table_create((Hash *) inner_node->plan, hash_join_state->hash_operators, false);
                outer_hash_table = exec_hash_table_create((Hash *) outer_node->plan, hash_join_state->hash_operators, false);
                hash_join_state->inner_hash_table = inner_hash_table;
                hash_join_state->outer_hash_table = outer_hash_table;

                /*
                 * Move to the next state, where we need new tuples from either side to
                 * continue probing both hash tables.
                 */
                hash_join_state->join_state_machine = HJ_NEED_NEW_TUPLE;
                /* FALL THROUGH */

            case HJ_NEED_NEW_TUPLE:
                /*
                 * Alternate between getting a tuple from the outer or the inner relation.
                 */
                if (current_probe_side == HJ_PROBE_OUTER) {
                    outer_tuple_slot = exec_hash_join_get_next_tuple(hash_join_state, outer_node, &hash_value_outer);
                    if (!TupIsNull(outer_tuple_slot)) {
                        expr_context->outer_tuple = outer_tuple_slot;
                        exec_hash_table_insert(outer_hash_table, outer_tuple_slot, hash_value_outer);
                        hash_join_state->probe_side = HJ_PROBE_INNER; // Switch probing side to inner
                        hash_join_state->join_state_machine = HJ_PROBE_OUTER;
                        continue;
                    }
                } else {
                    inner_tuple_slot = exec_hash_join_get_next_tuple(hash_join_state, inner_node, &hash_value_inner);
                    if (!TupIsNull(inner_tuple_slot)) {
                        expr_context->inner_tuple = inner_tuple_slot;
                        exec_hash_table_insert(inner_hash_table, inner_tuple_slot, hash_value_inner);
                        hash_join_state->probe_side = HJ_PROBE_OUTER; // Switch probing side to outer
                        hash_join_state->join_state_machine = HJ_PROBE_INNER;
                        continue;
                    }
                }

                /*
                 * If there are no new tuples from either side, switch to batching if needed.
                 */
                hash_join_state->join_state_machine = HJ_NEED_NEW_BATCH;
                /* FALL THROUGH */

            case HJ_PROBE_OUTER:
                /*
                 * Probe the outer hash table for matches with the inner relation.
                 */
                if (!exec_scan_hash_bucket(hash_join_state, expr_context)) {
                    /* No more matches, continue with the next batch */
                    hash_join_state->join_state_machine = HJ_NEED_NEW_BATCH;
                    continue;
                }

                /* Test non-hash join quals */
                if (join_qual == NULL || exec_qual(join_qual, expr_context)) {
                    heap_tuple_header_set_match(HJTUPLE_MINTUPLE(hash_join_state->inner_current_tuple));
                    if (other_qual == NULL || exec_qual(other_qual, expr_context)) {
                        return exec_project(hash_join_state->join_state.plan_state.proj_info);
                    }
                }
                break;

            case HJ_PROBE_INNER:
                /*
                 * Probe the inner hash table for matches with the outer relation.
                 */
                if (!exec_scan_hash_bucket(hash_join_state, expr_context)) {
                    /* No more matches, continue with the next batch */
                    hash_join_state->join_state_machine = HJ_NEED_NEW_BATCH;
                    continue;
                }

                /* Test non-hash join quals */
                if (join_qual == NULL || exec_qual(join_qual, expr_context)) {
                    heap_tuple_header_set_match(HJTUPLE_MINTUPLE(hash_join_state->outer_current_tuple));
                    if (other_qual == NULL || exec_qual(other_qual, expr_context)) {
                        return exec_project(hash_join_state->join_state.plan_state.proj_info);
                    }
                }
                break;

            case HJ_NEED_NEW_BATCH:
                /*
                 * Switch to a new batch if available.
                 */
                if (!exec_hash_join_new_batch(hash_join_state)) {
                    return NULL; /* No more batches, join is complete */
                }
                hash_join_state->join_state_machine = HJ_NEED_NEW_TUPLE;
                break;

            default:
                elog(ERROR, "unrecognized hashjoin state: %d", (int) hash_join_state->join_state_machine);
        }
    }
}

/* ----------------------------------------------------------------
 *  exec_hash_join_get_next_tuple
 *  Retrieve the next tuple for the hash join, from the provided plan node.
 * ---------------------------------------------------------------- */
static TupleTableSlot *
exec_hash_join_get_next_tuple(HashJoinState *hash_join_state, PlanState *plan_node, uint32 *hash_value) {
    TupleTableSlot *slot = exec_proc_node(plan_node);

    if (!TupIsNull(slot)) {
        /* Compute hash value for the tuple */
        ExprContext *expr_context = hash_join_state->join_state.plan_state.expr_context;
        expr_context->outer_tuple = slot;
        if (exec_hash_get_hash_value(hash_join_state->inner_hash_table, expr_context,
                                     hash_join_state->outer_hash_keys, true, false,
                                     hash_value)) {
            return slot;
        }
    }
    return NULL;
}

/* ----------------------------------------------------------------
 * exec_init_hash_join
 * 
 * This function initializes the state required to perform a Hash Join operation.
 * It sets up all necessary data structures, initializes child nodes, and
 * prepares the hash join for execution. The function follows these main steps:
 * 
 * - Validates flags to ensure unsupported operations are not requested.
 * - Initializes the HashJoinState data structure and other context information.
 * - Initializes the child nodes and sets up tuple slots and expressions.
 * - Handles outer joins by creating appropriate null tuple slots.
 * - Deconstructs hash clauses into their components for individual evaluation.
 * - Sets up projection information for the resulting joined tuples.
 * 
 * Parameters:
 * - node: A pointer to the HashJoin plan node that specifies the join.
 * - estate: Execution state containing all necessary runtime data.
 * - eflags: Execution flags used to control execution options.
 * 
 * Returns:
 * - HashJoinState*: A pointer to the initialized HashJoinState structure.
 * ---------------------------------------------------------------- */

HashJoinState *
exec_init_hash_join(HashJoin *node, EState *estate, int eflags) {
    HashJoinState *hash_join_state;
    Plan *outer_node;
    Hash *hash_node;
    List *left_clauses;
    List *right_clauses;
    List *hash_operators;
    ListCell *list_cell;

    /* Check for unsupported flags */
    Assert(!(eflags & (EXEC_FLAG_BACKWARD | EXEC_FLAG_MARK)));

    /*
     * Create state structure for the hash join node.
     */
    hash_join_state = makeNode(HashJoinState);
    hash_join_state->join_state.plan_state.plan = (Plan *) node;
    hash_join_state->join_state.plan_state.state = estate;
    hash_join_state->join_state.plan_state.exec_proc_node = exec_hash_join;

    /*
     * Miscellaneous initialization - create expression context for the node.
     */
    exec_assign_expr_context(estate, &hash_join_state->join_state.plan_state);

    /*
     * Initialize child expressions for join and hash clauses.
     */
    hash_join_state->join_state.plan_state.qual =
        exec_init_qual(node->join.plan.qual, (PlanState *) hash_join_state);
    hash_join_state->join_state.join_type = node->join.jointype;
    hash_join_state->join_state.join_qual =
        exec_init_qual(node->join.joinqual, (PlanState *) hash_join_state);
    hash_join_state->hash_clauses =
        exec_init_qual(node->hashclauses, (PlanState *) hash_join_state);

    /*
     * Initialize child nodes.
     *
     * Note: The REWIND flag for the inner input could be suppressed, assuming
     * the hash is a single batch. This optimization might not always be beneficial.
     */
    outer_node = outer_plan(node);
    hash_node = (Hash *) inner_plan(node);

    outer_plan_state(hash_join_state) = exec_init_node(outer_node, estate, eflags);
    inner_plan_state(hash_join_state) = exec_init_node((Plan *) hash_node, estate, eflags);

    /*
     * Initialize tuple table slots for result and outer tuples.
     */
    exec_init_result_tuple_slot(estate, &hash_join_state->join_state.plan_state);
    hash_join_state->outer_tuple_slot = exec_init_extra_tuple_slot(estate);

    /*
     * Determine if we need to consider only the first matching inner tuple.
     */
    hash_join_state->join_state.single_match = (node->join.inner_unique ||
                                                node->join.jointype == JOIN_SEMI);

    /*
     * Set up null tuples for outer joins, if needed.
     */
    switch (node->join.jointype) {
        case JOIN_INNER:
        case JOIN_SEMI:
            break;
        case JOIN_LEFT:
        case JOIN_ANTI:
            hash_join_state->null_inner_tuple_slot =
                    exec_init_null_tuple_slot(estate,
                                              exec_get_result_type(inner_plan_state(hash_join_state)));
            break;
        case JOIN_RIGHT:
            hash_join_state->null_inner_tuple_slot =
                    exec_init_null_tuple_slot(estate,
                                              exec_get_result_type(outer_plan_state(hash_join_state)));
            break;
        case JOIN_FULL:
            hash_join_state->null_inner_tuple_slot =
                    exec_init_null_tuple_slot(estate,
                                              exec_get_result_type(outer_plan_state(hash_join_state)));
            hash_join_state->null_inner_tuple_slot =
                    exec_init_null_tuple_slot(estate,
                                              exec_get_result_type(inner_plan_state(hash_join_state)));
            break;
        default:
            elog(ERROR, "unrecognized join type: %d",
                 (int) node->join.jointype);
    }

    /*
     * Assign result tuple slot from the hash node's result slot.
     */
    {
        HashState  *hash_state = (HashState *) inner_plan_state(hash_join_state);
        TupleTableSlot *slot = hash_state->plan_state.result_tuple_slot;

        hash_join_state->outer_tuple_slot = slot;
    }

    /*
     * Initialize the result type and projection information for joined tuples.
     */
    exec_assign_result_type_from_tl(&hash_join_state->join_state.plan_state);
    exec_assign_projection_info(&hash_join_state->join_state.plan_state, NULL);

    exec_set_slot_descriptor(hash_join_state->outer_tuple_slot,
                              exec_get_result_type(outer_plan_state(hash_join_state)));

    /*
     * Initialize hash-specific information.
     */
    hash_join_state->inner_hash_table = NULL;
    hash_join_state->first_outer_tuple_slot = NULL;

    hash_join_state->inner_current_hash_value = 0;
    hash_join_state->outer_current_bucket_no = 0;
    hash_join_state->current_skew_bucket_no = INVALID_SKEW_BUCKET_NO;
    hash_join_state->inner_current_tuple = NULL;

    /*
     * Deconstruct hash clauses into their outer and inner components for evaluation.
     * Also, prepare hash operator OIDs for later use when looking up hash functions.
     */
    left_clauses = NIL;
    right_clauses = NIL;
    hash_operators = NIL;
    foreach(list_cell, node->hashclauses) {
        OpExpr *hash_clause = lfirst_node(OpExpr, list_cell);

        left_clauses = lappend(left_clauses, exec_init_expr(linitial(hash_clause->args),
                                                            (PlanState *) hash_join_state));
        right_clauses = lappend(right_clauses, exec_init_expr(lsecond(hash_clause->args),
                                                              (PlanState *) hash_join_state));
        hash_operators = lappend_oid(hash_operators, hash_clause->opno);
    }
    hash_join_state->outer_hash_keys = left_clauses;
    hash_join_state->inner_hash_keys = right_clauses;
    hash_join_state->hash_operators = hash_operators;
    /* Child Hash node also needs to evaluate inner hash keys */
    ((HashState *) inner_plan_state(hash_join_state))->hash_keys = right_clauses;

    /* Set initial state for hash join operation */
    hash_join_state->join_state_machine = HJ_BUILD_HASHTABLE;
    hash_join_state->matched_outer = false;
    hash_join_state->outer_not_empty = false;

    return hash_join_state;
}

/* -------------------------------------------------------------------------
 *  exec_end_hash_join
 *  -------------------------------------------------------------------------
 *  Function to clean up resources and perform deallocation for the HashJoin 
 *  node after it is no longer required. This includes destroying the hash
 *  tables, freeing expression contexts, clearing tuples, and cleaning up any
 *  child nodes.
 *
 *  Parameters:
 *      - HashJoinState *node: State of the current HashJoin node which contains
 *        references to the hash tables, subnodes, and expression contexts.
 *
 *  Returns: void
 * ------------------------------------------------------------------------- */
void exec_end_hash_join(HashJoinState *hash_join_state) {
    // Destroy the inner hash table if it exists, freeing allocated memory.
    if (hash_join_state->inner_hash_table) {
        exec_hash_table_destroy(hash_join_state->inner_hash_table);
        hash_join_state->inner_hash_table = NULL;
    }

    // Free the expression context allocated for the join node.
    exec_free_expr_context(&hash_join_state->join_state.plan_state);

    // Clear tuple slots to ensure no lingering data remains.
    exec_clear_tuple(hash_join_state->join_state.plan_state.result_tuple_slot);
    exec_clear_tuple(hash_join_state->outer_tuple_slot);

    // Recursively end the subnodes that were used in the join.
    exec_end_node(outer_plan_state(hash_join_state));
    exec_end_node(inner_plan_state(hash_join_state));
}

/* -------------------------------------------------------------------------
 *  exec_hash_join_outer_get_tuple
 *  -------------------------------------------------------------------------
 *  Retrieves the next outer tuple for the HashJoin, either by executing the 
 *  outer plan node during the first pass or by reading from the temporary 
 *  files used for hash join batching. Returns NULL if no more tuples are 
 *  available in the current batch. Stores the computed or previously read 
 *  hash value of the tuple in the provided hash_value pointer.
 *
 *  Parameters:
 *      - PlanState *outer_node: The state of the outer plan node providing tuples.
 *      - HashJoinState *hash_join_state: The current state of the hash join containing
 *        references to hash tables and batch files.
 *      - uint32 *hash_value: Pointer to store the computed or retrieved hash value.
 *
 *  Returns:
 *      - TupleTableSlot *: The next available tuple or NULL if none available.
 * ------------------------------------------------------------------------- */
static TupleTableSlot *exec_hash_join_outer_get_tuple(PlanState *outer_node, HashJoinState *hash_join_state, uint32 *hash_value) {
    HashJoinTable hash_table = hash_join_state->inner_hash_table;
    int current_batch = hash_table->current_batch;
    TupleTableSlot *slot;

    if (current_batch == 0) {  // First pass - fetch the next tuple from the outer node.
        // Check if the first tuple was already fetched but not used.
        slot = hash_join_state->first_outer_tuple_slot;
        if (!TupIsNull(slot)) {
            hash_join_state->first_outer_tuple_slot = NULL;
        } else {
            slot = exec_proc_node(outer_node);
        }

        while (!TupIsNull(slot)) {
            // Compute the hash value for the current tuple.
            ExprContext *expr_context = hash_join_state->join_state.plan_state.expr_context;
            expr_context->outer_tuple = slot;
            if (exec_hash_get_hash_value(hash_table, expr_context, hash_join_state->outer_hash_keys, true, HJ_FILL_OUTER(hash_join_state), hash_value)) {
                // Outer relation is not empty, possibly rescan later.
                hash_join_state->outer_not_empty = true;
                return slot;
            }
            // The tuple couldn't be matched due to NULL, proceed to the next one.
            slot = exec_proc_node(outer_node);
        }
    } else if (current_batch < hash_table->num_batches) {
        BufFile *batch_file = hash_table->outer_batch_file[current_batch];

        // Check if the outer batch file exists (could be empty in some cases).
        if (batch_file == NULL) {
            return NULL;
        }

        // Fetch the tuple saved in the temporary batch file.
        slot = exec_hash_join_get_saved_tuple(hash_join_state, batch_file, hash_value, hash_join_state->outer_tuple_slot);
        if (!TupIsNull(slot)) {
            return slot;
        }
    }

    // No more tuples available in the current batch.
    return NULL;
}

/* -------------------------------------------------------------------------
 *  exec_hash_join_new_batch
 *  -------------------------------------------------------------------------
 *  Switches the hash join to a new batch if one exists, preparing the state
 *  for processing the next set of tuples. This involves resetting hash tables
 *  and loading new batch files as needed.
 *
 *  Parameters:
 *      - HashJoinState *hash_join_state: Current state of the HashJoin containing
 *        hash table information and references to batch files.
 *
 *  Returns:
 *      - bool: True if switching to a new batch was successful, false if no
 *        more batches are available.
 * ------------------------------------------------------------------------- */
static bool exec_hash_join_new_batch(HashJoinState *hash_join_state) {
    HashJoinTable hash_table = hash_join_state->inner_hash_table;
    int num_batches = hash_table->num_batches;
    int current_batch = hash_table->current_batch;
    BufFile *inner_file;
    TupleTableSlot *slot;
    uint32 hash_value;

    if (current_batch > 0) {
        // Close the previous outer batch file to free disk space.
        if (hash_table->outer_batch_file[current_batch]) {
            BufFileClose(hash_table->outer_batch_file[current_batch]);
        }
        hash_table->outer_batch_file[current_batch] = NULL;
    } else {  // Finished the first batch - reset skew optimization state.
        hash_table->skew_enabled = false;
        hash_table->skew_bucket = NULL;
        hash_table->skew_bucket_nums = NULL;
        hash_table->num_skew_buckets = 0;
        hash_table->space_used_skew = 0;
    }

    // Increment the current batch and determine if more batches exist.
    current_batch++;
    while (current_batch < num_batches &&
           (hash_table->outer_batch_file[current_batch] == NULL || hash_table->inner_batch_file[current_batch] == NULL)) {
        if ((hash_table->outer_batch_file[current_batch] && HJ_FILL_OUTER(hash_join_state)) ||
            (hash_table->inner_batch_file[current_batch] && HJ_FILL_INNER(hash_join_state)) ||
            (hash_table->inner_batch_file[current_batch] && num_batches != hash_table->original_num_batches) ||
            (hash_table->outer_batch_file[current_batch] && num_batches != hash_table->num_batches_at_outer_start)) {
            break;
        }
        // Release any empty or unused batch files immediately.
        if (hash_table->inner_batch_file[current_batch]) {
            BufFileClose(hash_table->inner_batch_file[current_batch]);
        }
        hash_table->inner_batch_file[current_batch] = NULL;
        if (hash_table->outer_batch_file[current_batch]) {
            BufFileClose(hash_table->outer_batch_file[current_batch]);
        }
        hash_table->outer_batch_file[current_batch] = NULL;
        current_batch++;
    }

    // If no more batches exist, return false to indicate completion.
    if (current_batch >= num_batches) {
        return false;
    }

    hash_table->current_batch = current_batch;

    // Reset hash table for processing the new batch.
    exec_hash_table_reset(hash_table);

    inner_file = hash_table->inner_batch_file[current_batch];

    // Reload the hash table with new inner batch data.
    if (inner_file != NULL) {
        if (BufFileSeek(inner_file, 0, 0L, SEEK_SET)) {
            ereport(ERROR, (errcode_for_file_access(), errmsg("could not rewind hash-join temporary file: %m")));
        }

        while ((slot = exec_hash_join_get_saved_tuple(hash_join_state, inner_file, &hash_value, hash_join_state->outer_tuple_slot))) {
            exec_hash_table_insert(hash_table, slot, hash_value);
        }

        // After building the hash table, the inner batch file is no longer needed.
        BufFileClose(inner_file);
        hash_table->inner_batch_file[current_batch] = NULL;
    }

    // Rewind the outer batch file for reading, if present.
    if (hash_table->outer_batch_file[current_batch] != NULL) {
        if (BufFileSeek(hash_table->outer_batch_file[current_batch], 0, 0L, SEEK_SET)) {
            ereport(ERROR, (errcode_for_file_access(), errmsg("could not rewind hash-join temporary file: %m")));
        }
    }

    return true;
}

/* -------------------------------------------------------------------------
 *  exec_hash_join_save_tuple
 *  -------------------------------------------------------------------------
 *  Saves a tuple to the appropriate batch file. The tuple data recorded in the 
 *  file includes the hash value followed by the tuple in MinimalTuple format.
 *  The function ensures that buffers do not get corrupted by only writing
 *  within the regular executor context.
 *
 *  Parameters:
 *      - MinimalTuple tuple: The tuple to be saved.
 *      - uint32 hash_value: The hash value of the tuple.
 *      - BufFile **file_ptr: Pointer to the batch file where the tuple should be saved.
 *
 *  Returns: void
 * ------------------------------------------------------------------------- */
void exec_hash_join_save_tuple(MinimalTuple tuple, uint32 hash_value, BufFile **file_ptr) {
    BufFile *file = *file_ptr;
    size_t written;

    // Open the batch file if this is the first write.
    if (file == NULL) {
        file = BufFileCreateTemp(false);
        *file_ptr = file;
    }

    // Write the hash value followed by the tuple data to the batch file.
    written = BufFileWrite(file, (void *) &hash_value, sizeof(uint32));
    if (written != sizeof(uint32)) {
        ereport(ERROR, (errcode_for_file_access(), errmsg("could not write to hash-join temporary file: %m")));
    }

    written = BufFileWrite(file, (void *) tuple, tuple->t_len);
    if (written != tuple->t_len) {
        ereport(ERROR, (errcode_for_file_access(), errmsg("could not write to hash-join temporary file: %m")));
    }
}

/* -------------------------------------------------------------------------
 *  exec_hash_join_get_saved_tuple
 *  -------------------------------------------------------------------------
 *  Reads the next tuple from a batch file. Returns NULL if there are no more
 *  tuples to read. The hash value is also retrieved and stored in the provided
 *  pointer.
 *
 *  Parameters:
 *      - HashJoinState *hash_join_state: State of the HashJoin containing file references.
 *      - BufFile *file: The batch file to read from.
 *      - uint32 *hash_value: Pointer to store the hash value of the retrieved tuple.
 *      - TupleTableSlot *tuple_slot: The slot to store the retrieved tuple.
 *
 *  Returns:
 *      - TupleTableSlot *: The retrieved tuple or NULL if no more tuples are available.
 * ------------------------------------------------------------------------- */
static TupleTableSlot *exec_hash_join_get_saved_tuple(HashJoinState *hash_join_state, BufFile *file, uint32 *hash_value, TupleTableSlot *tuple_slot) {
    uint32 header[2];
    size_t num_read;
    MinimalTuple tuple;

    // Check for interrupts, as this function is an alternative path to exec_proc_node().
    CHECK_FOR_INTERRUPTS();

    // Read the hash value and tuple length from the file.
    num_read = BufFileRead(file, (void *) header, sizeof(header));
    if (num_read == 0) {  // End of file.
        exec_clear_tuple(tuple_slot);
        return NULL;
    }
    if (num_read != sizeof(header)) {
        ereport(ERROR, (errcode_for_file_access(), errmsg("could not read from hash-join temporary file: %m")));
    }
    *hash_value = header[0];

    // Allocate memory for the tuple and read the tuple data from the file.
    tuple = (MinimalTuple) palloc(header[1]);
    tuple->t_len = header[1];
    num_read = BufFileRead(file, (void *) ((char *) tuple + sizeof(uint32)), header[1] - sizeof(uint32));
    if (num_read != header[1] - sizeof(uint32)) {
        ereport(ERROR, (errcode_for_file_access(), errmsg("could not read from hash-join temporary file: %m")));
    }
    return exec_store_minimal_tuple(tuple, tuple_slot, true);
}

/* -------------------------------------------------------------------------
 *  exec_rescan_hash_join
 *  -------------------------------------------------------------------------
 *  Rescans the HashJoin node to handle multi-batch joins or parameter changes.
 *  If it is a single-batch join, the function attempts to reuse the existing
 *  hash table without rebuilding it.
 *
 *  Parameters:
 *      - HashJoinState *hash_join_state: The state of the HashJoin node, containing the
 *        hash tables and subnodes that need to be reset.
 *
 *  Returns: void
 * ------------------------------------------------------------------------- */
void exec_rescan_hash_join(HashJoinState *hash_join_state) {
    if (hash_join_state->inner_hash_table != NULL) {
        if (hash_join_state->inner_hash_table->num_batches == 1 && hash_join_state->join_state.plan_state.right_tree->chg_param == NULL) {
            // If reusing the hash table, reset match flags for right/full joins.
            if (HJ_FILL_INNER(hash_join_state)) {
                exec_hash_table_reset_match_flags(hash_join_state->inner_hash_table);
            }
            hash_join_state->outer_not_empty = false;
            hash_join_state->join_state_machine = HJ_NEED_NEW_OUTER;
        } else {
            // Destroy and rebuild the hash table for rescanning.
            exec_hash_table_destroy(hash_join_state->inner_hash_table);
            hash_join_state->inner_hash_table = NULL;
            hash_join_state->join_state_machine = HJ_BUILD_HASHTABLE;
            if (hash_join_state->join_state.plan_state.right_tree->chg_param == NULL) {
                exec_rescan(hash_join_state->join_state.plan_state.right_tree);
            }
        }
    }

    // Reset intra-tuple state to ensure a clean rescan.
    hash_join_state->inner_current_hash_value = 0;
    hash_join_state->outer_current_bucket_no = 0;
    hash_join_state->current_skew_bucket_no = INVALID_SKEW_BUCKET_NO;
    hash_join_state->inner_current_tuple = NULL;

    hash_join_state->matched_outer = false;
    hash_join_state->first_outer_tuple_slot = NULL;

    // Rescan the left subtree if parameters haven't changed.
    if (hash_join_state->join_state.plan_state.left_tree->chg_param == NULL) {
        exec_rescan(hash_join_state->join_state.plan_state.left_tree);
    }
}
