/*
 * SubProcessGroupWrapper - filters a ProcessGroupWrapper to a logical subset of ranks.
 *
 * <p>Used by hybrid parallel trainers (DP/TP/PP) to form subgroups without spawning new
 * native process groups.
 */
package org.bytedeco.pytorch.distributed;

import java.util.function.IntPredicate;

/**
 * A filtered view over an existing {@link ProcessGroupWrapper}.
 *
 * <p>This is a lightweight wrapper that holds a delegate plus a rank predicate.
 * The delegate is used for actual communication; the predicate is consulted
 * for membership decisions.
 */
public final class SubProcessGroupWrapper {
    private final ProcessGroupWrapper delegate;
    private final IntPredicate rankFilter;

    public SubProcessGroupWrapper(ProcessGroupWrapper delegate, IntPredicate rankFilter) {
        this.delegate = delegate;
        this.rankFilter = rankFilter != null ? rankFilter : r -> true;
    }

    /** Returns true if the given global rank is a member of this subgroup. */
    public boolean containsRank(int rank) {
        return rankFilter.test(rank);
    }

    /** Returns the underlying delegate group. */
    public ProcessGroupWrapper delegate() {
        return delegate;
    }

    /** Convenience accessor for callers that previously used ProcessGroupWrapper directly. */
    public ProcessGroupWrapper underlying() {
        return delegate;
    }

    /** Returns the world size of the underlying group, or 1 if no delegate. */
    public int getWorldSize() {
        return delegate != null ? delegate.getWorldSize() : 1;
    }

    /** Returns the rank of the underlying group, or 0 if no delegate. */
    public int getRank() {
        return delegate != null ? delegate.getRank() : 0;
    }

    /** Forward to delegate. */
    public Work allreduce(java.util.List<org.bytedeco.pytorch.Tensor> tensors,
                          ReduceOp.RedOpType op) {
        return delegate != null ? delegate.allreduce(tensors, op) : null;
    }
}
