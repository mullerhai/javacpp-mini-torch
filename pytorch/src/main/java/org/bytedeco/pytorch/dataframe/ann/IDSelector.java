package org.bytedeco.pytorch.dataframe.ann;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Filter used by {@link HnswIndex#search(float[], int, int, IDSelector)} to keep
 * only externally-id'd vectors that match the predicate.
 *
 * <p>Implementations:
 * <ul>
 *   <li>{@link #all()}      — match every id (the default)</li>
 *   <li>{@link #none()}     — match no id (useful for "exclude everything")</li>
 *   <li>{@link #of(long...)} — explicit allow-list</li>
 *   <li>{@link #range(long, long)} — inclusive id range</li>
 *   <li>{@link #negate(IDSelector)} — invert another selector</li>
 * </ul>
 */
public interface IDSelector {
    boolean is_member(long id);

    static IDSelector all() { return ALL; }
    static IDSelector none() { return NONE; }
    static IDSelector of(long... ids) { return new ArraySelector(ids); }
    static IDSelector range(long from, long toInclusive) {
        return id -> id >= from && id <= toInclusive;
    }
    static IDSelector negate(IDSelector inner) {
        return id -> !inner.is_member(id);
    }

    IDSelector ALL = id -> true;
    IDSelector NONE = id -> false;

    final class ArraySelector implements IDSelector {
        private final long[] ids;
        private final Set<Long> set;
        ArraySelector(long[] ids) {
            this.ids = ids == null ? new long[0] : ids;
            this.set = new HashSet<>();
            for (long id : this.ids) set.add(id);
        }
        @Override public boolean is_member(long id) { return set.contains(id); }
        @Override public String toString() {
            return "IDSelector.ArraySelector(" + Arrays.toString(ids) + ")";
        }
    }
}