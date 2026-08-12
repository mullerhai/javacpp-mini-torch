/*
 * CountMinSketch -- 4-bit, depth-4 sketch used by W-TinyLFU.
 *
 * <p>Tradeoffs against a full CountMinSketch:
 * <ul>
 *   <li>4-bit counters (saturating at 15) keep memory tight (~6 bytes per slot
 *       at depth=4) -- appropriate for cache workloads where the top-frequency
 *       bands are what matters.</li>
 *   <li>depth=4 gives FPR ~ 0.01% with the default width of 16k. Increasing
 *       width lowers FPR but uses more memory.</li>
 *   <li>No reset -- relies on randomized eviction (half-life mode) to keep
 *       counters fresh. This is the design choice in Caffeine.</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache.eviction;

final class CountMinSketch {

    private final long[] seeds;
    private final short[][] table;   // [depth][width]
    private final int depth;
    private final int width;
    private long size;               // total increments (used for half-life reset)

    CountMinSketch(int depth, int width) {
        this.depth = depth;
        this.width = width;
        this.table = new short[depth][width];
        this.seeds = new long[depth];
        for (int i = 0; i < depth; i++) {
            seeds[i] = 0x9E3779B97F4A7C15L ^ (0xBF58476D1CE4E5B9L * (i + 1));
        }
    }

    int estimate(int hash) {
        int min = Short.MAX_VALUE;
        for (int i = 0; i < depth; i++) {
            int idx = indexFor(i, hash);
            int v = table[i][idx] & 0xFFFF;
            if (v < min) min = v;
        }
        return min;
    }

    void increment(int hash) {
        boolean added = false;
        for (int i = 0; i < depth; i++) {
            int idx = indexFor(i, hash);
            int v = table[i][idx] & 0xFFFF;
            if (v < 15) {
                table[i][idx] = (short) (v + 1);
                added = true;
            }
        }
        if (added) size++;
    }

    void reset() {
        for (int i = 0; i < depth; i++) {
            short[] row = table[i];
            for (int j = 0; j < width; j++) {
                row[j] = (short) ((row[j] & 0xFFFF) >>> 1);
            }
        }
        size = size >>> 1;
    }

    long totalAdds() { return size; }

    int depth() { return depth; }
    int width() { return width; }

    private int indexFor(int row, int hash) {
        long h = hash * 0x9E3779B97F4A7C15L ^ seeds[row];
        h ^= (h >>> 32);
        h *= 0xBF58476D1CE4E5B9L;
        h ^= (h >>> 32);
        return (int) (Math.floorMod(h, width));
    }
}
