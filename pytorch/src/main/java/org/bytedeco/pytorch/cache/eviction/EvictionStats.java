/*
 * EvictionStats -- observed counters for an eviction policy.
 */
package org.bytedeco.pytorch.cache.eviction;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

public final class EvictionStats {

    private final LongAdder touches = new LongAdder();
    private final LongAdder admits = new LongAdder();
    private final LongAdder evictions = new LongAdder();
    private final LongAdder touchesDropped = new LongAdder();
    private final LongAdder admitsRejected = new LongAdder();
    private final AtomicLong lastEvictionAtMs = new AtomicLong();

    public void recordTouch()            { touches.increment(); }
    public void recordAdmit()            { admits.increment(); }
    public void recordEviction()         { evictions.increment(); lastEvictionAtMs.set(System.currentTimeMillis()); }
    public void recordTouchDropped()     { touchesDropped.increment(); }
    public void recordAdmitRejected()    { admitsRejected.increment(); }

    public long touches()                { return touches.sum(); }
    public long admits()                 { return admits.sum(); }
    public long evictions()              { return evictions.sum(); }
    public long touchesDropped()         { return touchesDropped.sum(); }
    public long admitsRejected()         { return admitsRejected.sum(); }
    public long lastEvictionAtMs()       { return lastEvictionAtMs.get(); }

    public Snapshot snapshot() {
        long t = touches.sum();
        return new Snapshot(t, admits.sum(), evictions.sum(),
                touchesDropped.sum(), admitsRejected.sum(), lastEvictionAtMs.get());
    }

    public void reset() {
        touches.reset(); admits.reset(); evictions.reset();
        touchesDropped.reset(); admitsRejected.reset();
        lastEvictionAtMs.set(0);
    }

    public static final class Snapshot {
        public final long touches, admits, evictions, touchesDropped, admitsRejected;
        public final long lastEvictionAtMs;

        public Snapshot(long touches, long admits, long evictions,
                        long touchesDropped, long admitsRejected, long lastEvictionAtMs) {
            this.touches = touches;
            this.admits = admits;
            this.evictions = evictions;
            this.touchesDropped = touchesDropped;
            this.admitsRejected = admitsRejected;
            this.lastEvictionAtMs = lastEvictionAtMs;
        }

        @Override
        public String toString() {
            return "EvictionStats{touch=" + touches + ", admit=" + admits
                    + ", eviction=" + evictions + ", touchDropped=" + touchesDropped
                    + ", admitRejected=" + admitsRejected + "}";
        }
    }
}
