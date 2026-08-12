/*
 * FeatureCrossingGuard — prevents "feature crossing" / future-dated features
 * from leaking into live serving.
 *
 * <p>In real online inference traffic, two failure modes are common:
 * <ul>
 *   <li><b>Feature crossing</b> — a feature's event_timestamp is later than the
 *       prediction time, leaking future information into the model. Common
 *       cause: batch jobs writing yesterday's labels with a delayed timestamp;
 *       some online stores replace the value without checking the timestamp.
 *       Referred to in Tectonic/Feast docs as "feature leakage via clock skew"
 *       and in Chinese ops literature as "特征穿越".</li>
 *   <li><b>Stale-event promotion</b> — a query result from a week-old request
 *       is mixed with fresh user data because of TTL-only refresh semantics.</li>
 * </ul>
 *
 * <p>Defense: every write must carry an eventTimestamp ≤ the request's
 * observation time. Reads in serving must compare the cached eventTimestamp
 * against the request's nowMs and reject anything fresher-than-allowed.
 *
 * <p>Two layers:
 * <ul>
 *   <li>1st line: write-side gate — when {@link CacheValue#eventTimestampMs()}
 *       is greater than the loader's "now" by more than the configured skew,
 *       either block (mode=BLOCK) or rewrite with current timestamp
 *       (mode=MARK_DIRTY).</li>
 *   <li>2nd line: read-side gate — never regress to an older event_ts.</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache;

import org.bytedeco.pytorch.cache.metrics.CacheMetrics;

public final class FeatureCrossingGuard {

    public enum Mode { BLOCK, MARK_DIRTY, ALLOW }

    private final long maxSkewMs;
    private final Mode mode;
    private final CacheMetrics metrics;

    public FeatureCrossingGuard(long maxSkewMs, Mode mode, CacheMetrics metrics) {
        this.maxSkewMs = maxSkewMs < 0 ? 0 : maxSkewMs;
        this.mode = mode == null ? Mode.BLOCK : mode;
        this.metrics = metrics;
    }

    public static FeatureCrossingGuard defaults(CacheMetrics m) {
        return new FeatureCrossingGuard(60_000L, Mode.BLOCK, m);
    }

    /**
     * Admit a candidate value, returning the (possibly demoted) value to cache.
     * Returns {@code null} when the candidate should be rejected.
     */
    public CacheValue<Object> admit(CacheKey key, CacheValue<Object> candidate, long observedAtMs) {
        if (candidate == null) return null;
        long evt = candidate.eventTimestampMs();
        if (evt <= 0) return candidate;
        long delta = evt - observedAtMs;
        if (delta <= maxSkewMs) return candidate;
        if (metrics != null) metrics.recordFeatureCrossingBlocked();
        if (mode == Mode.ALLOW) return candidate;
        if (mode == Mode.BLOCK) return null;
        // MARK_DIRTY: rewrite eventTimestamp to observedAtMs with a marker tag
        return candidate.toBuilder()
                .eventTimestampMs(observedAtMs)
                .sourceTag((candidate.sourceTag() == null ? "" : candidate.sourceTag()) + ":crossing-demoted")
                .build();
    }

    /** Decide whether a fresh-load should clobber an existing entry. */
    public boolean shouldPromote(CacheKey key, CacheValue<Object> existing, CacheValue<Object> fresh,
                                  long observedAtMs) {
        if (existing == null) return true;
        if (existing.eventTimestampMs() > fresh.eventTimestampMs()) {
            // existing is fresher; never regress
            if (metrics != null) metrics.recordFeatureCrossingBlocked();
            return false;
        }
        return true;
    }

    public Mode mode() { return mode; }
    public long maxSkewMs() { return maxSkewMs; }
}
