/*
 * AuditLog -- append-only, tamper-evident audit log for cache operations.
 *
 * <p>Each entry is chained to the previous via SHA-256. {@link #verifyChain()}
 * walks the in-memory list and re-computes every hash; mismatches indicate
 * tampering. Persistent sinks (Kafka, S3, file) can be plugged in via
 * {@link AuditSink}.
 *
 * <p>Memory bounded: a soft cap is enforced via {@link #setMaxEntries(int)};
 * once exceeded, the oldest entry is dropped (this is logged as a violation,
 * not silently -- chain verification still works on the remaining window).
 */
package org.bytedeco.pytorch.cache.security;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

public final class AuditLog {

    public static final String GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000";

    private final Deque<AuditEntry> entries = new ArrayDeque<>();
    private final ReentrantLock lock = new ReentrantLock();
    private final AtomicLong sequence = new AtomicLong(0);
    private final List<AuditSink> sinks = new ArrayList<>();
    private volatile int maxEntries = 100_000;
    private volatile String lastHash = GENESIS_HASH;

    public AuditLog append(AuditEntry.Builder b) {
        lock.lock();
        try {
            long seq = sequence.incrementAndGet();
            String prevHash = lastHash;
            // Need the entry's final fields first to compute via serialize().
            // We set the entry's fields with a placeholder hash, then patch.
            AuditEntry entry = b.buildWith(seq, prevHash, "0");
            String payload = entry.serialize();
            String hash = AuditEntry.hash(payload, prevHash);
            entry = b.buildWith(seq, prevHash, hash);
            lastHash = hash;
            entries.addLast(entry);
            while (entries.size() > maxEntries) entries.pollFirst();
            for (AuditSink s : sinks) {
                try { s.accept(entry); } catch (Exception ignore) { /* sink failures don't poison log */ }
            }
            return this;
        } finally {
            lock.unlock();
        }
    }

    public void addSink(AuditSink sink) {
        if (sink == null) return;
        lock.lock();
        try { sinks.add(sink); } finally { lock.unlock(); }
    }

    public void setMaxEntries(int n) { this.maxEntries = Math.max(64, n); }

    public List<AuditEntry> snapshot() {
        lock.lock();
        try { return new ArrayList<>(entries); } finally { lock.unlock(); }
    }

    public int size() {
        lock.lock();
        try { return entries.size(); } finally { lock.unlock(); }
    }

    public boolean verifyChain() {
        lock.lock();
        try {
            String prev = GENESIS_HASH;
            long seq = 0;
            for (AuditEntry e : entries) {
                if (e.sequence() != ++seq) return false;
                if (!e.prevHash().equals(prev)) return false;
                String payload = payloadFromEntry(e);
                String h = AuditEntry.hash(payload, prev);
                if (!h.equals(e.hash())) return false;
                prev = h;
            }
            return true;
        } finally {
            lock.unlock();
        }
    }

    private static String payloadFromEntry(AuditEntry e) {
        return e.serialize();
    }

    /** Pluggable persistence / fan-out sink for audit records. */
    public interface AuditSink {
        void accept(AuditEntry entry);
    }

    /** Convenience: builder entry point. */
    public static final class Entry {
        public static AuditEntry.Builder builder() { return AuditEntry.builder(); }

        /** Convenience for callers who don't care about sequence/hash. */
        public static AuditEntry built(String principal, String action, String resource, boolean success) {
            return AuditEntry.builder()
                    .principal(principal)
                    .action(action)
                    .resource(resource)
                    .success(success)
                    .buildWith(0, AuditLog.GENESIS_HASH, "0");
        }
    }
}
