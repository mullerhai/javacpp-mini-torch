/*
 * AuditEntry -- single entry in the cache audit log.
 *
 * <p>Each entry is part of a hash chain (linked-list of SHA-256 over the
 * previous entry + this entry's payload) so a missing or altered entry is
 * detectable via {@link AuditLog#verifyChain()}.
 */
package org.bytedeco.pytorch.cache.security;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

public final class AuditEntry {

    private final long timestampMs;
    private final String principal;
    private final String action;
    private final String resource;
    private final boolean success;
    private final String detail;
    private final long sequence;
    private final String prevHash;
    private final String hash;

    AuditEntry(long timestampMs, String principal, String action, String resource,
               boolean success, String detail, long sequence, String prevHash, String hash) {
        this.timestampMs = timestampMs;
        this.principal = principal;
        this.action = action;
        this.resource = resource;
        this.success = success;
        this.detail = detail;
        this.sequence = sequence;
        this.prevHash = prevHash;
        this.hash = hash;
    }

    public long timestampMs() { return timestampMs; }
    public String principal() { return principal; }
    public String action() { return action; }
    public String resource() { return resource; }
    public boolean success() { return success; }
    public String detail() { return detail; }
    public long sequence() { return sequence; }
    public String prevHash() { return prevHash; }
    public String hash() { return hash; }

    public String serialize() {
        StringBuilder sb = new StringBuilder(256);
        sb.append(sequence).append('|')
          .append(timestampMs).append('|')
          .append(principal).append('|')
          .append(action).append('|')
          .append(resource).append('|')
          .append(success).append('|')
          .append(detail == null ? "" : detail).append('|')
          .append(prevHash);
        return sb.toString();
    }

    public static String hash(String payload, String prevHash) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            md.update(prevHash.getBytes(StandardCharsets.UTF_8));
            md.update(payload.getBytes(StandardCharsets.UTF_8));
            return hex(md.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private static String hex(byte[] b) {
        StringBuilder sb = new StringBuilder(b.length * 2);
        for (byte x : b) sb.append(String.format("%02x", x));
        return sb.toString();
    }

    public static final class Builder {
        private long timestampMs = System.currentTimeMillis();
        private String principal;
        private String action;
        private String resource;
        private boolean success = true;
        private String detail;

        public Builder timestampMs(long t) { this.timestampMs = t; return this; }
        public Builder principal(String p)   { this.principal = p; return this; }
        public Builder action(String a)      { this.action = a; return this; }
        public Builder resource(String r)    { this.resource = r; return this; }
        public Builder success(boolean s)    { this.success = s; return this; }
        public Builder detail(String d)      { this.detail = d; return this; }

        // sequence / prevHash / hash are filled in by AuditLog.append()
        AuditEntry buildWith(long seq, String prevHash, String hash) {
            return new AuditEntry(timestampMs, principal, action, resource,
                    success, detail, seq, prevHash, hash);
        }
    }

    static Builder builder() { return new Builder(); }
}
