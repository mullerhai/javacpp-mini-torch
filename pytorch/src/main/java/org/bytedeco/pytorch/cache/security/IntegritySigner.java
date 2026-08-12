/*
 * IntegritySigner -- HMAC-SHA256 over (key, value) so any tampering is
 * detectable on read.
 *
 * <p>Wraps {@link CacheValue}; the signature is stored as a hex tag so it
 * can be inspected without unpacking the value.
 *
 * <p>Key rotation is supported via {@link #rotateKey(String, byte[])}; older
 * envelopes remain verifiable using their recorded key version.
 */
package org.bytedeco.pytorch.cache.security;

import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

public final class IntegritySigner {

    private final AtomicLong activeVersion = new AtomicLong(0);
    private volatile SecretKeySpec activeKey;
    private final Map<String, SecretKeySpec> historical = new HashMap<>();

    public IntegritySigner(byte[] initialKey) {
        this(initialKey, "k1");
    }

    public IntegritySigner(byte[] initialKey, String initialVersion) {
        if (initialKey == null || initialKey.length < 16)
            throw new IllegalArgumentException("HMAC key must be >= 16 bytes");
        this.activeKey = new SecretKeySpec(initialKey, "HmacSHA256");
        activeVersion.set(1);
        historical.put(initialVersion, this.activeKey);
    }

    public String sign(CacheKey key, CacheValue<Object> value) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(activeKey);
            mac.update(key.toStorageKey().getBytes(StandardCharsets.UTF_8));
            if (value != null) {
                Object v = value.value();
                if (v != null) mac.update(String.valueOf(v).getBytes(StandardCharsets.UTF_8));
                mac.update(longToBytes(value.eventTimestampMs()));
                mac.update(longToBytes(value.expireAtMs()));
            }
            byte[] tag = mac.doFinal();
            return "v1." + currentVersion() + "." + hex(tag);
        } catch (GeneralSecurityException e) {
            throw new SecurityException("sign failed", e);
        }
    }

    public boolean verify(CacheKey key, CacheValue<Object> value, String token) {
        if (token == null) return false;
        String[] parts = token.split("\\.", 3);
        if (parts.length != 3) return false;
        if (!"v1".equals(parts[0])) return false;
        String ver = parts[1];
        String expected = signWithVersion(key, value, ver);
        return expected.equals(token);
    }

    public synchronized void rotateKey(String newVersion, byte[] newKey) {
        if (newKey == null || newKey.length < 16)
            throw new IllegalArgumentException("HMAC key must be >= 16 bytes");
        activeVersion.incrementAndGet();
        this.activeKey = new SecretKeySpec(newKey, "HmacSHA256");
        historical.put(newVersion, this.activeKey);
    }

    private String signWithVersion(CacheKey key, CacheValue<Object> value, String ver) {
        try {
            SecretKeySpec k = historical.get(ver);
            if (k == null) return "v1." + ver + ".invalid";
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(k);
            mac.update(key.toStorageKey().getBytes(StandardCharsets.UTF_8));
            if (value != null) {
                Object v = value.value();
                if (v != null) mac.update(String.valueOf(v).getBytes(StandardCharsets.UTF_8));
                mac.update(longToBytes(value.eventTimestampMs()));
                mac.update(longToBytes(value.expireAtMs()));
            }
            return "v1." + ver + "." + hex(mac.doFinal());
        } catch (GeneralSecurityException e) {
            throw new SecurityException("verify failed", e);
        }
    }

    private String currentVersion() { return "k" + activeVersion.get(); }

    private static String hex(byte[] b) {
        StringBuilder sb = new StringBuilder(b.length * 2);
        for (byte x : b) sb.append(String.format("%02x", x));
        return sb.toString();
    }

    private static byte[] longToBytes(long v) {
        return new byte[]{
                (byte) ((v >>> 56) & 0xFF),
                (byte) ((v >>> 48) & 0xFF),
                (byte) ((v >>> 40) & 0xFF),
                (byte) ((v >>> 32) & 0xFF),
                (byte) ((v >>> 24) & 0xFF),
                (byte) ((v >>> 16) & 0xFF),
                (byte) ((v >>> 8) & 0xFF),
                (byte) (v & 0xFF)
        };
    }
}
