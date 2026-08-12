/*
 * SecureCacheWriter -- composes encryption, signing, redaction, and audit
 * into a single write/read façade.
 *
 * <p>Write path:
 * <pre>
 *   value -> redact -> sign -> [optionally encrypt signed payload] -> backend.put
 * </pre>
 *
 * <p>Read path:
 * <pre>
 *   backend.get -> [decrypt] -> verify -> return
 * </pre>
 *
 * <p>If verification fails the entry is dropped (not returned to the caller)
 * and an audit entry is written so the security team can investigate.
 */
package org.bytedeco.pytorch.cache.security;

import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;

import java.util.Optional;

public final class SecureCacheWriter {

    private final EncryptionPolicy encryption;     // optional
    private final IntegritySigner signer;          // optional
    private final RedactionPolicy redaction;       // optional
    private final AuditLog auditLog;               // optional
    private final SecureSerde serde = new SecureSerde();

    public SecureCacheWriter(EncryptionPolicy encryption, IntegritySigner signer,
                             RedactionPolicy redaction, AuditLog auditLog) {
        this.encryption = encryption;
        this.signer = signer;
        this.redaction = redaction == null ? RedactionPolicy.identity() : redaction;
        this.auditLog = auditLog;
    }

    public static SecureCacheWriter plain() {
        return new SecureCacheWriter(null, null, null, null);
    }

    public CacheValue<Object> prepareForWrite(CacheKey key, CacheValue<Object> value) {
        if (value == null) return null;
        CacheValue<Object> v = redaction.apply(key, value);
        if (signer != null) {
            String token = signer.sign(key, v);
            v = v.toBuilder().tag("integrity", token).build();
        }
        if (encryption != null) {
            try {
                byte[] plain = serde.encode(v);
                EncryptedEnvelope env = encryption.encrypt(plain, keyBytes(key));
                v = v.toBuilder().value(env.serialize()).tag("encrypted", "true").build();
            } catch (Exception e) {
                throw new SecurityException("encryption failed", e);
            }
        }
        return v;
    }

    public Optional<CacheValue<Object>> verifyAfterRead(CacheKey key, CacheValue<Object> stored) {
        if (stored == null) return Optional.empty();
        if (stored.tag("encrypted") != null && encryption != null) {
            try {
                EncryptedEnvelope env = EncryptedEnvelope.deserialize((byte[]) stored.value());
                byte[] plain = encryption.decrypt(env, keyBytes(key));
                stored = serde.decode(plain, stored);
            } catch (Exception e) {
                audit("DECRYPT_FAIL", key, false);
                return Optional.empty();
            }
        }
        if (signer != null) {
            String token = stored.tag("integrity");
            if (token == null || !signer.verify(key, stored, token)) {
                audit("TAMPER", key, false);
                return Optional.empty();
            }
        }
        return Optional.of(stored);
    }

    private void audit(String action, CacheKey key, boolean success) {
        if (auditLog == null) return;
        auditLog.append(AuditLog.Entry.builder()
                .action(action)
                .resource(key == null ? "<null>" : key.toStorageKey())
                .success(success));
    }

    private static byte[] keyBytes(CacheKey k) {
        return k == null ? new byte[0] : k.toStorageKey().getBytes(java.nio.charset.StandardCharsets.UTF_8);
    }
}
