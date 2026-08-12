/*
 * AesGcmEncryptionPolicy -- AES-GCM-256 encryption with key rotation.
 *
 * <p>Uses the JCE provider that ships with the JDK (no external dependencies).
 * For non-JDK environments (Android < 26, native image), swap this with
 * ChaCha20-Poly1305 -- the wire format is identical thanks to the
 * {@link EncryptedEnvelope} framing.
 *
 * <p>IV handling: 12-byte random IV per encrypt (RFC 5116 recommended). The
 * 16-byte GCM tag is concatenated to the ciphertext.
 *
 * <p>Key rotation: rotateKey() installs a new key; the old key is kept
 * available for decrypting existing envelopes. The active key version is
 * recorded in every envelope so decrypt dispatches to the right key.
 */
package org.bytedeco.pytorch.cache.security;

import javax.crypto.Cipher;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.security.SecureRandom;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

public final class AesGcmEncryptionPolicy implements EncryptionPolicy {

    public static final int IV_LEN = 12;
    public static final int TAG_LEN_BITS = 128;

    private final String name;
    private final AtomicLong activeVersion = new AtomicLong(0);
    private volatile SecretKeySpec activeKey;
    private final Map<String, SecretKeySpec> historicalKeys = new HashMap<>();
    private final SecureRandom rng = new SecureRandom();

    public AesGcmEncryptionPolicy(byte[] initialKey) {
        this("AES-GCM-256", initialKey, "v1");
    }

    public AesGcmEncryptionPolicy(String name, byte[] initialKey, String initialKeyVersion) {
        if (initialKey == null || initialKey.length != 16 && initialKey.length != 24 && initialKey.length != 32)
            throw new IllegalArgumentException("AES key must be 16/24/32 bytes");
        this.name = name;
        this.activeKey = new SecretKeySpec(initialKey, "AES");
        this.activeVersion.set(1);
        historicalKeys.put(initialKeyVersion, this.activeKey);
    }

    @Override public String name() { return name; }

    @Override
    public EncryptedEnvelope encrypt(byte[] plaintext, byte[] aad) throws Exception {
        if (plaintext == null) throw new IllegalArgumentException("plaintext==null");
        byte[] iv = new byte[IV_LEN];
        rng.nextBytes(iv);
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        cipher.init(Cipher.ENCRYPT_MODE, activeKey, new GCMParameterSpec(TAG_LEN_BITS, iv));
        if (aad != null && aad.length > 0) cipher.updateAAD(aad);
        byte[] ct = cipher.doFinal(plaintext);
        // Last 16 bytes are the GCM tag
        int plen = ct.length - 16;
        byte[] body = new byte[plen];
        byte[] tag = new byte[16];
        System.arraycopy(ct, 0, body, 0, plen);
        System.arraycopy(ct, plen, tag, 0, 16);
        return new EncryptedEnvelope(EncryptedEnvelope.VERSION_1, currentVersion(), iv, body, tag);
    }

    @Override
    public byte[] decrypt(EncryptedEnvelope envelope, byte[] aad) throws Exception {
        if (envelope == null) throw new IllegalArgumentException("envelope==null");
        SecretKeySpec key = historicalKeys.get(envelope.keyVersion());
        if (key == null) throw new SecurityException("no key for version " + envelope.keyVersion());
        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        GCMParameterSpec spec = new GCMParameterSpec(TAG_LEN_BITS, envelope.iv());
        cipher.init(Cipher.DECRYPT_MODE, key, spec);
        if (aad != null && aad.length > 0) cipher.updateAAD(aad);
        byte[] joined = new byte[envelope.ciphertext().length + envelope.tag().length];
        System.arraycopy(envelope.ciphertext(), 0, joined, 0, envelope.ciphertext().length);
        System.arraycopy(envelope.tag(), 0, joined, envelope.ciphertext().length, envelope.tag().length);
        return cipher.doFinal(joined);
    }

    @Override
    public synchronized void rotateKey(byte[] newKeyMaterial) {
        if (newKeyMaterial == null || newKeyMaterial.length != 16 && newKeyMaterial.length != 24 && newKeyMaterial.length != 32)
            throw new IllegalArgumentException("AES key must be 16/24/32 bytes");
        long v = activeVersion.incrementAndGet();
        String ver = "v" + v;
        SecretKeySpec next = new SecretKeySpec(newKeyMaterial, "AES");
        activeKey = next;
        historicalKeys.put(ver, next);
    }

    @Override
    public String activeKeyVersion() { return currentVersion(); }

    private String currentVersion() {
        return "v" + activeVersion.get();
    }
}
