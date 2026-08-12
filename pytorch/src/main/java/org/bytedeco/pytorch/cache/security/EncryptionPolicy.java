/*
 * EncryptionPolicy -- transparent at-rest encryption for cache values.
 *
 * <p>Design:
 * <ul>
 *   <li>operates on a serialised byte[] payload (so the caller controls framing)</li>
 *   <li>versioned envelope (see {@link EncryptedEnvelope}) so the key/version
 *       of the wrapping key is recorded alongside the ciphertext</li>
 *   <li>thread-safe</li>
 * </ul>
 *
 * <p>Production policies (AES-GCM, ChaCha20-Poly1305) live in siblings; this
 * interface is the SPI.
 */
package org.bytedeco.pytorch.cache.security;

public interface EncryptionPolicy {

    /** Stable label for diagnostics. */
    String name();

    /**
     * Encrypt {@code plaintext} under the current active key.
     *
     * @param aad  additional authenticated data (e.g. cache key); may be null
     */
    EncryptedEnvelope encrypt(byte[] plaintext, byte[] aad) throws Exception;

    /** Decrypt using the key version recorded in the envelope. */
    byte[] decrypt(EncryptedEnvelope envelope, byte[] aad) throws Exception;

    /** Rotate the active key; subsequent encrypts use the new key. */
    void rotateKey(byte[] newKeyMaterial);

    /** Identifier of the currently active key. */
    String activeKeyVersion();
}
