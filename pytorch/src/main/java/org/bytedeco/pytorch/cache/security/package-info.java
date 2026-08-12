/*
 * Security -- at-rest and in-flight security protections for cache tiers.
 *
 * <p>Composition is layered:
 * <pre>
 *   raw CacheValue
 *     -> enforcer (ACL guard, access controller)
 *     -> encryption (AES-GCM, key rotation)
 *     -> redaction (PII scrub)
 *     -> signing (HMAC for integrity)
 *     -> backend
 * </pre>
 *
 * <p>Design notes:
 * <ul>
 *   <li>Each layer is a policy/SPI so callers can swap e.g. AES-GCM for ChaCha20
 *       (mobile-class) without changing the wiring.</li>
 *   <li>Tamper-evident audit log uses a hash chain (Merkle-style) so an officer
 *       can verify the operator saw every get/put without gaps.</li>
 *   <li>Encryption is field-scoped via {@link EncryptionPolicy}; the framework
 *       never encrypts the cache key itself because keys are not user data.</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache.security;
