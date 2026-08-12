/*
 * Role -- coarse-grained role names for cache authorisation.
 *
 * <p>Roles are kept narrow on purpose to avoid tangled RBAC graphs:
 * <ul>
 *   <li>READ -- read values, no mutation</li>
 *   <li>WRITE -- read + write values, no policy changes</li>
 *   <li>ADMIN -- cache administration (invalidate, replay, sign)</li>
 *   <li>SYSTEM -- internal role for the cache layer itself (loaders, refreshers)</li>
 *   <li>AUDITOR -- read + audit log access, no data mutation</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache.security;

public enum Role {
    READ,
    WRITE,
    ADMIN,
    SYSTEM,
    AUDITOR
}
