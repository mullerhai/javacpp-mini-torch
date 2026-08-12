/*
 * CacheAcl -- per-entry access control list.
 *
 * <p>Carried alongside the cached value as a structured field. The
 * {@link AccessController} consults {@link CacheAcl#allowedPrincipals} and
 * {@link CacheAcl#requiredPermissions} before returning data.
 *
 * <p>Empty ACL = open to any authenticated principal (default).
 * Null ACL = use the cache-level default policy.
 *
 * <p>ACLs are themselves covered by the entry's signature so a malicious
 * operator cannot rewrite them.
 */
package org.bytedeco.pytorch.cache.security;

import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.Set;

public final class CacheAcl {

    public static final CacheAcl PUBLIC = new CacheAcl(
            Collections.emptySet(), EnumSet.of(Permission.READ), false);

    private final Set<String> allowedPrincipals;   // empty = no principal filter
    private final Set<Permission> requiredPermissions;
    private final boolean sealed;                  // once sealed, no bypass

    public CacheAcl(Set<String> allowedPrincipals, Set<Permission> requiredPermissions, boolean sealed) {
        this.allowedPrincipals = Collections.unmodifiableSet(new LinkedHashSet<>(allowedPrincipals));
        this.requiredPermissions = requiredPermissions == null
                ? EnumSet.of(Permission.READ)
                : Collections.unmodifiableSet(EnumSet.copyOf(requiredPermissions));
        this.sealed = sealed;
    }

    public static CacheAcl allow(String... principals) {
        Set<String> p = new LinkedHashSet<>();
        Collections.addAll(p, principals);
        return new CacheAcl(p, EnumSet.of(Permission.READ), false);
    }

    public static CacheAcl require(Set<Permission> perms) {
        return new CacheAcl(Collections.emptySet(), perms, false);
    }

    public static CacheAcl sealedFor(String principal, Set<Permission> perms) {
        Set<String> p = new LinkedHashSet<>();
        p.add(principal);
        return new CacheAcl(p, perms, true);
    }

    public Set<String> allowedPrincipals() { return allowedPrincipals; }
    public Set<Permission> requiredPermissions() { return requiredPermissions; }
    public boolean sealed() { return sealed; }

    public boolean isPublic() {
        return allowedPrincipals.isEmpty() && !sealed;
    }

    public CacheAcl withPermissions(Set<Permission> perms) {
        return new CacheAcl(allowedPrincipals, perms, sealed);
    }

    @Override
    public String toString() {
        return "CacheAcl{principals=" + allowedPrincipals
                + ", perms=" + requiredPermissions + ", sealed=" + sealed + "}";
    }
}
