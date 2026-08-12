/*
 * Principal -- authenticated identity for cache access checks.
 *
 * <p>Production caches are typically fronted by a service mesh that injects
 * mTLS peer identity into every request. The cache layer accepts that
 * principal as a structured object rather than re-parsing the token.
 */
package org.bytedeco.pytorch.cache.security;

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

public final class Principal {

    public static final Principal SYSTEM = new Principal(
            "system", "system", Collections.singleton(Role.SYSTEM));

    private final String id;        // stable unique identifier (subject DN, SPIFFE ID)
    private final String display;   // human-readable name
    private final Set<Role> roles;
    private final Long expiresAtMs; // 0 = no expiry

    public Principal(String id, String display, Set<Role> roles) {
        this(id, display, roles, 0L);
    }

    public Principal(String id, String display, Set<Role> roles, long expiresAtMs) {
        this.id = Objects.requireNonNull(id);
        this.display = display == null ? id : display;
        this.roles = Collections.unmodifiableSet(new LinkedHashSet<>(roles));
        this.expiresAtMs = expiresAtMs;
    }

    public String id() { return id; }
    public String display() { return display; }
    public Set<Role> roles() { return roles; }
    public long expiresAtMs() { return expiresAtMs; }

    public boolean hasRole(Role r) { return roles.contains(r); }

    public boolean isExpired(long nowMs) {
        return expiresAtMs > 0 && nowMs >= expiresAtMs;
    }

    public Principal withRole(Role r) {
        Set<Role> ns = new LinkedHashSet<>(roles);
        ns.add(r);
        return new Principal(id, display, ns, expiresAtMs);
    }

    @Override public int hashCode() { return id.hashCode(); }

    @Override
    public boolean equals(Object o) {
        return o instanceof Principal && ((Principal) o).id.equals(id);
    }

    @Override public String toString() { return "Principal(" + id + "," + roles + ")"; }
}
