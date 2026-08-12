/*
 * AccessController -- SPI that decides whether a (Principal, Action, Resource)
 * triple is allowed.
 *
 * <p>Implementations are stateless and reentrant; the framework calls them on
 * every get/put/delete. They are expected to be cheap (deny-by-default, then
 * white-list).
 */
package org.bytedeco.pytorch.cache.security;

import org.bytedeco.pytorch.cache.CacheKey;

public interface AccessController {

    void check(Principal principal, Action action, CacheKey resource) throws AccessDeniedException;

    enum Action {
        READ(0),
        WRITE(0),
        DELETE(0),
        INVALIDATE(0),
        ADMIN(0),
        AUDIT(0),
        KEY_ROTATE(0);

        private final int unused;
        Action(int unused) { this.unused = unused; }
    }

    static AccessController permitAll() {
        return (p, a, r) -> {};
    }

    static AccessController denyAll() {
        return (p, a, r) -> {
            throw new AccessDeniedException(
                    AccessDeniedException.Reason.MISSING_PERMISSION,
                    "deny-all: " + a + " on " + r, p == null ? "anonymous" : p.id(), String.valueOf(r));
        };
    }

    static AccessController roleBased(Role minRole) {
        return (principal, action, resource) -> {
            if (principal == null) {
                throw new AccessDeniedException(
                        AccessDeniedException.Reason.MISSING_ROLE,
                        "no principal", "anonymous", String.valueOf(resource));
            }
            if (principal.hasRole(Role.SYSTEM)) return;
            if (principal.hasRole(Role.ADMIN)) return;
            if (principal.hasRole(Role.AUDITOR) && action == Action.AUDIT) return;
            if (!principal.hasRole(minRole)) {
                throw new AccessDeniedException(
                        AccessDeniedException.Reason.MISSING_ROLE,
                        "requires " + minRole + " but has " + principal.roles(),
                        principal.id(), String.valueOf(resource));
            }
        };
    }

    private static boolean impliesRole(Role have, Role want) {
        // WRITE implies READ; ADMIN implies all (handled above); AUDITOR implies READ only for AUDIT.
        if (have == want) return true;
        if (want == Role.READ && have == Role.WRITE) return true;
        return false;
    }

    static AccessController roleBasedWithImplication(Role minRole) {
        return (principal, action, resource) -> {
            if (principal == null) {
                throw new AccessDeniedException(
                        AccessDeniedException.Reason.MISSING_ROLE,
                        "no principal", "anonymous", String.valueOf(resource));
            }
            if (principal.hasRole(Role.SYSTEM)) return;
            if (principal.hasRole(Role.ADMIN)) return;
            if (principal.hasRole(Role.AUDITOR) && action == Action.AUDIT) return;
            boolean ok = false;
            for (Role r : principal.roles()) {
                if (impliesRole(r, minRole)) { ok = true; break; }
            }
            if (!ok) {
                throw new AccessDeniedException(
                        AccessDeniedException.Reason.MISSING_ROLE,
                        "requires " + minRole + " but has " + principal.roles(),
                        principal.id(), String.valueOf(resource));
            }
        };
    }

    default AccessController and(AccessController other) {
        AccessController self = this;
        return (p, a, r) -> { self.check(p, a, r); other.check(p, a, r); };
    }
}
