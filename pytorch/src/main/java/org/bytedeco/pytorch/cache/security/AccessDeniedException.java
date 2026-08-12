/*
 * AccessDeniedException -- thrown when a principal is not allowed to perform
 * the requested action on a cache entry or tier.
 */
package org.bytedeco.pytorch.cache.security;

public class AccessDeniedException extends SecurityException {

    public enum Reason {
        MISSING_ROLE,
        MISSING_PERMISSION,
        PRINCIPAL_EXPIRED,
        ACL_SEALED,
        KEY_NOT_FOUND
    }

    private final Reason reason;
    private final String principal;
    private final String resource;

    public AccessDeniedException(Reason reason, String message, String principal, String resource) {
        super(message);
        this.reason = reason;
        this.principal = principal;
        this.resource = resource;
    }

    public Reason reason() { return reason; }
    public String principal() { return principal; }
    public String resource() { return resource; }
}
