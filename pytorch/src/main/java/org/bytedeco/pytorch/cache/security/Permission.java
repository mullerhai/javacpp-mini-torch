/*
 * Permission -- fine-grained action bitmask.
 *
 * <p>Splits role from action so a tenant scope can grant READ:user_features
 * but not READ:model_metadata without inventing a new role.
 */
package org.bytedeco.pytorch.cache.security;

import java.util.EnumSet;
import java.util.Set;

public enum Permission {
    READ(1 << 0),
    WRITE(1 << 1),
    DELETE(1 << 2),
    INVALIDATE(1 << 3),
    ADMIN(1 << 4),
    AUDIT_READ(1 << 5),
    KEY_ROTATE(1 << 6);

    private final int mask;

    Permission(int mask) { this.mask = mask; }

    public int mask() { return mask; }

    public static Set<Permission> fromMask(int m) {
        EnumSet<Permission> s = EnumSet.noneOf(Permission.class);
        for (Permission p : values()) {
            if ((m & p.mask()) != 0) s.add(p);
        }
        return s;
    }

    public static int toMask(Set<Permission> perms) {
        int m = 0;
        for (Permission p : perms) m |= p.mask();
        return m;
    }
}
