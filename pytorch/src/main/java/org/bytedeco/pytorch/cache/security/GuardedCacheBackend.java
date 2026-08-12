/*
 * GuardedCacheBackend -- decorator that wraps a {@link CacheBackend} and
 * gates every operation through an {@link AccessController} plus an
 * optional {@link AuditLog}.
 *
 * <p>Failure mode: any denial is surfaced as {@link AccessDeniedException};
 * missing keys are not denied (they are returned as Optional.empty()), so
 * the decorator is a drop-in for the standard SPI.
 */
package org.bytedeco.pytorch.cache.security;

import org.bytedeco.pytorch.cache.CacheBackend;
import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

public final class GuardedCacheBackend implements CacheBackend {

    private final CacheBackend delegate;
    private final AccessController controller;
    private final AuditLog auditLog;
    private final Principal principal;
    private final String name;

    public GuardedCacheBackend(CacheBackend delegate, AccessController controller,
                               AuditLog auditLog, Principal principal, String name) {
        this.delegate = delegate;
        this.controller = controller == null ? AccessController.permitAll() : controller;
        this.auditLog = auditLog;
        this.principal = principal == null ? Principal.SYSTEM : principal;
        this.name = name == null ? delegate.name() : name;
    }

    public static GuardedCacheBackend of(CacheBackend delegate, Principal p) {
        return new GuardedCacheBackend(delegate, AccessController.permitAll(), null, p, delegate.name());
    }

    @Override public String name() { return "guarded:" + name; }
    @Override public int tier() { return delegate.tier(); }

    @Override
    public Optional<CacheValue<Object>> get(CacheKey key) {
        controller.check(principal, AccessController.Action.READ, key);
        Optional<CacheValue<Object>> v = delegate.get(key);
        audit("GET", key, v.isPresent());
        return v;
    }

    @Override
    public Map<CacheKey, CacheValue<Object>> getBatch(Collection<CacheKey> keys) {
        Map<CacheKey, CacheValue<Object>> out = new LinkedHashMap<>();
        if (keys == null) return out;
        for (CacheKey k : keys) {
            try {
                Optional<CacheValue<Object>> v = get(k);
                v.ifPresent(val -> out.put(k, val));
            } catch (AccessDeniedException ade) {
                // skip denied keys silently rather than poisoning the batch
            }
        }
        return out;
    }

    @Override
    public void put(CacheKey key, CacheValue<Object> value) {
        controller.check(principal, AccessController.Action.WRITE, key);
        delegate.put(key, value);
        audit("PUT", key, true);
    }

    @Override
    public void delete(CacheKey key) {
        controller.check(principal, AccessController.Action.DELETE, key);
        delegate.delete(key);
        audit("DELETE", key, true);
    }

    @Override
    public long size() {
        controller.check(principal, AccessController.Action.ADMIN, null);
        return delegate.size();
    }

    @Override
    public boolean ping() {
        return delegate.ping();
    }

    private void audit(String action, CacheKey key, boolean success) {
        if (auditLog == null) return;
        auditLog.append(AuditLog.Entry.builder()
                .principal(principal.id())
                .action(action)
                .resource(key == null ? "<null>" : key.toStorageKey())
                .success(success));
    }
}
