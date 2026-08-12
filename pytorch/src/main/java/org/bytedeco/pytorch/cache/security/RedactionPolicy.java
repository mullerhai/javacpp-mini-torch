/*
 * RedactionPolicy -- strip sensitive fields from cached values before
 * persistence or external exposure.
 *
 * <p>Two reference implementations:
 * <ul>
 *   <li>{@link JsonPathRedactionPolicy} -- uses a simple JSON path matcher
 *       (no third-party dependency; covers common ${tenant}.user.email style paths)</li>
 *   <li>{@link MetadataTagRedactionPolicy} -- matches against a key's tags
 *       (declared at write time)</li>
 * </ul>
 *
 * <p>Output: a new {@link CacheValue} whose value is the redacted payload and
 * whose tags include a `redacted=true` flag so downstream consumers know the
 * value has been altered.
 */
package org.bytedeco.pytorch.cache.security;

import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;

public interface RedactionPolicy {

    /** Stable label. */
    String name();

    /** Returns true if the policy redacted anything in the value. */
    CacheValue<Object> apply(CacheKey key, CacheValue<Object> value);

    static RedactionPolicy identity() {
        return new RedactionPolicy() {
            @Override public String name() { return "identity"; }
            @Override public CacheValue<Object> apply(CacheKey k, CacheValue<Object> v) { return v; }
        };
    }
}
