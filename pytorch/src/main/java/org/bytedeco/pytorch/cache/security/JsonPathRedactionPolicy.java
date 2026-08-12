/*
 * JsonPathRedactionPolicy -- redaction policy that matches a configured list
 * of "dot-paths" (e.g. user.email, user.phone, *.token) and replaces the
 * matched scalar with {@link #REDACTED}.
 *
 * <p>Constraints:
 * <ul>
 *   <li>toString-based parser -- supports Map<String,Object> and List<Object>
 *       trees, as produced by JSON deserialisers like Jackson</li>
 *   <li>does not allocate a new parser; cheap enough for hot-path redaction</li>
 *   <li>limited to scalar leaves (numerics, booleans, strings); structural
 *       traversal is supported but replacement is shallow</li>
 * </ul>
 */
package org.bytedeco.pytorch.cache.security;

import org.bytedeco.pytorch.cache.CacheKey;
import org.bytedeco.pytorch.cache.CacheValue;

import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

public final class JsonPathRedactionPolicy implements RedactionPolicy {

    public static final String REDACTED = "[REDACTED]";

    private final Set<String> paths;
    private final Pattern compiled;

    public JsonPathRedactionPolicy(Collection<String> dottedPaths) {
        this.paths = new LinkedHashSet<>(dottedPaths);
        StringBuilder re = new StringBuilder();
        boolean first = true;
        for (String p : paths) {
            if (!first) re.append('|');
            first = false;
            String escaped = Pattern.quote(p).replace("*", "\\E[^.]+\\Q");
            re.append(escaped);
        }
        this.compiled = paths.isEmpty() ? null : Pattern.compile(re.toString());
    }

    @Override public String name() { return "json-path"; }

    @Override
    public CacheValue<Object> apply(CacheKey key, CacheValue<Object> value) {
        if (value == null || value.value() == null || paths.isEmpty()) return value;
        Object redacted = walk(value.value(), "", false);
        if (redacted == value.value()) return value;
        return value.toBuilder().value(redacted).tag("redacted", "true").build();
    }

    private Object walk(Object node, String path, boolean parentWasList) {
        if (node instanceof Map) {
            Map<Object, Object> m = (Map<Object, Object>) node;
            for (Map.Entry<Object, Object> e : m.entrySet()) {
                String key = String.valueOf(e.getKey());
                String childPath = parentWasList ? path : (path.isEmpty() ? key : path + "." + key);
                Object child = e.getValue();
                if (matches(childPath)) {
                    m.put(key, REDACTED);
                } else if (child instanceof Map || child instanceof List) {
                    walk(child, childPath, child instanceof List);
                }
            }
            return node;
        }
        if (node instanceof List) {
            List<Object> l = (List<Object>) node;
            for (int i = 0; i < l.size(); i++) {
                Object child = l.get(i);
                String childPath = path + "[" + i + "]";
                if (matches(childPath)) {
                    l.set(i, REDACTED);
                } else if (child instanceof Map || child instanceof List) {
                    walk(child, childPath, child instanceof List);
                }
            }
            return node;
        }
        return node;
    }

    private boolean matches(String path) {
        if (compiled == null) return false;
        return compiled.matcher(path).matches();
    }
}
