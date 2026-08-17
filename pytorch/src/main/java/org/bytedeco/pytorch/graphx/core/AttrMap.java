/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.core;
import org.bytedeco.pytorch.jit.*;

import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Attribute storage for nodes/edges. Mirrors NetworkX's keyword-attribute dictionary
 * semantics: any {@code (k, v)} pair can be attached; well-known keys ({@code weight},
 * {@code color}, {@code label}) are exposed via typed accessors.
 *
 * <p>This is a <b>memory-efficient</b> alternative to allocating a HashMap when no
 * attributes are present: an empty AttrMap is a singleton.
 */
public final class AttrMap {
    private static final AttrMap EMPTY = new AttrMap(Collections.emptyMap());

    private final Map<String, Object> data;

    private AttrMap(Map<String, Object> data) {
        this.data = data;
    }

    public static AttrMap empty() { return EMPTY; }

    /** Build an AttrMap with a copy of {@code initial}. */
    public static AttrMap of(Map<String, Object> initial) {
        if (initial == null || initial.isEmpty()) return EMPTY;
        return new AttrMap(new LinkedHashMap<>(initial));
    }

    public static Builder builder() { return new Builder(); }

    public boolean isEmpty() { return data.isEmpty(); }
    public int size() { return data.size(); }

    public Object get(String key) { return data.get(key); }
    public String getString(String key) {
        Object v = data.get(key);
        return v == null ? null : v.toString();
    }
    public Double getDouble(String key) {
        Object v = data.get(key);
        if (v == null) return null;
        if (v instanceof Number) return ((Number) v).doubleValue();
        if (v instanceof String) try { return Double.parseDouble((String) v); } catch (Exception ignored) { return null; }
        return null;
    }
    public Integer getInt(String key) {
        Object v = data.get(key);
        if (v == null) return null;
        if (v instanceof Number) return ((Number) v).intValue();
        if (v instanceof String) try { return Integer.parseInt((String) v); } catch (Exception ignored) { return null; }
        return null;
    }
    public boolean getBoolean(String key, boolean def) {
        Object v = data.get(key);
        if (v == null) return def;
        if (v instanceof Boolean) return (Boolean) v;
        if (v instanceof Number) return ((Number) v).intValue() != 0;
        return def;
    }

    public boolean contains(String key) { return data.containsKey(key); }

    public Map<String, Object> asMap() {
        return Collections.unmodifiableMap(data);
    }

    public Set<String> keys() { return data.keySet(); }

    /** Returns a copy with {@code key} set to {@code value}. */
    public AttrMap with(String key, Object value) {
        LinkedHashMap<String, Object> m2 = new LinkedHashMap<>(data);
        m2.put(key, value);
        return new AttrMap(m2);
    }

    /** Returns a copy without {@code key}. */
    public AttrMap without(String key) {
        if (!data.containsKey(key)) return this;
        LinkedHashMap<String, Object> m2 = new LinkedHashMap<>(data);
        m2.remove(key);
        return m2.isEmpty() ? EMPTY : new AttrMap(m2);
    }

    public AttrMap merged(AttrMap other) {
        if (other == null || other.data.isEmpty()) return this;
        if (this.data.isEmpty()) return other;
        LinkedHashMap<String, Object> m2 = new LinkedHashMap<>(this.data);
        m2.putAll(other.data);
        return new AttrMap(m2);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof AttrMap)) return false;
        return Objects.equals(data, ((AttrMap) o).data);
    }

    @Override
    public int hashCode() {
        return data.hashCode();
    }

    @Override
    public String toString() {
        return "AttrMap" + data;
    }

    public static final class Builder {
        private final LinkedHashMap<String, Object> m = new LinkedHashMap<>();

        public Builder put(String key, Object value) { m.put(key, value); return this; }
        public Builder putAll(Map<String, Object> src) {
            if (src != null) m.putAll(src);
            return this;
        }
        public boolean contains(String key) { return m.containsKey(key); }
        public Builder remove(String key) { m.remove(key); return this; }

        public AttrMap build() {
            return m.isEmpty() ? EMPTY : new AttrMap(m);
        }
    }

    /** Internal: package-private accessor for serialization/random access. */
    Map<String, Object> rawData() { return data; }
}