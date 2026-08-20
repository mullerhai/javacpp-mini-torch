/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Simple object for storing attributes.
 *
 * <p>Mirrors Python's {@code argparse.Namespace}. Construct with
 * {@code Namespace.stringMap} or {@link #set(String, Object)} / {@link #get(String)}.
 * Equality follows Python semantics: equal iff their {@link #asMap()} are equal.
 *
 * <p>Also implements {@code __contains__} via {@link #has(String)} to enable
 * {@code ns.containsKey("foo")} / Python {@code "foo" in ns}.
 */
public final class Namespace extends _AttributeHolder {

    private final LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();

    public Namespace() {}

    public Namespace(Map<String, Object> initial) {
        if (initial != null) {
            for (Map.Entry<String, Object> e : initial.entrySet()) {
                attrs.put(e.getKey(), e.getValue());
            }
        }
    }

    public Namespace set(String name, Object value) {
        attrs.put(name, value);
        return this;
    }

    public Object get(String name) {
        return attrs.get(name);
    }

    @SuppressWarnings("unchecked")
    public <T> T getAs(String name) {
        return (T) attrs.get(name);
    }

    public boolean has(String name) {
        return attrs.containsKey(name);
    }

    public Map<String, Object> asMap() {
        return new LinkedHashMap<>(attrs);
    }

    public Map<String, Object> asUnmodifiableMap() {
        return java.util.Collections.unmodifiableMap(new LinkedHashMap<>(attrs));
    }

    @Override
    protected Iterable<Map.Entry<String, Object>> getKwargs() {
        return attrs.entrySet();
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Namespace other)) return false;
        return attrs.equals(other.attrs);
    }

    @Override
    public int hashCode() {
        return attrs.hashCode();
    }
}