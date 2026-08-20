/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.StringJoiner;
import java.util.TreeSet;

/**
 * Base class for objects that have a {@code __repr__}-style string and
 * ordered kwargs. Mirrors Python's {@code _AttributeHolder}.
 *
 * <p>Subclasses opt in by listing their public properties through
 * {@link #collectKwargNames()} (used by {@link #repr()} to print
 * {@code ClassName(field=val, field=val, ...)}).
 */
public abstract class _AttributeHolder {

    /**
     * Subclasses override to declare the attribute names to print.
     * The default implementation uses reflection on all declared getters.
     */
    protected List<String> collectKwargNames() {
        List<String> names = new ArrayList<>();
        for (Method m : getClass().getMethods()) {
            if (m.getParameterCount() == 0
                    && m.getName().startsWith("get")
                    && !m.getName().equals("getClass")) {
                String n = m.getName().substring(3);
                if (!n.isEmpty() && Character.isUpperCase(n.charAt(0))) {
                    names.add(Character.toLowerCase(n.charAt(0)) + n.substring(1));
                }
            }
        }
        Collections.sort(names);
        return names;
    }

    /** Iterates over attribute names and current values. */
    protected Iterable<Map.Entry<String, Object>> getKwargs() {
        Map<String, Object> map = new LinkedHashMap<>();
        for (String name : collectKwargNames()) {
            map.put(name, readAttribute(name));
        }
        return map.entrySet();
    }

    /** Read a declared attribute by name; subclass can override for richer lookup. */
    public Object readAttribute(String name) {
        String cap = Character.toUpperCase(name.charAt(0)) + name.substring(1);
        for (Method m : getClass().getMethods()) {
            if (m.getParameterCount() == 0
                    && (m.getName().equals("get" + cap) || m.getName().equals("is" + cap))) {
                try {
                    return m.invoke(this);
                } catch (ReflectiveOperationException ignored) {
                    return null;
                }
            }
        }
        return null;
    }

    /** Iterates over positional args. Default empty (only Namespace uses). */
    protected Iterable<Object> getArgs() {
        return Collections.emptyList();
    }

    @Override
    public String toString() {
        return repr();
    }

    /** Replicates Python {@code __repr__}. */
    public String repr() {
        String typeName = getClass().getSimpleName();
        List<String> argStrings = new ArrayList<>();
        Map<String, Object> starArgs = new LinkedHashMap<>();
        for (Object a : getArgs()) {
            argStrings.add(reprValue(a));
        }
        for (Map.Entry<String, Object> e : getKwargs()) {
            String name = e.getKey();
            Object value = e.getValue();
            if (isIdentifier(name)) {
                argStrings.add(name + "=" + reprValue(value));
            } else {
                starArgs.put(name, value);
            }
        }
        if (!starArgs.isEmpty()) {
            argStrings.add("**" + reprValue(starArgs));
        }
        return typeName + "(" + String.join(", ", argStrings) + ")";
    }

    static boolean isIdentifier(String s) {
        if (s == null || s.isEmpty()) return false;
        if (!Character.isJavaIdentifierStart(s.charAt(0))) return false;
        for (int i = 1; i < s.length(); i++) {
            if (!Character.isJavaIdentifierPart(s.charAt(i))) return false;
        }
        return true;
    }

    static String reprValue(Object o) {
        if (o == null) return "null";
        if (o instanceof String s) {
            return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
        }
        if (o instanceof Number || o instanceof Boolean) {
            return o.toString();
        }
        if (o instanceof char[] ca) {
            return new String(ca);
        }
        if (o instanceof Object[] arr) {
            StringJoiner sj = new StringJoiner(", ", "[", "]");
            for (Object e : arr) sj.add(reprValue(e));
            return sj.toString();
        }
        if (o instanceof Iterable<?> it) {
            StringJoiner sj = new StringJoiner(", ", "[", "]");
            for (Object e : it) sj.add(reprValue(e));
            return sj.toString();
        }
        if (o instanceof Map<?, ?> m) {
            StringJoiner sj = new StringJoiner(", ", "{", "}");
            TreeSet<String> keys = new TreeSet<>();
            for (Object k : m.keySet()) keys.add(reprValue(k));
            for (String k : keys) sj.add(k + ": " + reprValue(m.get(k)));
            return sj.toString();
        }
        return o.toString();
    }
}