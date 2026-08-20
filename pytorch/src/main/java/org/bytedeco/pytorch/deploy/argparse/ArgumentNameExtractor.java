/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * Tiny helper that computes Python's {@code _get_action_name} semantics:
 * the displayed name of an action used in error/help strings.
 *
 * <p>This package-private helper intentionally lives outside the main API
 * to avoid leaking implementation choices.
 */
final class ArgumentNameExtractor {

    private ArgumentNameExtractor() {}

    static String nameOf(Object argument) {
        if (argument == null) {
            return null;
        }
        try {
            java.lang.reflect.Method getOptionStrings = argument.getClass().getMethod("getOptionStrings");
            Object val = getOptionStrings.invoke(argument);
            if (val instanceof List<?> list) {
                if (!list.isEmpty()) {
                    return list.stream().map(Object::toString).collect(Collectors.joining("/"));
                }
            } else if (val instanceof String[] arr) {
                if (arr.length > 0) {
                    return String.join("/", arr);
                }
            }
        } catch (ReflectiveOperationException ignored) {
            // not an Action
        }
        try {
            java.lang.reflect.Method getMetavar = argument.getClass().getMethod("getMetavar");
            Object val = getMetavar.invoke(argument);
            String metavar = val == null ? null : val.toString();
            if (metavar != null && !ArgparseConstants.SUPPRESS.equals(metavar)) {
                return metavar;
            }
        } catch (ReflectiveOperationException ignored) {
            // not an Action
        }
        try {
            java.lang.reflect.Method getDest = argument.getClass().getMethod("getDest");
            Object val = getDest.invoke(argument);
            String dest = val == null ? null : val.toString();
            if (dest != null && !ArgparseConstants.SUPPRESS.equals(dest)) {
                return dest;
            }
        } catch (ReflectiveOperationException ignored) {
            // not an Action
        }
        try {
            java.lang.reflect.Method getChoices = argument.getClass().getMethod("getChoices");
            Object val = getChoices.invoke(argument);
            if (val instanceof Iterable<?> it) {
                String joined = StreamSupport.stream(it.spliterator(), false)
                        .map(Object::toString)
                        .collect(Collectors.joining(","));
                return "{" + joined + "}";
            }
        } catch (ReflectiveOperationException ignored) {
            // not an Action
        }
        return null;
    }
}