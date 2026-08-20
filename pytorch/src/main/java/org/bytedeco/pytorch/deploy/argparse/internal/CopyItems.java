/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.internal;

/**
 * Mirrors Python's {@code _copy_items}: deep-copies a list (shallow per element).
 * Used by {@code _AppendAction} / {@code _ExtendAction} to avoid mutating
 * default list values in place.
 */
public final class CopyItems {
    private CopyItems() {}

    public static <T> java.util.List<T> copy(java.util.List<T> items) {
        if (items == null) return new java.util.ArrayList<>();
        return new java.util.ArrayList<>(items);
    }
}