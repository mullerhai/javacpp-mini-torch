/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.internal;

/** Identity function used as the default {@code type=} callable. */
public final class Identity {
    private Identity() {}

    /** Mirrors the expected "callable" interface used by {@code getValue()}. */
    public static Object call(Object s) {
        return s == null ? null : s.toString();
    }

    static Object apply(Object s) {
        return call(s);
    }
}