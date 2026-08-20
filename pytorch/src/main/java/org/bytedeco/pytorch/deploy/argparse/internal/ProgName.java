/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.internal;

/**
 * Mirrors Python {@code _prog_name}: figures out a sensible default program
 * name when the caller does not pass {@code prog=}.
 *
 * <p>Resolution order:
 * <ol>
 *   <li>{@code -Dargparse.prog=...} JVM property (highest priority),</li>
 *   <li>{@code Thread.currentThread().getStackTrace()[2]} (best effort),</li>
 *   <li>"argparse" as fallback.</li>
 * </ol>
 *
 * <p>This avoids depending on {@code sys.argv[0]} semantics, which is
 * unreliable when called from a server / library entry point.
 */
public final class ProgName {

    private ProgName() {}

    public static String resolve(String explicit) {
        if (explicit != null && !explicit.isEmpty()) return explicit;
        String prop = System.getProperty("argparse.prog");
        if (prop != null && !prop.isEmpty()) return prop;
        try {
            StackTraceElement[] trace = new Throwable().getStackTrace();
            for (int i = Math.min(2, trace.length - 1); i < trace.length; i++) {
                String cls = trace[i].getClassName();
                if (cls != null && !cls.startsWith("argparse.") && !cls.startsWith("java.")) {
                    String simple = cls.substring(cls.lastIndexOf('.') + 1);
                    if (!simple.isEmpty()) {
                        return simple.contains(".")
                                ? simple.substring(simple.lastIndexOf('.') + 1)
                                : simple;
                    }
                }
            }
        } catch (Exception ignored) {}
        return "argparse";
    }
}