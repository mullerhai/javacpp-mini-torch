/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.internal;

/**
 * Bridge for retrieving "default" argv. Mirrors Python's implicit
 * {@code sys.argv[1:]} fallback.
 *
 * <p>Java has no reliable {@code sys.argv}; by default we look at:
 * <ol>
 *   <li>the {@code argparse.args} system property (comma-separated),</li>
 *   <li>{@code Thread.currentThread().getName()} extras (best effort),</li>
 *   <li>an empty list.</li>
 * </ol>
 *
 * <p>Most users will pass arguments explicitly via {@code parseArgs(String[])},
 * so this is a niche helper used only when {@code parseArgs()} is called with
 * no args.
 */
public final class SystemArgsBridge {
    private SystemArgsBridge() {}

    public static java.util.List<String> getSystemArgs() {
        String prop = System.getProperty("argparse.args");
        if (prop != null && !prop.isEmpty()) {
            String[] arr = prop.split("\\s+");
            return new java.util.ArrayList<>(java.util.Arrays.asList(arr));
        }
        return new java.util.ArrayList<>();
    }
}