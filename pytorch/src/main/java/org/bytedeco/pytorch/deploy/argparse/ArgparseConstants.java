/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
package org.bytedeco.pytorch.deploy.argparse;

/**
 * Mirrors Python's argparse constants. Java reuses {@link String} and {@link Integer}
 * in place of Python's typed literals; we expose the same symbolic constants so user
 * code reads identically to the Python version.
 */
public final class ArgparseConstants {

    private ArgparseConstants() {}

    /** Allow zero or one command-line argument; equivalent to Python {@code '?'}. */
    public static final String OPTIONAL = "?";

    /** Allow zero or more command-line arguments; equivalent to Python {@code '*'}. */
    public static final String ZERO_OR_MORE = "*";

    /** Allow one or more command-line arguments; equivalent to Python {@code '+'}. */
    public static final String ONE_OR_MORE = "+";

    /** Consume the remaining arguments into another parser (subparsers sentinel). */
    public static final String PARSER = "A...";

    /** Consume all remaining arguments literally (no option parsing). */
    public static final String REMAINDER = "...";

    /** Sentinel that suppresses default/help printing. */
    public static final String SUPPRESS = "==SUPPRESS==";

    /** Internal attribute name holding unrecognized args, mirrors Python {@code _UNRECOGNIZED_ARGS_ATTR}. */
    public static final String UNRECOGNIZED_ARGS_ATTR = "_unrecognized_args";

    /** Library version, mirrors {@code __version__ = '1.1'}. */
    public static final String VERSION = "1.1";

    /** Mirrors Python's {@code __all__} (used in docs/javadoc only). */
    public static final String[] ALL = new String[] {
            "ArgumentParser",
            "ArgumentError",
            "ArgumentTypeError",
            "BooleanOptionalAction",
            "FileType",
            "HelpFormatter",
            "ArgumentDefaultsHelpFormatter",
            "RawDescriptionHelpFormatter",
            "RawTextHelpFormatter",
            "MetavarTypeHelpFormatter",
            "Namespace",
            "Action",
            "ONE_OR_MORE",
            "OPTIONAL",
            "PARSER",
            "REMAINDER",
            "SUPPRESS",
            "ZERO_OR_MORE",
    };
}