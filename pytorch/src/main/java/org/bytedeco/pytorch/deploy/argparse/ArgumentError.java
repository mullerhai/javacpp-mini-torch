/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse;

/**
 * Error raised when an argument cannot be created or used.
 * <p>Mirrors Python {@code argparse.ArgumentError}. The {@link #getMessage()} method
 * yields a Python-style formatted string such as
 * {@code "argument --foo: conflicting option string: --foo"}.
 */
public class ArgumentError extends RuntimeException {

    private final String argumentName;

    public ArgumentError(Object argument, String message) {
        super(formatMessage(argument, message));
        this.argumentName = ArgumentNameExtractor.nameOf(argument);
    }

    public ArgumentError(String message) {
        super(formatMessage(null, message));
        this.argumentName = null;
    }

    public String getArgumentName() {
        return argumentName;
    }

    private static String formatMessage(Object argument, String message) {
        String name = ArgumentNameExtractor.nameOf(argument);
        if (name == null) {
            return message == null ? "" : message;
        }
        return "argument " + name + ": " + message;
    }

    /** Friendly detail line for embedding in tests/logs. */
    public String detail() {
        return super.getMessage();
    }
}