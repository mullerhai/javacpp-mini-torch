/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse;

/**
 * Error raised by a custom {@code type=} function when conversion fails.
 * <p>Mirrors Python {@code argparse.ArgumentTypeError}. ArgumentParser catches
 * this specifically and uses its message verbatim (without the
 * {@code "invalid X value"} wrapper that wraps ordinary {@link NumberFormatException}).
 */
public class ArgumentTypeError extends RuntimeException {

    public ArgumentTypeError() {
        super();
    }

    public ArgumentTypeError(String message) {
        super(message);
    }

    public ArgumentTypeError(String message, Throwable cause) {
        super(message, cause);
    }
}