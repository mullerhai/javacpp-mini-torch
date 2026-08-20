/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse;

/**
 * Sentinel exception used by ArgumentParser's {@code exit()} method to mimic
 * Python's {@code SystemExit} without actually terminating the JVM.
 * Tests can catch this to verify that {@code error()} / {@code print_help()} behaved correctly.
 */
public class ExitTrappedException extends RuntimeException {

    private final int status;

    public ExitTrappedException(int status, String message) {
        super(message);
        this.status = status;
    }

    public int getStatus() {
        return status;
    }
}