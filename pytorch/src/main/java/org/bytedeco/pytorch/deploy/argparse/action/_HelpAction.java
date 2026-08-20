/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.action;

import org.bytedeco.pytorch.deploy.argparse.Action;
import org.bytedeco.pytorch.deploy.argparse.ArgparseConstants;
import org.bytedeco.pytorch.deploy.argparse.ArgumentParser;
import org.bytedeco.pytorch.deploy.argparse.Namespace;

import java.util.List;

/** Mirrors Python {@code _HelpAction}: prints help and exits. */
public class _HelpAction extends Action {

    public _HelpAction() {}

    public _HelpAction(List<String> optionStrings) {
        super(optionStrings, ArgparseConstants.SUPPRESS,
                0, null, ArgparseConstants.SUPPRESS, null, null,
                false, "show this help message and exit", null, false);
    }

    @Override
    public void call(ArgumentParser parser, Namespace namespace,
                     Object values, String optionString) {
        parser.printHelp();
        parser.exit(0);
    }

    public static _HelpAction create(List<String> optionStrings, String dest, Object defaultValue,
                                     String help, boolean deprecated) {
        return new _HelpAction(optionStrings);
    }
}