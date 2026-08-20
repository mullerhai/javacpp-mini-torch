/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.formatter;

import org.bytedeco.pytorch.deploy.argparse.Action;
import org.bytedeco.pytorch.deploy.argparse.ArgparseConstants;

/**
 * Mirrors Python's {@code ArgumentDefaultsHelpFormatter}: appends
 * {@code (default: X)} to help strings when no explicit
 * {@code %(default)s} placeholder is present.
 */
public class ArgumentDefaultsHelpFormatter extends HelpFormatter {

    public ArgumentDefaultsHelpFormatter(String prog) {
        super(prog);
    }

    public ArgumentDefaultsHelpFormatter(String prog, int indentIncrement, int maxHelpPosition, int width) {
        super(prog, indentIncrement, maxHelpPosition, width);
    }

    @Override
    public String _getHelpString(Action action) {
        String help = action.getHelp();
        if (help == null) help = "";
        if (!help.contains("%(default)s")
                && !ArgparseConstants.SUPPRESS.equals(action.getDefault())
                && !action.isRequired()) {
            boolean defaultingNargs = ArgparseConstants.OPTIONAL.equals(action.getNargs())
                    || ArgparseConstants.ZERO_OR_MORE.equals(action.getNargs());
            if (action.getOptionStrings() != null && !action.getOptionStrings().isEmpty()
                    || defaultingNargs) {
                help += " (default: %(default)s)";
            }
        }
        return help;
    }
}