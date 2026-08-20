/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.formatter;

import org.bytedeco.pytorch.deploy.argparse.Action;

/**
 * Mirrors Python's {@code MetavarTypeHelpFormatter}: use {@code type.__name__}
 * as the default metavar (instead of {@code dest.upper()} / {@code dest}).
 */
public class MetavarTypeHelpFormatter extends HelpFormatter {

    public MetavarTypeHelpFormatter(String prog) {
        super(prog);
    }

    public MetavarTypeHelpFormatter(String prog, int indentIncrement, int maxHelpPosition, int width) {
        super(prog, indentIncrement, maxHelpPosition, width);
    }

    @Override
    public Object _getDefaultMetavarForOptional(Action action) {
        Object t = action.getType();
        if (t instanceof Class<?> cls) return cls.getSimpleName();
        if (t != null) return t.toString();
        return "";
    }

    @Override
    public Object _getDefaultMetavarForPositional(Action action) {
        Object t = action.getType();
        if (t instanceof Class<?> cls) return cls.getSimpleName();
        if (t != null) return t.toString();
        return "";
    }
}