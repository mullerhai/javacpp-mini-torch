/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.action;

import org.bytedeco.pytorch.deploy.argparse.*;
import org.bytedeco.pytorch.deploy.argparse.internal.CopyItems;

import java.util.List;

import static org.bytedeco.pytorch.deploy.argparse.ArgparseConstants.OPTIONAL;

/** Mirrors Python {@code _AppendAction}. */
public class _AppendAction extends Action {

    public _AppendAction() {}

    public _AppendAction(List<String> optionStrings,
                         String dest,
                         Object nargs,
                         Object constValue,
                         Object defaultValue,
                         Object type,
                         Object choices,
                         boolean required,
                         String help,
                         Object metavar,
                         boolean deprecated) {
        super(optionStrings, dest, nargs, constValue, defaultValue, type, choices,
                required, help, metavar, deprecated);
    }

    @Override
    @SuppressWarnings("unchecked")
    public void call(ArgumentParser parser, Namespace namespace,
                     Object values, String optionString) {
        List<Object> items = (List<Object>) namespace.get(getDest());
        if (items == null) {
            items = new java.util.ArrayList<>();
        } else {
            items = CopyItems.copy(items);
        }
        if (values instanceof List) {
            items.add(values);
        } else {
            items.add(values);
        }
        namespace.set(getDest(), items);
    }

    public static _AppendAction create(List<String> optionStrings, String dest, Object nargs,
                                       Object constValue, Object defaultValue, Object type,
                                       Object choices, boolean required, String help,
                                       Object metavar, boolean deprecated) {
        if (nargs instanceof Integer && ((Integer) nargs) == 0) {
            throw new ArgumentError("nargs for append actions must be != 0; if arg "
                    + "strings are not supplying the value to append, the append const "
                    + "action may be more appropriate");
        }
        if (constValue != null && !OPTIONAL.equals(nargs)) {
            throw new ArgumentError("nargs must be '" + ArgparseConstants.OPTIONAL
                    + "' to supply const");
        }
        return new _AppendAction(optionStrings, dest, nargs, constValue, defaultValue,
                type, choices, required, help, metavar, deprecated);
    }
}