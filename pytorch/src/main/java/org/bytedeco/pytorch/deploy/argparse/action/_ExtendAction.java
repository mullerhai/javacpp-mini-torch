/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.action;

import org.bytedeco.pytorch.deploy.argparse.ArgparseConstants;
import org.bytedeco.pytorch.deploy.argparse.ArgumentError;
import org.bytedeco.pytorch.deploy.argparse.ArgumentParser;
import org.bytedeco.pytorch.deploy.argparse.Namespace;
import org.bytedeco.pytorch.deploy.argparse.internal.CopyItems;

import java.util.List;

/** Mirrors Python {@code _ExtendAction}. */
public class _ExtendAction extends _AppendAction {

    public _ExtendAction() {}

    public _ExtendAction(List<String> optionStrings,
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
            items.addAll((List<?>) values);
        } else {
            items.add(values);
        }
        namespace.set(getDest(), items);
    }

    public static _ExtendAction create(List<String> optionStrings, String dest, Object nargs,
                                       Object constValue, Object defaultValue, Object type,
                                       Object choices, boolean required, String help,
                                       Object metavar, boolean deprecated) {
        if (nargs instanceof Integer && ((Integer) nargs) == 0) {
            throw new ArgumentError("nargs for extend actions must be != 0");
        }
        if (constValue != null && !ArgparseConstants.OPTIONAL.equals(nargs)) {
            throw new ArgumentError("nargs must be '" + ArgparseConstants.OPTIONAL
                    + "' to supply const");
        }
        return new _ExtendAction(optionStrings, dest, nargs, constValue, defaultValue,
                type, choices, required, help, metavar, deprecated);
    }
}