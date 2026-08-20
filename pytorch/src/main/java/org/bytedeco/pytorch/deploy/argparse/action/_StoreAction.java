/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.action;

import org.bytedeco.pytorch.deploy.argparse.*;

import java.util.List;

/** Mirrors Python {@code _StoreAction}. */
public class _StoreAction extends Action {

    public _StoreAction() {}

    public _StoreAction(List<String> optionStrings,
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
        if (values instanceof List) {
            namespace.set(getDest(), values);
        } else {
            namespace.set(getDest(), values);
        }
    }

    /** Static factory for the registry pattern. */
    public static _StoreAction create(List<String> optionStrings, String dest, Object nargs,
                                      Object constValue, Object defaultValue, Object type,
                                      Object choices, boolean required, String help,
                                      Object metavar, boolean deprecated) {
        if (nargs instanceof Integer && ((Integer) nargs) == 0) {
            throw new ArgumentError("nargs for store actions must be != 0; if you "
                    + "have nothing to store, actions such as store true or store const "
                    + "may be more appropriate");
        }
        if (constValue != null && !ArgparseConstants.OPTIONAL.equals(nargs)) {
            throw new ArgumentError("nargs must be '" + ArgparseConstants.OPTIONAL
                    + "' to supply const");
        }
        return new _StoreAction(optionStrings, dest, nargs, constValue, defaultValue,
                type, choices, required, help, metavar, deprecated);
    }
}