/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.action;

import org.bytedeco.pytorch.deploy.argparse.Action;
import org.bytedeco.pytorch.deploy.argparse.ArgumentParser;
import org.bytedeco.pytorch.deploy.argparse.Namespace;

import java.util.List;

/** Mirrors Python {@code _StoreConstAction}. */
public class _StoreConstAction extends Action {

    public _StoreConstAction() {}

    public _StoreConstAction(List<String> optionStrings,
                             String dest,
                             Object constValue,
                             Object defaultValue,
                             boolean required,
                             String help,
                             Object metavar,
                             boolean deprecated) {
        super(optionStrings, dest, 0, constValue, defaultValue, null, null,
                required, help, metavar, deprecated);
    }

    @Override
    public void call(ArgumentParser parser, Namespace namespace,
                     Object values, String optionString) {
        namespace.set(getDest(), getConst());
    }

    public static _StoreConstAction create(List<String> optionStrings, String dest,
                                           Object constValue, Object defaultValue,
                                           boolean required, String help, Object metavar,
                                           boolean deprecated) {
        return new _StoreConstAction(optionStrings, dest, constValue, defaultValue,
                required, help, metavar, deprecated);
    }
}