/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.action;

import org.bytedeco.pytorch.deploy.argparse.Action;
import org.bytedeco.pytorch.deploy.argparse.ArgumentParser;
import org.bytedeco.pytorch.deploy.argparse.Namespace;
import org.bytedeco.pytorch.deploy.argparse.internal.CopyItems;

import java.util.List;

/** Mirrors Python {@code _AppendConstAction}. */
public class _AppendConstAction extends Action {

    public _AppendConstAction() {}

    public _AppendConstAction(List<String> optionStrings,
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
    @SuppressWarnings("unchecked")
    public void call(ArgumentParser parser, Namespace namespace,
                     Object values, String optionString) {
        List<Object> items = (List<Object>) namespace.get(getDest());
        if (items == null) {
            items = new java.util.ArrayList<>();
        } else {
            items = CopyItems.copy(items);
        }
        items.add(getConst());
        namespace.set(getDest(), items);
    }

    public static _AppendConstAction create(List<String> optionStrings, String dest,
                                            Object constValue, Object defaultValue,
                                            boolean required, String help, Object metavar,
                                            boolean deprecated) {
        return new _AppendConstAction(optionStrings, dest, constValue, defaultValue,
                required, help, metavar, deprecated);
    }
}