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

/** Mirrors Python {@code _CountAction}. */
public class _CountAction extends Action {

    public _CountAction() {}

    public _CountAction(List<String> optionStrings,
                        String dest,
                        Object defaultValue,
                        boolean required,
                        String help,
                        boolean deprecated) {
        super(optionStrings, dest, 0, null, defaultValue, null, null,
                required, help, null, deprecated);
    }

    @Override
    public void call(ArgumentParser parser, Namespace namespace,
                     Object values, String optionString) {
        Integer count = (Integer) namespace.get(getDest());
        if (count == null) count = 0;
        namespace.set(getDest(), count + 1);
    }

    public static _CountAction create(List<String> optionStrings, String dest,
                                      Object defaultValue, boolean required, String help,
                                      boolean deprecated) {
        return new _CountAction(optionStrings, dest, defaultValue, required, help, deprecated);
    }
}