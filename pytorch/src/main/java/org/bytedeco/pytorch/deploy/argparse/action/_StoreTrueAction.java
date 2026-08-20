/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.action;

import java.util.List;

/** Mirrors Python {@code _StoreTrueAction}. */
public class _StoreTrueAction extends _StoreConstAction {

    public _StoreTrueAction() {}

    public _StoreTrueAction(List<String> optionStrings, String dest, Object defaultValue,
                            boolean required, String help, boolean deprecated) {
        super(optionStrings, dest, Boolean.TRUE,
                defaultValue == null ? Boolean.FALSE : defaultValue,
                required, help, null, deprecated);
    }

    public static _StoreTrueAction create(List<String> optionStrings, String dest,
                                          Object defaultValue, boolean required,
                                          String help, boolean deprecated) {
        return new _StoreTrueAction(optionStrings, dest, defaultValue, required, help, deprecated);
    }
}