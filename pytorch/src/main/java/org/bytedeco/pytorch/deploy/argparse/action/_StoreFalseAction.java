/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.action;

import java.util.List;

/** Mirrors Python {@code _StoreFalseAction}. */
public class _StoreFalseAction extends _StoreConstAction {

    public _StoreFalseAction() {}

    public _StoreFalseAction(List<String> optionStrings, String dest, Object defaultValue,
                             boolean required, String help, boolean deprecated) {
        super(optionStrings, dest, Boolean.FALSE,
                defaultValue == null ? Boolean.TRUE : defaultValue,
                required, help, null, deprecated);
    }

    public static _StoreFalseAction create(List<String> optionStrings, String dest,
                                           Object defaultValue, boolean required,
                                           String help, boolean deprecated) {
        return new _StoreFalseAction(optionStrings, dest, defaultValue, required, help, deprecated);
    }
}