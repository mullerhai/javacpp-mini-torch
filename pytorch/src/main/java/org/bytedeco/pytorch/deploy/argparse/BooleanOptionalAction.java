/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse;

import java.util.ArrayList;
import java.util.List;

/**
 * Mirrors Python {@code argparse.BooleanOptionalAction}.
 *
 * <p>Given {@code addArgument("--foo")}, registers both {@code --foo} and
 * {@code --no-foo}. Calling with {@code --foo} sets dest=true,
 * with {@code --no-foo} sets dest=false.
 */
public class BooleanOptionalAction extends Action {

    public BooleanOptionalAction() {}

    public BooleanOptionalAction(List<String> optionStrings, String dest,
                                 Object defaultValue, boolean required, String help,
                                 boolean deprecated) {
        super(expandAndCheck(optionStrings), dest, 0, null, defaultValue, null, null,
                required, help, null, deprecated);
    }

    private static List<String> expandAndCheck(List<String> optionStrings) {
        List<String> expanded = new ArrayList<>();
        for (String optionString : optionStrings) {
            expanded.add(optionString);
            if (optionString.startsWith("--")) {
                if (optionString.startsWith("--no-")) {
                    throw new IllegalArgumentException(
                            "invalid option name '" + optionString
                                    + "' for BooleanOptionalAction");
                }
                expanded.add("--no-" + optionString.substring(2));
            }
        }
        return expanded;
    }

    @Override
    public void call(ArgumentParser parser, Namespace namespace,
                     Object values, String optionString) {
        namespace.set(getDest(), !optionString.startsWith("--no-"));
    }

    @Override
    public String formatUsage() {
        return String.join(" | ", getOptionStrings());
    }

    public static BooleanOptionalAction create(List<String> optionStrings, String dest,
                                               Object defaultValue, boolean required,
                                               String help, boolean deprecated) {
        return new BooleanOptionalAction(optionStrings, dest, defaultValue, required,
                help, deprecated);
    }
}