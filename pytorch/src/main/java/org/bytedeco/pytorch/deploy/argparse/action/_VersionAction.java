/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.action;

import org.bytedeco.pytorch.deploy.argparse.Action;
import org.bytedeco.pytorch.deploy.argparse.ArgparseConstants;
import org.bytedeco.pytorch.deploy.argparse.ArgumentParser;
import org.bytedeco.pytorch.deploy.argparse.Namespace;
import org.bytedeco.pytorch.deploy.argparse.formatter.HelpFormatter;

import java.util.List;

/** Mirrors Python {@code _VersionAction}. */
public class _VersionAction extends Action {

    private String version;

    public _VersionAction() {}

    public _VersionAction(List<String> optionStrings, String version) {
        super(optionStrings, ArgparseConstants.SUPPRESS, 0, null,
                ArgparseConstants.SUPPRESS, null, null,
                false,
                version == null ? "show program's version number and exit"
                        : "show program's version number and exit",
                null, false);
        this.version = version;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String v) {
        this.version = v;
    }

    @Override
    public void call(ArgumentParser parser, Namespace namespace,
                     Object values, String optionString) {
        String v = version;
        if (v == null) {
            try {
                java.lang.reflect.Method gv = parser.getClass().getMethod("getVersion");
                Object got = gv.invoke(parser);
                v = got == null ? null : got.toString();
            } catch (ReflectiveOperationException ex) {
                v = null;
            }
        }
        HelpFormatter formatter = parser.getFormatter();
        formatter.addText(v);
        parser.printMessage(formatter.formatHelp());
        parser.exit(0);
    }

    public static _VersionAction create(List<String> optionStrings, String version,
                                        String dest, Object defaultValue, String help,
                                        boolean deprecated) {
        return new _VersionAction(optionStrings, version);
    }
}