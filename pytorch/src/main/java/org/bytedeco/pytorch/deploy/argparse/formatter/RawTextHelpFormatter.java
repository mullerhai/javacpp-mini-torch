/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.formatter;

import java.util.ArrayList;
import java.util.List;

/**
 * Mirrors Python's {@code RawTextHelpFormatter}: do not reformat description OR
 * help text for actions.
 */
public class RawTextHelpFormatter extends RawDescriptionHelpFormatter {

    public RawTextHelpFormatter(String prog) {
        super(prog);
    }

    public RawTextHelpFormatter(String prog, int indentIncrement, int maxHelpPosition, int width) {
        super(prog, indentIncrement, maxHelpPosition, width);
    }

    @Override
    public List<String> _splitLines(String text, int width) {
        if (text == null) return new ArrayList<>();
        List<String> out = new ArrayList<>();
        for (String line : text.split("\\r?\\n", -1)) out.add(line);
        return out;
    }
}