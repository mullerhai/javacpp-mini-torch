/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.formatter;

/**
 * Help message formatter which retains any formatting in descriptions.
 * Mirrors Python's {@code RawDescriptionHelpFormatter}.
 */
public class RawDescriptionHelpFormatter extends HelpFormatter {

    public RawDescriptionHelpFormatter(String prog) {
        super(prog);
    }

    public RawDescriptionHelpFormatter(String prog, int indentIncrement, int maxHelpPosition, int width) {
        super(prog, indentIncrement, maxHelpPosition, width);
    }

    @Override
    public String _fillText(String text, int width, String indent) {
        if (text == null || text.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        for (String line : text.split("\\r?\\n", -1)) {
            sb.append(indent).append(line).append('\n');
        }
        // drop the trailing newline that the Python version does NOT add
        if (sb.length() > 0) sb.setLength(sb.length() - 1);
        return sb.toString();
    }
}