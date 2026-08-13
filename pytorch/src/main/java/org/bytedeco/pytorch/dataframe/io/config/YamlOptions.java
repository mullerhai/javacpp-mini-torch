package org.bytedeco.pytorch.dataframe.io.config;

/**
 * Options for YAML writing.
 */
public final class YamlOptions {

    public enum Style { BLOCK, FLOW }
    public enum Orient { ROWS, COLUMNS, INDEX, RECORDS, SPLIT }

    private Style style = Style.BLOCK;
    private Orient orient = Orient.ROWS;
    private boolean includeHeader = true;
    private boolean includeMetadata = false;
    private boolean flowArrays = false;
    private int indent = 2;

    public static YamlOptions defaults() { return new YamlOptions(); }

    public Style style() { return style; }
    public YamlOptions style(Style s) { this.style = s; return this; }

    public Orient orient() { return orient; }
    public YamlOptions orient(Orient o) { this.orient = o; return this; }

    public boolean includeHeader() { return includeHeader; }
    public YamlOptions includeHeader(boolean b) { this.includeHeader = b; return this; }

    public boolean includeMetadata() { return includeMetadata; }
    public YamlOptions includeMetadata(boolean b) { this.includeMetadata = b; return this; }

    public boolean flowArrays() { return flowArrays; }
    public YamlOptions flowArrays(boolean b) { this.flowArrays = b; return this; }

    public int indent() { return indent; }
    public YamlOptions indent(int n) { this.indent = n; return this; }
}
