package org.bytedeco.pytorch.dataframe.io.lmdb;

import java.util.ArrayList;
import java.util.List;

/**
 * Shared options for LMDB reader and writer.
 */
public class LmdbOptions {
    private int start = 0;
    private int limit = -1; // -1 means unlimited
    private int sampleSize = 1000;
    private boolean isImageDatabase = false;
    private boolean isTensorDatabase = false;
    private boolean keysOnly = false;
    private int pageSize = 4096;
    private String keyColumn = "key";
    private List<String> columns = new ArrayList<>();

    public static LmdbOptions defaults() {
        return new LmdbOptions();
    }

    public LmdbOptions start(int s) { this.start = s; return this; }
    public LmdbOptions limit(int l) { this.limit = l; return this; }
    public LmdbOptions sampleSize(int s) { this.sampleSize = s; return this; }
    public LmdbOptions isImageDatabase(boolean b) { this.isImageDatabase = b; return this; }
    public LmdbOptions isTensorDatabase(boolean b) { this.isTensorDatabase = b; return this; }
    public LmdbOptions keysOnly(boolean b) { this.keysOnly = b; return this; }
    public LmdbOptions pageSize(int s) { this.pageSize = s; return this; }
    public LmdbOptions keyColumn(String c) { this.keyColumn = c; return this; }
    public LmdbOptions columns(List<String> cols) { this.columns = cols; return this; }

    public int start() { return start; }
    public int limit() { return limit; }
    public int sampleSize() { return sampleSize; }
    public boolean isImageDatabase() { return isImageDatabase; }
    public boolean isTensorDatabase() { return isTensorDatabase; }
    public boolean keysOnly() { return keysOnly; }
    public int pageSize() { return pageSize; }
    public String keyColumn() { return keyColumn; }
    public List<String> columns() { return columns; }
}