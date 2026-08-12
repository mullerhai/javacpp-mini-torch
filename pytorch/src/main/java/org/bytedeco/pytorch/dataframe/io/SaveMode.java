package org.bytedeco.pytorch.dataframe.io;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

/**
 * Spark-compatible {@code SaveMode} for {@link DataFrameWriter}.
 *
 * <p>Semantics (when the target path already exists):
 * <ul>
 *   <li>{@link #OVERWRITE} (default in Spark) — replace the existing data.</li>
 *   <li>{@link #OVERWRITE_APPEND} — overwrite if exists, otherwise append.</li>
 *   <li>{@link #APPEND} — append to existing data when the format supports it.</li>
 *   <li>{@link #IGNORE} — silently no-op if data already exists.</li>
 *   <li>{@link #ERRORIFEXISTS} — throw {@link java.io.IOException} if data exists.</li>
 * </ul>
 *
 * <p>Modes are matched case-insensitively from {@link #fromString(String)}. Spark users
 * typically pass {@code "append"}, {@code "overwrite"}, {@code "ignore"}, or
 * {@code "error"} / {@code "errorifexists"}.
 */
public enum SaveMode {
    /** Replace any existing data at the destination. Default behaviour. */
    OVERWRITE,
    /** Replace existing data when present, otherwise behave like append. */
    OVERWRITE_APPEND,
    /** Append new rows to the existing data when the format supports it. */
    APPEND,
    /** Silently do nothing if data already exists at the destination. */
    IGNORE,
    /** Throw {@link java.io.IOException} if data already exists. */
    ERRORIFEXISTS;

    /**
     * Case-insensitive Spark-style parsing.
     *
     * <p>Accepted inputs: {@code overwrite}, {@code append}, {@code ignore},
     * {@code error}, {@code errorifexists}, {@code error_if_exists},
     * {@code overwriteappend}, {@code overwrite_append}, plus the enum names
     * {@code OVERWRITE_APPEND}, etc.
     */
    public static SaveMode fromString(String s) {
        if (s == null || s.isEmpty()) return OVERWRITE;
        String norm = s.trim().toLowerCase(Locale.ROOT).replace("-", "").replace("_", "");
        switch (norm) {
            case "overwrite": return OVERWRITE;
            case "overwriteappend": return OVERWRITE_APPEND;
            case "append": return APPEND;
            case "ignore": return IGNORE;
            case "error":
            case "errorifexists":
                return ERRORIFEXISTS;
            default:
                throw new IllegalArgumentException("Unknown SaveMode: " + s);
        }
    }

    /**
     * Apply the pre-write existence check for the target path.
     *
     * @return {@code true} when the caller should proceed with the write,
     *         {@code false} when the mode says to silently no-op.
     * @throws java.io.IOException for {@link #ERRORIFEXISTS} when path exists.
     */
    public boolean preCheck(String path) throws java.io.IOException {
        if (path == null) return true;
        Path p;
        try {
            p = Paths.get(path);
        } catch (Exception e) {
            return true;
        }
        boolean exists = Files.exists(p);
        switch (this) {
            case IGNORE:
                return !exists;
            case ERRORIFEXISTS:
                if (exists) {
                    throw new java.io.IOException(
                        "SaveMode.ERRORIFEXISTS: destination already exists: " + path);
                }
                return true;
            case OVERWRITE:
            case OVERWRITE_APPEND:
            case APPEND:
            default:
                return true;
        }
    }
}
