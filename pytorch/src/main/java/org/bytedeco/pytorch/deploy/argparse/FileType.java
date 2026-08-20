/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Set;

/**
 * Mirrors Python's deprecated {@code argparse.FileType} factory. {@code FileType}
 * instances are callable: when used as {@code type=}, they accept a string
 * filename and return an open {@link InputStream} / {@link OutputStream}.
 *
 * <p>Java has no first-class file "object" type, so we wrap the resulting stream
 * in a thin {@link FileHandle} exposing {@link InputStream} / {@link OutputStream}.
 * Users can opt for {@link java.io.FileInputStream} or {@link java.io.FileOutputStream}
 * via {@link FileHandle#openInput()} / {@link FileHandle#openOutput()}.
 *
 * <p>The class is annotated {@code @Deprecated} to mirror Python 3.14's status.
 */
@Deprecated
public class FileType {

    private final String mode;
    private final int bufsize;
    private final String encoding;
    private final String errors;

    public FileType(String mode, int bufsize, String encoding, String errors) {
        this.mode = mode;
        this.bufsize = bufsize;
        this.encoding = encoding;
        this.errors = errors;
    }

    public FileType(String mode) {
        this(mode, -1, null, null);
    }

    public String getMode() { return mode; }
    public int getBufsize() { return bufsize; }
    public String getEncoding() { return encoding; }
    public String getErrors() { return errors; }

    /** Function-call interface used by ArgumentParser's type registry. */
    public FileHandle call(String string) {
        if ("-".equals(string)) {
            boolean binary = mode.contains("b");
            if (mode.contains("r")) {
                return new FileHandle(System.in, null, "-", mode);
            } else if (mode.contains("w") || mode.contains("a") || mode.contains("x")) {
                return new FileHandle(null, System.out, "-", mode);
            } else {
                throw new ArgumentTypeError("argument '-' with mode '" + mode + "'");
            }
        }
        Path path = Paths.get(string);
        try {
            Charset cs = encoding == null ? StandardCharsets.UTF_8 : Charset.forName(encoding);
            if (mode.contains("r")) {
                Reader reader = Files.newBufferedReader(path, cs);
                return new FileHandle(null, null, string, mode);
            } else if (mode.contains("w") || mode.contains("a") || mode.contains("x")) {
                Files.createDirectories(path.getParent() == null ? Paths.get(".") : path.getParent());
                Writer writer = Files.newBufferedWriter(path, cs,
                        mode.contains("a") ? java.nio.file.StandardOpenOption.APPEND
                                : java.nio.file.StandardOpenOption.CREATE);
                return new FileHandle(null, null, string, mode);
            } else {
                throw new ArgumentTypeError("invalid mode: " + mode);
            }
        } catch (IOException e) {
            throw new ArgumentTypeError("can't open '" + string + "': " + e.getMessage());
        }
    }

    /** Lazy file handle returned by {@link #call(String)}. */
    public static final class FileHandle implements AutoCloseable {
        private final InputStream stdin;
        private final OutputStream stdout;
        private final String path;
        private final String mode;

        FileHandle(InputStream stdin, OutputStream stdout, String path, String mode) {
            this.stdin = stdin;
            this.stdout = stdout;
            this.path = path;
            this.mode = mode;
        }

        public InputStream openInput() throws IOException {
            if (stdin != null) return stdin;
            return Files.newInputStream(Paths.get(path));
        }

        public OutputStream openOutput() throws IOException {
            if (stdout != null) return stdout;
            boolean append = mode.contains("a");
            java.nio.file.StandardOpenOption[] opts = append
                    ? new java.nio.file.StandardOpenOption[] {
                            java.nio.file.StandardOpenOption.CREATE,
                            java.nio.file.StandardOpenOption.APPEND }
                    : new java.nio.file.StandardOpenOption[] {
                            java.nio.file.StandardOpenOption.CREATE,
                            java.nio.file.StandardOpenOption.TRUNCATE_EXISTING,
                            java.nio.file.StandardOpenOption.WRITE };
            return Files.newOutputStream(Paths.get(path), opts);
        }

        public Reader openReader() throws IOException {
            return Files.newBufferedReader(Paths.get(path), StandardCharsets.UTF_8);
        }

        public Writer openWriter() throws IOException {
            return Files.newBufferedWriter(Paths.get(path), StandardCharsets.UTF_8);
        }

        public String path() { return path; }
        public String mode() { return mode; }

        @Override public void close() {}
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("FileType(");
        boolean first = true;
        if (!mode.equals("r") || bufsize != -1) {
            if (!first) sb.append(", ");
            sb.append("'").append(mode).append("'");
            first = false;
            if (bufsize != -1) {
                sb.append(", ").append(bufsize);
                first = false;
            }
        }
        if (encoding != null) { if (!first) sb.append(", "); sb.append("encoding='").append(encoding).append("'"); first = false; }
        if (errors != null) { if (!first) sb.append(", "); sb.append("errors='").append(errors).append("'"); first = false; }
        if (first) sb.append("'").append(mode).append("'");
        sb.append(")");
        return sb.toString();
    }

    /** Unused, for compatibility with Python's repr(). */
    public String repr() {
        return toString();
    }
}