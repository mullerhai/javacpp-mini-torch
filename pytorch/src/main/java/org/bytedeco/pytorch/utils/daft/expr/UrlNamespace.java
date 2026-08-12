/*
 * URL namespace — host extraction + bulk download.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/**
 * URL namespace.
 *
 * <p>{@code col("url").url.host()} → string column of hosts.
 * {@code col("url").url.download()} → binary column with the bytes.
 * {@code col("url").url.download(timeoutMs)} → custom timeout.
 */
public final class UrlNamespace {

    private final Expression inner;
    public UrlNamespace(Expression inner) { this.inner = Objects.requireNonNull(inner); }

    public Expression host() {
        return new Expression() {
            @Override public String name() { return inner.name() + ".host"; }
            @Override public Column eval(DataFrame df) {
                Column c = inner.eval(df);
                if (c == null) return null;
                Column out = new Column(name(), Column.DType.STRING);
                for (int i = 0; i < c.size(); i++) {
                    Object v = c.get(i);
                    if (v == null) { out.add(null); continue; }
                    try {
                        URI u = URI.create(v.toString());
                        out.add(u.getHost());
                    } catch (RuntimeException e) { out.add(null); }
                }
                return out;
            }
        };
    }

    public Expression path() {
        return new Expression() {
            @Override public String name() { return inner.name() + ".path"; }
            @Override public Column eval(DataFrame df) {
                Column c = inner.eval(df);
                if (c == null) return null;
                Column out = new Column(name(), Column.DType.STRING);
                for (int i = 0; i < c.size(); i++) {
                    Object v = c.get(i);
                    if (v == null) { out.add(null); continue; }
                    try {
                        out.add(URI.create(v.toString()).getPath());
                    } catch (RuntimeException e) { out.add(null); }
                }
                return out;
            }
        };
    }

    public Expression scheme() {
        return new Expression() {
            @Override public String name() { return inner.name() + ".scheme"; }
            @Override public Column eval(DataFrame df) {
                Column c = inner.eval(df);
                if (c == null) return null;
                Column out = new Column(name(), Column.DType.STRING);
                for (int i = 0; i < c.size(); i++) {
                    Object v = c.get(i);
                    if (v == null) { out.add(null); continue; }
                    try {
                        out.add(URI.create(v.toString()).getScheme());
                    } catch (RuntimeException e) { out.add(null); }
                }
                return out;
            }
        };
    }

    /** Synchronously download each URL's content. */
    public Expression download() {
        return download(30_000);
    }

    public Expression download(long timeoutMs) {
        return new Expression() {
            @Override public String name() { return inner.name() + "_bytes"; }
            @Override public Column eval(DataFrame df) {
                Column c = inner.eval(df);
                if (c == null) return null;
                Column out = new Column(name(), Column.DType.BINARY);
                for (int i = 0; i < c.size(); i++) {
                    Object v = c.get(i);
                    if (v == null) { out.add(null); continue; }
                    try {
                        out.add(downloadBytes(v.toString(), timeoutMs));
                    } catch (Exception e) { out.add(null); }
                }
                return out;
            }
        };
    }

    /** Async parallel download using the session's IO pool. */
    public Expression downloadParallel(ExecutorService pool, long timeoutMs) {
        return new Expression() {
            @Override public String name() { return inner.name() + "_bytes"; }
            @Override public Column eval(DataFrame df) {
                Column c = inner.eval(df);
                if (c == null) return null;
                int n = c.size();
                byte[][] results = new byte[n][];
                Throwable[] errors = new Throwable[n];
                java.util.concurrent.CountDownLatch latch = new java.util.concurrent.CountDownLatch(n);
                for (int i = 0; i < n; i++) {
                    final int idx = i;
                    final String url = c.get(i) == null ? null : c.get(i).toString();
                    pool.submit(() -> {
                        try {
                            if (url == null) {
                                results[idx] = null;
                            } else {
                                results[idx] = downloadBytes(url, timeoutMs);
                            }
                        } catch (Throwable t) {
                            errors[idx] = t;
                        } finally {
                            latch.countDown();
                        }
                    });
                }
                try {
                    latch.await(timeoutMs * (long) n + 30_000, TimeUnit.MILLISECONDS);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }
                Column out = new Column(name(), Column.DType.BINARY);
                for (int i = 0; i < n; i++) {
                    out.add(results[i]);
                }
                return out;
            }
        };
    }

    private static byte[] downloadBytes(String urlStr, long timeoutMs) throws Exception {
        URL u = URI.create(urlStr).toURL();
        HttpURLConnection conn = (HttpURLConnection) u.openConnection();
        conn.setConnectTimeout((int) Math.min(timeoutMs, 30_000));
        conn.setReadTimeout((int) Math.min(timeoutMs, 30_000));
        conn.setRequestProperty("User-Agent", "Daft-Java/1.0");
        try (BufferedInputStream in = new BufferedInputStream(conn.getInputStream());
             ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buf = new byte[16 * 1024];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
            return out.toByteArray();
        } finally {
            conn.disconnect();
        }
    }
}