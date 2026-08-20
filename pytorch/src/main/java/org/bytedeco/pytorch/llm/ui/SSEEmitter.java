/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Minimal Server-Sent Events (SSE) writer. Per the W3C HTML5 spec an event looks like:
 * <pre>
 *   event:&lt;name&gt;
 *   data:&lt;data&gt;
 *   id:&lt;optional id&gt;
 *
 *   (blank line terminates the event)
 * </pre>
 *
 * <p>This class wraps an {@link OutputStream} (typically {@code HttpExchange.getResponseBody()})
 * with a {@link ReentrantLock} so concurrent callers (the inference thread + the SSE flush
 * thread) don't interleave bytes mid-event.
 *
 * <p>Always call {@link #close()} when done — it emits the final newline + flushes.
 */
public final class SSEEmitter {

    private final OutputStream out;
    private final ReentrantLock lock = new ReentrantLock();
    private boolean closed = false;

    public SSEEmitter(OutputStream out) {
        this.out = out;
    }

    public SSEEmitter emitEvent(String event, String data) throws IOException {
        return emitEvent(event, data, null);
    }

    public SSEEmitter emitEvent(String event, String data, String id) throws IOException {
        lock.lock();
        try {
            if (closed) throw new IOException("SSEEmitter is closed");
            StringBuilder sb = new StringBuilder(64 + (data == null ? 0 : data.length()));
            if (event != null && !event.isEmpty()) {
                sb.append("event:").append(event).append('\n');
            }
            if (data != null) {
                // Per spec: data may span multiple lines, each prefixed with "data:".
                String[] lines = data.split("\\r?\\n", -1);
                for (String line : lines) {
                    sb.append("data:").append(line).append('\n');
                }
            }
            if (id != null && !id.isEmpty()) {
                sb.append("id:").append(id).append('\n');
            }
            sb.append('\n'); // blank line terminates the event
            out.write(sb.toString().getBytes(StandardCharsets.UTF_8));
            out.flush();
            return this;
        } finally {
            lock.unlock();
        }
    }

    public SSEEmitter emitComment(String comment) throws IOException {
        lock.lock();
        try {
            if (closed) return this;
            out.write((": " + (comment == null ? "" : comment) + "\n\n").getBytes(StandardCharsets.UTF_8));
            out.flush();
            return this;
        } finally {
            lock.unlock();
        }
    }

    public void flush() throws IOException {
        lock.lock();
        try {
            out.flush();
        } finally {
            lock.unlock();
        }
    }

    public void close() throws IOException {
        lock.lock();
        try {
            if (closed) return;
            closed = true;
            // best-effort final flush — we don't close the underlying stream; the SSE handler does.
            try { out.flush(); } catch (IOException ignored) {}
        } finally {
            lock.unlock();
        }
    }

    public boolean isClosed() { return closed; }
}