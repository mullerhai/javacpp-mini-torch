/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.generation;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Hugging Face {@code TextIteratorStreamer} analog. Other threads (TF, DedicatedThread) push
 * tokens into {@link #put(int)} as the model produces them; the calling thread polls
 * {@link #textSoFar()} to get the partial output.
 *
 * <p>Used by Ex22 (Gradio Chatbot) and Ex33 (Unsloth LLaMA3).
 */
public final class TextIteratorStreamer {

    private final StringBuilder sb = new StringBuilder();
    private final List<String> chunks = new CopyOnWriteArrayList<>();
    private final Consumer<String> onChunk;
    private boolean ended = false;
    private long startTimeMillis = 0L;
    private long endTimeMillis = 0L;

    public TextIteratorStreamer() { this(null); }
    public TextIteratorStreamer(Consumer<String> onChunk) {
        this.onChunk = onChunk;
        this.startTimeMillis = System.currentTimeMillis();
    }

    public synchronized void put(int tokenId) {
        // The decoder is owned by the model wrapper; we just append raw placeholder text.
        // The actual detokenization is done by callers (UnslothEx inference) feeding in
        // already-decoded pieces via {@link #putText(String)} when available.
        String s = String.valueOf(tokenId);
        sb.append(s);
        chunks.add(s);
        if (onChunk != null) onChunk.accept(s);
    }

    public synchronized void putText(String text) {
        if (text == null) return;
        sb.append(text);
        chunks.add(text);
        if (onChunk != null) onChunk.accept(text);
    }

    public synchronized String textSoFar() {
        return sb.toString();
    }

    public synchronized List<String> chunks() {
        return java.util.Collections.unmodifiableList(chunks);
    }

    public synchronized void end() {
        ended = true;
        endTimeMillis = System.currentTimeMillis();
        notifyAll();
    }

    public synchronized boolean isEnded() {
        return ended;
    }

    /**
     * Mark the streamer as ended with explicit wall-clock timestamp.
     * Useful for tests / deterministic metrics.
     */
    public synchronized void markEnd(long tsMillis) {
        ended = true;
        endTimeMillis = tsMillis;
        notifyAll();
    }

    public synchronized long startTimeMillis() {
        return startTimeMillis == 0L ? System.currentTimeMillis() : startTimeMillis;
    }

    public synchronized long endTimeMillis() {
        return endTimeMillis;
    }

    public synchronized long elapsedMillis() {
        long end = endTimeMillis == 0L ? System.currentTimeMillis() : endTimeMillis;
        long start = startTimeMillis == 0L ? end : startTimeMillis;
        return Math.max(0L, end - start);
    }

    /**
     * Reset all state so the same streamer can be reused for a new turn.
     * Callers must ensure no other thread is mid-{@code put}/{@code awaitEnd}.
     */
    public synchronized void reset() {
        sb.setLength(0);
        chunks.clear();
        ended = false;
        startTimeMillis = System.currentTimeMillis();
        endTimeMillis = 0L;
    }

    /**
     * Alias for {@link #end()} kept for consistency with the Web Demo plan.
     */
    public void markEnd() {
        end();
    }

    /** How many chunks have been produced in this turn (since the last reset). */
    public synchronized int addedChunks() {
        return chunks.size();
    }

    public synchronized String awaitEnd(long timeoutMs) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (!ended) {
            long remaining = deadline - System.currentTimeMillis();
            if (remaining <= 0) break;
            wait(remaining);
        }
        return sb.toString();
    }
}