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

    public TextIteratorStreamer() { this(null); }
    public TextIteratorStreamer(Consumer<String> onChunk) { this.onChunk = onChunk; }

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
        notifyAll();
    }

    public synchronized boolean isEnded() {
        return ended;
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