/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import org.bytedeco.pytorch.llm.generation.StoppingCriteria;
import org.bytedeco.pytorch.llm.generation.TextIteratorStreamer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;

/**
 * Programs the user-facing chat UI. Mirrors the chat interface used by Ex22 (Gradio).
 *
 * <p>The actual HTTP/GUI plumbing is delegated to the {@link #renderer}; users who want
 * the Gradio look-and-feel feed in a {@code GradioRenderer} that connects to a real
 * Gradio server, while tests use a {@code InMemoryRenderer}.
 *
 * <p>Phase 4 of WEB_DEMO_IMPLEMENTATION_PLAN.md: added {@link StreamingRenderer} which
 * receives a chunk per generated token (vs {@link Renderer#onMessage} which only fires
 * once with the complete text). The Web Demo's SSE pipeline wires its SSE emitter into
 * this callback.
 */
public final class ChatbotDemo {

    public interface Renderer {
        void onMessage(String userMessage, String history);
        void onError(Throwable t);
    }

    /**
     * Streaming variant of {@link Renderer} — fires once per text chunk produced by the
     * model. Implementations should be tolerant of {@code null} / empty chunks.
     */
    public interface StreamingRenderer extends Renderer {
        void onChunk(String chunk);
        @Override default void onMessage(String userMessage, String history) {
            // Default no-op; streaming renderers compose end-of-turn separately.
        }
    }

    public static final class InMemoryRenderer implements Renderer {
        public final List<String> log = new ArrayList<>();
        @Override public void onMessage(String userMessage, String history) {
            log.add("user: " + userMessage);
            log.add("bot: " + history);
        }
        @Override public void onError(Throwable t) { log.add("error: " + t.getMessage()); }
    }

    /**
     * In-memory streaming renderer — captures every chunk + the final assembled text,
     * useful in unit tests.
     */
    public static final class InMemoryStreamingRenderer implements StreamingRenderer {
        public final List<String> chunks = new ArrayList<>();
        public final List<String> errors = new ArrayList<>();
        @Override public void onChunk(String chunk) {
            if (chunk != null) chunks.add(chunk);
        }
        @Override public void onMessage(String userMessage, String history) {
            // no-op
        }
        @Override public void onError(Throwable t) {
            errors.add(t.getMessage());
        }
        public String assembled() {
            StringBuilder sb = new StringBuilder();
            for (String c : chunks) sb.append(c);
            return sb.toString();
        }
    }

    private final BiFunction<String, List<String>, String> generate;
    private final StoppingCriteria criteria;
    private final TextIteratorStreamer streamer;
    private final Renderer renderer;

    public ChatbotDemo(BiFunction<String, List<String>, String> generate,
                       StoppingCriteria criteria, TextIteratorStreamer streamer,
                       Renderer renderer) {
        this.generate = generate;
        this.criteria = criteria;
        this.streamer = streamer;
        this.renderer = renderer;
    }

    public String reply(String userMessage, List<String> history) {
        try {
            String reply = generate.apply(userMessage + (history.isEmpty() ? "" : "\n" + String.join("\n", history)),
                                          history);
            if (streamer != null) streamer.putText(reply);
            if (renderer != null) renderer.onMessage(userMessage, reply);
            return reply;
        } catch (Throwable t) {
            if (renderer != null) renderer.onError(t);
            throw t;
        }
    }

    /**
     * Run a chat turn and, if the renderer is a {@link StreamingRenderer}, push every chunk
     * to it. Otherwise falls back to the synchronous renderer.
     *
     * @return the assembled reply string.
     */
    public String replyStreaming(String userMessage, List<String> history) {
        try {
            String reply = generate.apply(userMessage + (history.isEmpty() ? "" : "\n" + String.join("\n", history)),
                                          history);
            if (streamer != null) {
                if (renderer instanceof StreamingRenderer) {
                    ((StreamingRenderer) renderer).onChunk(reply);
                } else {
                    streamer.putText(reply);
                }
            } else if (renderer instanceof StreamingRenderer) {
                ((StreamingRenderer) renderer).onChunk(reply);
            }
            if (renderer != null) renderer.onMessage(userMessage, reply);
            return reply;
        } catch (Throwable t) {
            if (renderer != null) renderer.onError(t);
            throw t;
        }
    }
}