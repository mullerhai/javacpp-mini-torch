/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import java.util.List;
import java.util.Map;

/**
 * Strategy interface that turns a {@link ChatSession} + the current {@link ChatTurn}
 * (plus all prior turns in {@code session.history}) into the literal string fed to the
 * tokenizer. The web demo doesn't need to know whether this is a ChatML format, a
 * Gemma-style template, or a raw concatenation — the caller decides at construction
 * time.
 *
 * <p>Implementing this interface lives in the {@code samples/llm/web} package; the
 * default reference implementation reads from a {@code PromptGenerator}.
 */
@FunctionalInterface
public interface PromptFormatter {
    String format(ChatSession session, ChatTurn turn, List<ChatTurn> history);

    /** Null-safe default: returns the raw concatenation "User: ...\nAssistant: ...". */
    static PromptFormatter simple() {
        return (session, turn, history) -> {
            StringBuilder sb = new StringBuilder();
            for (ChatTurn t : history) {
                sb.append("User: ").append(t.userMessage).append('\n');
                sb.append("Assistant: ").append(t.assistantText()).append('\n');
            }
            sb.append("User: ").append(turn.userMessage).append('\n');
            sb.append("Assistant: ");
            return sb.toString();
        };
    }

    /** Default no-op factory: useful for unit tests. */
    static PromptFormatter none() {
        return (session, turn, history) -> "";
    }

    /** Map-shaped factory helper: keys {@code "session","turn","history"}. */
    default Map<String, Object> toJson() {
        return Map.of("type", getClass().getSimpleName());
    }
}