/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;


/**
 * Per-chunk callback invoked by the engine while a turn is being generated.
 * Implementations must be thread-safe — invoked from the engine's worker thread.
 */
@FunctionalInterface
public interface ChunkListener {
    /**
     * Called for every decoded text chunk produced by the model.
     *
     * @param session owning session (never {@code null})
     * @param turn    owning turn (never {@code null}); tokens have already been appended
     * @param chunk   the new text fragment (never {@code null}, possibly empty)
     */
    void onChunk(ChatSession session, ChatTurn turn, String chunk);

    /**
     * Called once when generation completes (success or failure). Default no-op.
     */
    default void onComplete(ChatSession session, ChatTurn turn) {}

    /**
     * Called when generation aborts with an exception. Default no-op.
     */
    default void onError(ChatSession session, ChatTurn turn, Throwable t) {}
}
