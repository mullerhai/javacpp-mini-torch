/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import org.bytedeco.pytorch.llm.generation.StoppingCriteria;
import org.bytedeco.pytorch.llm.generation.TextIteratorStreamer;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.transformers.generation.Generator;
import org.bytedeco.pytorch.nn.Module;

import java.util.List;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.function.IntConsumer;

/**
 * Thin streaming wrapper around the framework's {@link Generator#generate(Module, int[], GenerationConfig, int, IntConsumer)}
 * call. Decodes each generated token id back to text via the tokenizer and forwards the
 * decoded piece to the supplied listeners.
 *
 * <p>This class lives in {@code samples/llm/web} (not {@code llm/ui}) because it depends
 * on sample-specific wiring (a {@code Module} + {@code FastTokenizer} + a model-specific
 * prompt formatter). It is intentionally tiny — the heavy lifting is delegated to the
 * existing framework API.
 */
public final class InferenceBridge {

    private final Module model;
    private final FastTokenizer tokenizer;
    private final int maxContext;
    private final boolean stub;
    private final String stubPrefix;

    public InferenceBridge(Module model, FastTokenizer tokenizer, int maxContext) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.maxContext = maxContext <= 0 ? 4096 : maxContext;
        this.stub = model == null || tokenizer == null;
        this.stubPrefix = stub ? "[stub] " : "";
    }

    public boolean isStub() { return stub; }

    public FastTokenizer tokenizer() { return tokenizer; }
    public Module model() { return model; }
    public int maxContext() { return maxContext; }

    /**
     * Tokenize {@code prompt}, run generation, push each decoded token chunk to the
     * supplied listeners, then end the streamer.
     *
     * @param prompt     already prompt-formatted string
     * @param gen        generation config (temperature, top_k, etc.)
     * @param streamer   target streamer (its internal buffer mirrors the onToken text)
     * @param criteria   stopping criteria consulted after each token
     * @param onText     per-chunk text callback (may be {@code null})
     * @param onToken    per-token callback (receives the raw token id, may be {@code null})
     */
    public void predictStreaming(String prompt, GenerationConfig gen,
                                 TextIteratorStreamer streamer,
                                 StoppingCriteria criteria,
                                 BiConsumer<String, Integer> onText,
                                 IntConsumer onToken) {
        Objects.requireNonNull(prompt, "prompt");
        Objects.requireNonNull(gen, "gen");
        if (criteria == null) {
            criteria = StoppingCriteria.stopOnTokens(gen.maxNewTokens);
        }
        final StoppingCriteria finalCriteria = criteria;
        // We always need a streamer for the chat pipeline — create one on the fly if the
        // caller didn't supply one (rare; the engine always does).
        TextIteratorStreamer effective = streamer == null ? new TextIteratorStreamer() : streamer;
        if (stub) {
            // Decoupled mode: emit a single "stub reply" chunk so the UI can be exercised
            // without a real model loaded (useful in CI / unit tests).
            String reply = stubPrefix + prompt;
            if (reply.length() > 256) reply = reply.substring(0, 256);
            effective.putText(reply);
            if (onText != null) onText.accept(reply, -1);
            effective.end();
            return;
        }
        int[] promptIds;
        try {
            promptIds = tokenizer.encode(prompt, true).ids();
        } catch (Throwable t) {
            throw new RuntimeException("Failed to tokenize prompt", t);
        }
        // Use the framework streaming generator: it invokes the IntConsumer for every
        // newly sampled token id. We decode each one and forward to the streamer.
        List<Integer> lastTokens = new java.util.ArrayList<>(gen.maxNewTokens + 16);

        // Wrapper IntConsumer — calls onToken then onText (with the decoded piece).
        IntConsumer wrapped = tokenId -> {
            lastTokens.add(tokenId);
            if (finalCriteria.shouldStop(lastTokens)) {
                // Abort the loop. Generator.generate checks shouldStop? No — it relies
                // on the caller to honor its return. Trick: throw to unwind.
                throw new StopGenerationException();
            }
            if (tokenId >= 0) {
                String decoded;
                try {
                    decoded = tokenizer.decode(new int[]{tokenId}, true);
                } catch (Throwable t) {
                    decoded = "";
                }
                if (decoded != null && !decoded.isEmpty()) {
                    effective.putText(decoded);
                    if (onText != null) onText.accept(decoded, tokenId);
                }
            }
            if (onToken != null) onToken.accept(tokenId);
        };
        boolean ended = false;
        try {
            int[] all = Generator.generate(model, promptIds, gen, maxContext, wrapped);
            // Sanity: count the tokens we generated (all.length - promptIds.length).
            ended = true;
            effective.markEnd();
            // If the last token was an EOS but shouldStop didn't trigger (e.g. eosStop=false),
            // we still want callers to know we reached end-of-sequence.
            if (lastTokens.isEmpty() && all.length > promptIds.length) {
                for (int i = promptIds.length; i < all.length; i++) lastTokens.add(all[i]);
            }
        } catch (StopGenerationException sge) {
            ended = true;
            effective.markEnd();
        } catch (Throwable t) {
            effective.markEnd();
            throw t;
        } finally {
            if (!ended) effective.markEnd();
        }
    }

    /**
     * Sentinel exception used by {@link #predictStreaming} to unwind out of the
     * {@code Generator.generate} loop once the {@link StoppingCriteria} triggers a stop.
     */
    public static final class StopGenerationException extends RuntimeException {
        public StopGenerationException() { super("generation stopped"); }
    }
}