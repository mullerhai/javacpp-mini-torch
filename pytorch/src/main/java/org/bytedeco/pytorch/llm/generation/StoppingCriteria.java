/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.generation;

import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

import java.util.ArrayList;
import java.util.List;

/**
 * Hugging Face {@code StoppingCriteria} / {@code StopOnTokens} equivalent.
 *
 * <p>Returns {@code true} (stop) when the last {@code N} tokens match any caller-supplied
 * stop sequence. Used by Ex22 (Gradio) and Ex33 (Unsloth).
 *
 * <p>Extensions added for the Web Demo (Phase 2 of WEB_DEMO_IMPLEMENTATION_PLAN.md):
 * <ul>
 *   <li>{@link #addStopString(String, FastTokenizer)} — register a free-text stop sequence
 *       (encoded with the supplied tokenizer); if the running decoded text ends with it,
 *       stop.</li>
 *   <li>{@link #addRepetitionLimit(int)} — stop when the last {@code n} generated tokens are
 *       identical (helps break infinite loops).</li>
 *   <li>{@link #reset()} — clear runtime history so the same criteria instance can be reused
 *       across turns.</li>
 * </ul>
 */
public final class StoppingCriteria {

    @FunctionalInterface
    public interface Predicate {
        boolean test(List<Integer> lastTokens);
    }

    private final List<int[]> stopSequences = new ArrayList<>();
    private final List<Predicate> predicates = new ArrayList<>();
    private final int window;
    /** Stop strings + their tokenized forms (Phase 2 extension). */
    private final List<String> stopStringTexts = new ArrayList<>();
    private final List<int[]> stopStringIds = new ArrayList<>();
    /** Consecutive identical token run length that triggers a stop (Phase 2 extension). */
    private int repetitionLimit = 0;

    public StoppingCriteria(int window) {
        this.window = window;
    }

    public static StoppingCriteria stopOnTokens(int maxTokensWatch) {
        StoppingCriteria c = new StoppingCriteria(maxTokensWatch);
        return c;
    }

    public StoppingCriteria addStopToken(int tokenId) {
        stopSequences.add(new int[]{tokenId});
        return this;
    }

    public StoppingCriteria addStopSequence(int[] ids) {
        stopSequences.add(ids.clone());
        return this;
    }

    public StoppingCriteria addPredicate(Predicate p) {
        predicates.add(p);
        return this;
    }

    /**
     * Register a free-text stop sequence. The string is encoded using the supplied tokenizer;
     * generation stops as soon as the running token history ends with this id sequence.
     * A null/empty string is ignored.
     */
    public StoppingCriteria addStopString(String s, FastTokenizer tokenizer) {
        if (s == null || s.isEmpty() || tokenizer == null) return this;
        int[] ids;
        try {
            ids = tokenizer.encode(s, false).ids();
        } catch (Throwable t) {
            // Best effort: refuse silently — caller gets to add via token ids directly.
            return this;
        }
        if (ids == null || ids.length == 0) return this;
        stopStringTexts.add(s);
        stopStringIds.add(ids);
        return this;
    }

    /**
     * Stop when the last {@code n} generated tokens are identical (defaults to 0 = disabled).
     * Useful to break loops such as "..." / "..." repetition.
     */
    public StoppingCriteria addRepetitionLimit(int n) {
        this.repetitionLimit = Math.max(0, n);
        return this;
    }

    public int repetitionLimit() { return repetitionLimit; }

    public List<String> stopStrings() {
        return java.util.Collections.unmodifiableList(stopStringTexts);
    }

    /**
     * Clear any internal per-turn state. Currently a no-op (criteria is stateless wrt
     * history since {@code shouldStop(history)} takes history as a parameter), but kept as
     * an explicit hook for future extensions.
     */
    public void reset() {
        // intentionally empty — predicates and stop sequences are configuration
    }

    public boolean shouldStop(List<Integer> history) {
        for (int[] seq : stopSequences) {
            if (seq.length == 0) continue;
            if (history.size() < seq.length) continue;
            boolean ok = true;
            for (int k = 0; k < seq.length; k++) {
                if (history.get(history.size() - seq.length + k) != seq[k]) { ok = false; break; }
            }
            if (ok) return true;
        }
        // Phase 2: free-text stop strings (tokenized)
        for (int[] seq : stopStringIds) {
            if (seq.length == 0) continue;
            if (history.size() < seq.length) continue;
            boolean ok = true;
            for (int k = 0; k < seq.length; k++) {
                if (history.get(history.size() - seq.length + k) != seq[k]) { ok = false; break; }
            }
            if (ok) return true;
        }
        // Phase 2: repetition limit
        if (repetitionLimit > 0 && history.size() >= repetitionLimit) {
            int last = history.get(history.size() - 1);
            boolean same = true;
            for (int k = 2; k <= repetitionLimit && same; k++) {
                if (history.get(history.size() - k) != last) same = false;
            }
            if (same) return true;
        }
        List<Integer> win = history.size() <= window
                ? history
                : history.subList(history.size() - window, history.size());
        for (Predicate p : predicates) {
            if (p.test(win)) return true;
        }
        return false;
    }
}