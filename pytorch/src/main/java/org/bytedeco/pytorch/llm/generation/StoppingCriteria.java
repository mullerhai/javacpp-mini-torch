/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.generation;

import java.util.ArrayList;
import java.util.List;

/**
 * Hugging Face {@code StoppingCriteria} / {@code StopOnTokens} equivalent.
 *
 * <p>Returns {@code true} (stop) when the last {@code N} tokens match any caller-supplied
 * stop sequence. Used by Ex22 (Gradio) and Ex33 (Unsloth).
 */
public final class StoppingCriteria {

    @FunctionalInterface
    public interface Predicate {
        boolean test(List<Integer> lastTokens);
    }

    private final List<int[]> stopSequences = new ArrayList<>();
    private final List<Predicate> predicates = new ArrayList<>();
    private final int window;

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
        List<Integer> win = history.size() <= window
                ? history
                : history.subList(history.size() - window, history.size());
        for (Predicate p : predicates) {
            if (p.test(win)) return true;
        }
        return false;
    }
}