/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ragas;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * RAGAS-style metric computations. Implements the surface used by the parent repository's
 * notebooks: {@code Faithfulness}, {@code AnswerRelevancy}, {@code ContextPrecision},
 * {@code ContextRecall}, {@code AnswerCorrectness}.
 */
public final class RAGASMetrics {

    public static final class RagasSample {
        public final String question;
        public final String answer;
        public final String groundTruth;
        public final List<String> contexts;
        public RagasSample(String question, String answer, String groundTruth, List<String> contexts) {
            this.question = question;
            this.answer = answer;
            this.groundTruth = groundTruth;
            this.contexts = contexts;
        }
    }

    public static final class RagasReport {
        public final double faithfulness;
        public final double answerRelevancy;
        public final double contextPrecision;
        public final double contextRecall;
        public final double answerCorrectness;
        public RagasReport(double f, double a, double cp, double cr, double acc) {
            this.faithfulness = f;
            this.answerRelevancy = a;
            this.contextPrecision = cp;
            this.contextRecall = cr;
            this.answerCorrectness = acc;
        }
        public Map<String, Double> toMap() {
            return java.util.Map.of(
                    "faithfulness", faithfulness,
                    "answer_relevancy", answerRelevancy,
                    "context_precision", contextPrecision,
                    "context_recall", contextRecall,
                    "answer_correctness", answerCorrectness);
        }
    }

    public static RagasReport evaluate(List<RagasSample> samples) {
        if (samples == null || samples.isEmpty()) {
            return new RagasReport(0, 0, 0, 0, 0);
        }
        double f = 0, a = 0, cp = 0, cr = 0, acc = 0;
        for (RagasSample s : samples) {
            f += faithfulness(s);
            a += answerRelevance(s);
            cp += contextPrecision(s);
            cr += contextRecall(s);
            acc += answerCorrectness(s);
        }
        int n = samples.size();
        return new RagasReport(f / n, a / n, cp / n, cr / n, acc / n);
    }

    public static double faithfulness(RagasSample s) {
        if (s.answer == null || s.contexts == null || s.contexts.isEmpty()) return 0.0;
        int hits = 0;
        for (String ctx : s.contexts) {
            for (String t : tokenize(s.answer)) {
                if (ctx.toLowerCase().contains(t.toLowerCase())) { hits++; break; }
            }
        }
        return hits / Math.max(1.0, s.contexts.size());
    }

    public static double answerRelevance(RagasSample s) {
        if (s.answer == null || s.question == null) return 0.0;
        // crude token overlap
        return tokenOverlap(s.answer, s.question);
    }

    public static double contextPrecision(RagasSample s) {
        if (s.contexts == null || s.contexts.isEmpty() || s.groundTruth == null) return 0.0;
        int hits = 0;
        for (String ctx : s.contexts) {
            for (String t : tokenize(s.groundTruth)) {
                if (ctx.toLowerCase().contains(t.toLowerCase())) { hits++; break; }
            }
        }
        return hits / (double) s.contexts.size();
    }

    public static double contextRecall(RagasSample s) {
        if (s.contexts == null || s.groundTruth == null) return 0.0;
        int hits = 0, total = 0;
        for (String t : tokenize(s.groundTruth)) {
            total++;
            for (String ctx : s.contexts) {
                if (ctx.toLowerCase().contains(t.toLowerCase())) { hits++; break; }
            }
        }
        return total == 0 ? 0 : (double) hits / total;
    }

    public static double answerCorrectness(RagasSample s) {
        if (s.answer == null || s.groundTruth == null) return 0.0;
        return tokenOverlap(s.answer, s.groundTruth);
    }

    private static List<String> tokenize(String text) {
        if (text == null) return new ArrayList<>();
        List<String> out = new ArrayList<>();
        for (String t : text.toLowerCase().split("\\W+")) if (!t.isEmpty() && t.length() > 2) out.add(t);
        return out;
    }

    private static double tokenOverlap(String a, String b) {
        List<String> ta = tokenize(a);
        if (ta.isEmpty()) return 0.0;
        int hits = 0;
        for (String t : ta) {
            if (b.toLowerCase().contains(t)) hits++;
        }
        return (double) hits / ta.size();
    }
}