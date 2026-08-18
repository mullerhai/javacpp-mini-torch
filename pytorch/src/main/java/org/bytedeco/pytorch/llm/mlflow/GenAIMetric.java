/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.mlflow;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Mirror of {@code mlflow.metrics.genai.make_genai_metric}. Captures name, definition,
 * grading prompt, examples, model, greater-is-better.
 */
public final class GenAIMetric {

    public static final class EvaluationExample {
        public final String input;
        public final String output;
        public final String score;
        public final String justification;
        public EvaluationExample(String input, String output, String score, String justification) {
            this.input = input;
            this.output = output;
            this.score = score;
            this.justification = justification;
        }
    }

    public final String name;
    public final String definition;
    public final String gradingPrompt;
    public final String version;
    public final List<EvaluationExample> examples;
    public final String model;
    public final boolean greaterIsBetter;

    public GenAIMetric(String name, String definition, String gradingPrompt, String version,
                        List<EvaluationExample> examples, String model, boolean greaterIsBetter) {
        this.name = name;
        this.definition = definition;
        this.gradingPrompt = gradingPrompt;
        this.version = version;
        this.examples = examples == null ? new ArrayList<>() : examples;
        this.model = model;
        this.greaterIsBetter = greaterIsBetter;
    }

    /** Concrete sub-metrics used in the tutorials. */
    public static GenAIMetric faithfulness(String modelName) {
        return new GenAIMetric(
                "faithfulness",
                "Whether the answer is faithful to the retrieved context.",
                "Score 1-5: how faithful is the answer to the context?",
                "1.0",
                new ArrayList<>(),
                modelName,
                true);
    }

    public static GenAIMetric relevance(String modelName) {
        return new GenAIMetric(
                "answer_relevance",
                "Whether the answer is relevant to the question.",
                "Score 1-5: how relevant is the answer to the question?",
                "1.0",
                new ArrayList<>(),
                modelName,
                true);
    }

    public static GenAIMetric answerCorrectness(String modelName) {
        return new GenAIMetric(
                "answer_correctness",
                "Whether the answer is correct compared to the ground truth.",
                "Score 1-5: how correct is the answer?",
                "1.0",
                new ArrayList<>(),
                modelName,
                true);
    }

    public Map<String, Object> toDict() {
        return Map.of(
                "name", name,
                "definition", definition,
                "model", model,
                "greater_is_better", greaterIsBetter);
    }
}