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
 * Mirror of {@code mlflow.evaluate(model, eval_df, model_type, evaluators, extra_metrics, evaluator_config)}.
 */
public final class MLflowEvaluate {

    public static final class EvalResult {
        public final List<GenAIMetric> metrics;
        public final Map<String, Double> aggregate;
        public EvalResult(List<GenAIMetric> metrics, Map<String, Double> aggregate) {
            this.metrics = metrics;
            this.aggregate = aggregate;
        }
    }

    public static EvalResult evaluate(Object model, List<Map<String, Object>> evalData,
                                       String modelType, String evaluators,
                                       List<GenAIMetric> metrics,
                                       Map<String, String> colMapping) {
        EvalResult r = new EvalResult(metrics == null ? new ArrayList<>() : metrics,
                new java.util.LinkedHashMap<>());
        for (GenAIMetric m : r.metrics) {
            // Naive proxy: average over the answer_correctness metric by computing per-row
            // accuracy and emitting the value.
            double sum = 0;
            int n = 0;
            for (Map<String, Object> row : evalData) {
                Object ans = row.get("answer");
                Object truth = row.get("ground_truth");
                if (ans == null || truth == null) continue;
                if (ans.toString().equalsIgnoreCase(truth.toString())) sum += 1.0;
                n++;
            }
            r.aggregate.put(m.name, n == 0 ? 0.0 : sum / n);
        }
        return r;
    }
}