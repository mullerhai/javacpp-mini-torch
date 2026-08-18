/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.bytedeco.pytorch.llm.mlflow.MLflowClient;
import org.bytedeco.pytorch.llm.mlflow.MLflowEvaluate;
import org.bytedeco.pytorch.llm.mlflow.GenAIMetric;
import org.bytedeco.pytorch.llm.ragas.RAGASMetrics;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

/** Ex19 — MLflow evaluate + RAGAS metrics. */
public final class Ex19_MLflowEvaluate {

    public static final String NAME = "Ex19_MLflowEvaluate";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(19, "MLflow Evaluate + RAGAS Metrics");

        MLflowClient ml = new MLflowClient("tunning-eval", "build/mlruns");
        ml.logParam("model", "gpt-4-stub");
        ml.logMetric("loss", 0.4, 100);
        ml.logMetric("loss", 0.35, 200);

        List<Map<String, Object>> evalData = new ArrayList<>();
        Map<String, Object> r1 = new LinkedHashMap<>();
        r1.put("question", "What is the capital of France?");
        r1.put("answer", "Paris");
        r1.put("ground_truth", "Paris is the capital of France.");
        evalData.add(r1);
        Map<String, Object> r2 = new LinkedHashMap<>();
        r2.put("question", "Who wrote Hamlet?");
        r2.put("answer", "Shakespeare");
        r2.put("ground_truth", "William Shakespeare wrote Hamlet.");
        evalData.add(r2);

        List<GenAIMetric> metrics = List.of(
                GenAIMetric.faithfulness("openai:/gpt-4"),
                GenAIMetric.relevance("openai:/gpt-4"),
                GenAIMetric.answerCorrectness("openai:/gpt-4"));
        MLflowEvaluate.EvalResult result = MLflowEvaluate.evaluate(
                null, evalData, "question-answering", "default", metrics, Map.of());
        System.out.println("MLflow eval aggregate: " + result.aggregate);

        List<RAGASMetrics.RagasSample> samples = new ArrayList<>();
        for (Map<String, Object> row : evalData) {
            samples.add(new RAGASMetrics.RagasSample(
                    (String) row.get("question"),
                    (String) row.get("answer"),
                    (String) row.get("ground_truth"),
                    List.of((String) row.get("ground_truth"))));
        }
        RAGASMetrics.RagasReport ragas = RAGASMetrics.evaluate(samples);
        System.out.println("RAGAS: " + ragas.toMap());

        ml.logTextArtifact("ragas.json", ragas.toMap().toString()).end();
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("mlflow-eval")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
