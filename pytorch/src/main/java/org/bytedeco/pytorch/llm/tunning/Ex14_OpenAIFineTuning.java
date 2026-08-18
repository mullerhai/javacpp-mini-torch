/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.io.File;

import org.bytedeco.pytorch.llm.openai.OpenAIClient;

/** Ex14 — OpenAI fine-tuning mirror. */
public final class Ex14_OpenAIFineTuning {
    public static final String NAME = "Ex14_OpenAIFineTuning";

    public static void run() {
        TunningSupport.banner(14, "OpenAI fine-tuning");
        File out = new File("build/ex14_outputs"); out.mkdirs();

        OpenAIClient client = new OpenAIClient(System.getenv("OPENAI_API_KEY"));
        OpenAIClient.FileRef file = client.uploadFile("training_data.jsonl", "fine-tune");
        OpenAIClient.FineTuningJob.Spec spec = new OpenAIClient.FineTuningJob.Spec();
        spec.trainingFile = file.id;
        spec.model = "gpt-3.5-turbo";
        spec.nEpochs = 1;
        OpenAIClient.FineTuningJob job = client.createFineTuningJob(spec);
        System.out.println("[openai] job=" + job.id + " status=" + job.status);
    }

    public static void main(String[] args) { run(); }
}
