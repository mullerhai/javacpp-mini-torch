/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.openai;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * OpenAI client mirror. Used by Ex14 (OpenAI fine-tuning) and the MLflow evaluation example.
 * Without a real HTTP layer we build a request map and dispatch the response synthetically.
 */
public final class OpenAIClient {

    private final String apiKey;
    private final String organization;
    private final String baseUrl;

    public OpenAIClient(String apiKey) { this(apiKey, null, "https://api.openai.com/v1"); }

    public OpenAIClient(String apiKey, String organization, String baseUrl) {
        this.apiKey = apiKey;
        this.organization = organization;
        this.baseUrl = baseUrl;
    }

    public FileRef uploadFile(String filePath, String purpose) {
        return new FileRef(filePath, "file-stub");
    }

    public FineTuningJob createFineTuningJob(FineTuningJob.Spec spec) {
        return new FineTuningJob(spec);
    }

    public ChatCompletion chat(ChatRequest req) {
        return new ChatCompletion(req.messages.get(req.messages.size() - 1).get("content"));
    }

    public String apiKey() { return apiKey; }
    public String organization() { return organization; }
    public String baseUrl() { return baseUrl; }

    public static final class FileRef {
        public final String path;
        public final String id;
        public FileRef(String path, String id) { this.path = path; this.id = id; }
    }

    public static final class FineTuningJob {
        public static final class Spec {
            public String trainingFile;
            public String validationFile;
            public String model = "gpt-3.5-turbo";
            public String suffix;
            public int nEpochs = 3;
            public boolean validationFileSet = false;
        }
        public final Spec spec;
        public final String id;
        public final String status;
        public FineTuningJob(Spec spec) {
            this.spec = spec;
            this.id = "ft-job-" + System.currentTimeMillis();
            this.status = "queued";
        }
    }

    public static final class ChatRequest {
        public final String model;
        public final List<Map<String, String>> messages;
        public final double temperature;
        public ChatRequest(String model, List<Map<String, String>> messages, double temperature) {
            this.model = model;
            this.messages = messages;
            this.temperature = temperature;
        }
    }

    public static final class ChatCompletion {
        public final String content;
        public ChatCompletion(String content) { this.content = content; }
    }
}