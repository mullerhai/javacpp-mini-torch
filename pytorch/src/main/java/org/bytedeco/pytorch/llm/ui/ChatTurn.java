/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * One chat turn: a single user message + the assistant reply that resulted from it.
 *
 * <p>Created by {@code ChatEngine.stream()} on a {@code POST /v1/chat/{id}/turn} request,
 * mutated incrementally as tokens arrive, then snapshotted into a {@code SessionRegistry}
 * history record.
 *
 * <p>Wire format (JSON):
 * <pre>
 * {
 *   "turnId": "uuid",
 *   "sessionId": "uuid",
 *   "tool": "marketing",
 *   "userMessage": "Hello",
 *   "assistantText": "Hi there!",
 *   "assistantTokens": ["Hi"," there","!"],
 *   "stopReason": "eos" | "max_tokens" | "stop_string" | "user_cancel" | "error",
 *   "elapsedMs": 246,
 *   "tokensGenerated": 12,
 *   "tokensPerSecond": 48.78,
 *   "createdAtMillis": 1724000000000,
 *   "finishedAtMillis": 1724000000246,
 *   "request": {
 *     "temperature": 0.7, "top_p": 0.9, "top_k": 40,
 *     "max_new_tokens": 256, "repetition_penalty": 1.05,
 *     "stop_strings": ["User:"]
 *   }
 * }
 * </pre>
 */
public final class ChatTurn {

    public static final String STOP_EOS = "eos";
    public static final String STOP_MAX_TOKENS = "max_tokens";
    public static final String STOP_STRING = "stop_string";
    public static final String STOP_USER_CANCEL = "user_cancel";
    public static final String STOP_ERROR = "error";

    public final String turnId;
    public final String sessionId;
    public final String userMessage;
    public final long createdAtMillis;
    public final RequestParams request = new RequestParams();
    /** Mutable: chunks appended as they arrive from the streamer. */
    public final List<String> assistantTokens = new ArrayList<>();
    /** Mutable: most-recent stop reason; defaults to {@link #STOP_MAX_TOKENS}. */
    public volatile String stopReason = STOP_MAX_TOKENS;
    public volatile long finishedAtMillis = 0L;
    public volatile long elapsedMs = 0L;
    public volatile int tokensGenerated = 0;
    public volatile double tokensPerSecond = 0.0;
    public volatile String errorMessage;
    public volatile String tool;

    public ChatTurn(String turnId, String sessionId, String userMessage, String tool) {
        this.turnId = Objects.requireNonNull(turnId, "turnId");
        this.sessionId = Objects.requireNonNull(sessionId, "sessionId");
        this.userMessage = userMessage == null ? "" : userMessage;
        this.tool = tool == null ? "base" : tool;
        this.createdAtMillis = System.currentTimeMillis();
    }

    public String assistantText() {
        if (assistantTokens.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        for (String t : assistantTokens) sb.append(t);
        return sb.toString();
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("turnId", turnId);
        m.put("sessionId", sessionId);
        m.put("tool", tool);
        m.put("userMessage", userMessage);
        m.put("assistantText", assistantText());
        m.put("assistantTokens", new ArrayList<>(assistantTokens));
        m.put("stopReason", stopReason);
        m.put("elapsedMs", elapsedMs);
        m.put("tokensGenerated", tokensGenerated);
        m.put("tokensPerSecond", tokensPerSecond);
        m.put("createdAtMillis", createdAtMillis);
        m.put("finishedAtMillis", finishedAtMillis);
        m.put("request", request.toMap());
        if (errorMessage != null) m.put("errorMessage", errorMessage);
        return m;
    }

    /**
     * Per-turn request overrides supplied by the client. Field names match the
     * {@code GenerationConfig} Builder setters so we can reuse the builder via
     * {@code RequestParams#applyTo(GenerationConfig.Builder)}.
     */
    public static final class RequestParams {
        public Double temperature;
        public Double topP;
        public Integer topK;
        public Integer maxNewTokens;
        public Double repetitionPenalty;
        public List<String> stopStrings = new ArrayList<>();
        public Integer repetitionLimit;
        public Boolean doSample;

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            if (temperature != null) m.put("temperature", temperature);
            if (topP != null) m.put("top_p", topP);
            if (topK != null) m.put("top_k", topK);
            if (maxNewTokens != null) m.put("max_new_tokens", maxNewTokens);
            if (repetitionPenalty != null) m.put("repetition_penalty", repetitionPenalty);
            if (!stopStrings.isEmpty()) m.put("stop_strings", stopStrings);
            if (repetitionLimit != null) m.put("repetition_limit", repetitionLimit);
            if (doSample != null) m.put("do_sample", doSample);
            return m;
        }

        /**
         * Apply the request overrides onto a {@link org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig.Builder}.
         */
        public void applyTo(org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig.Builder b) {
            if (b == null) return;
            if (temperature != null) b.temperature(temperature);
            if (topP != null) b.topP(topP);
            if (topK != null) b.topK(topK);
            if (maxNewTokens != null) b.maxNewTokens(maxNewTokens);
            if (repetitionPenalty != null) b.repetitionPenalty(repetitionPenalty);
            if (doSample != null) b.doSample(doSample);
        }
    }
}