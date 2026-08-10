/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.providers;

import org.bytedeco.pytorch.nn.Module;

import java.util.Objects;

/**
 * DeepSeek provider integration.
 *
 * <p>DeepSeek supports:
 * <ul>
 *   <li>Long context up to 128K tokens</li>
 *   <li>Code generation (DeepSeek Coder)</li>
 *   <li>Math reasoning (DeepSeek Math)</li>
 *   <li>RLHF and GRPO training</li>
 * </ul>
 *
 * <p>Reference: DeepSeek AI
 */
public class DeepSeekProvider implements LLMProvider {
    private volatile boolean closed;

    private final String apiKey;
    private final String model;
    private final String baseUrl;
    private final int maxTokens;
    private final double temperature;

    // DeepSeek-specific
    private final boolean enableGRPO;
    private final boolean enableThinking;

    public DeepSeekProvider(String apiKey, String model) {
        this(apiKey, model, null, 4096, 0.7, false, false);
    }

    public DeepSeekProvider(String apiKey, String model, String baseUrl, int maxTokens,
                           double temperature, boolean enableGRPO, boolean enableThinking) {
        this.apiKey = Objects.requireNonNull(apiKey, "apiKey");
        this.model = model != null ? model : "deepseek-chat";
        this.baseUrl = baseUrl != null ? baseUrl : "https://api.deepseek.com";
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        this.enableGRPO = enableGRPO;
        this.enableThinking = enableThinking;
    }

    @Override
    public String name() { return "DeepSeek"; }

    @Override
    public boolean supportsRLHF() { return true; }

    @Override
    public boolean supportsFunctionCalling() { return true; }

    @Override
    public int maxContextLength() {
        // DeepSeek supports up to 128K context
        switch (model) {
            case "deepseek-chat": return 32 * 1024;
            case "deepseek-coder": return 128 * 1024;
            case "deepseek-math": return 8 * 1024;
            case "deepseek-v2": return 128 * 1024;
            case "deepseek-v2.5": return 128 * 1024;
            default: return 64 * 1024;
        }
    }

    @Override
    public String[] supportedModalities() {
        if (model != null && model.contains("coder")) {
            return new String[]{"text", "code"};
        }
        if (model != null && model.contains("math")) {
            return new String[]{"text", "math"};
        }
        return new String[]{"text", "code", "math"};
    }

    /**
     * Check if GRPO training is enabled.
     */
    public boolean hasGRPOSupport() { return enableGRPO; }

    /**
     * Check if thinking mode is enabled.
     */
    public boolean hasThinkingSupport() { return enableThinking; }

    /**
     * Create a DeepSeek provider builder.
     */
    public static Builder builder() { return new Builder(); }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.println("[DeepSeekProvider] Closed: model=" + model + ", grpo=" + enableGRPO);
    }

    public boolean isClosed() { return closed; }

    public static final class Builder {
        private String apiKey;
        private String model = "deepseek-chat";
        private String baseUrl = "https://api.deepseek.com";
        private int maxTokens = 4096;
        private double temperature = 0.7;
        private boolean enableGRPO = true;
        private boolean enableThinking = false;

        public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }
        public Builder model(String model) { this.model = model; return this; }
        public Builder baseUrl(String baseUrl) { this.baseUrl = baseUrl; return this; }
        public Builder maxTokens(int maxTokens) { this.maxTokens = maxTokens; return this; }
        public Builder temperature(double temperature) { this.temperature = temperature; return this; }
        public Builder enableGRPO(boolean enableGRPO) { this.enableGRPO = enableGRPO; return this; }
        public Builder enableThinking(boolean enableThinking) { this.enableThinking = enableThinking; return this; }

        public DeepSeekProvider build() {
            return new DeepSeekProvider(apiKey, model, baseUrl, maxTokens, temperature, enableGRPO, enableThinking);
        }
    }
}
