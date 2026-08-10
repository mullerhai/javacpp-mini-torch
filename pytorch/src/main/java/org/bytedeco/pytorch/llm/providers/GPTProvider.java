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
 * GPT (OpenAI) provider integration.
 *
 * <p>GPT supports:
 * <ul>
 *   <li>GPT-4, GPT-4o, GPT-4 Turbo</li>
 *   <li>Vision support</li>
 * *   <li>Function calling</li>
 *   <li>DPO/PPO training support</li>
 * </ul>
 *
 * <p>Reference: OpenAI GPT
 */
public class GPTProvider implements LLMProvider {
    private volatile boolean closed;

    private final String apiKey;
    private final String model;
    private final String baseUrl;
    private final int maxTokens;
    private final double temperature;

    // GPT-specific
    private final String organization;
    private final boolean enableVision;
    private final String responseFormat;

    public GPTProvider(String apiKey, String model) {
        this(apiKey, model, null, 4096, 0.7, null, false, null);
    }

    public GPTProvider(String apiKey, String model, String baseUrl, int maxTokens,
                    double temperature, String organization, boolean enableVision,
                    String responseFormat) {
        this.apiKey = Objects.requireNonNull(apiKey, "apiKey");
        this.model = model != null ? model : "gpt-4o";
        this.baseUrl = baseUrl != null ? baseUrl : "https://api.openai.com/v1";
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        this.organization = organization;
        this.enableVision = enableVision;
        this.responseFormat = responseFormat;
    }

    @Override
    public String name() { return "GPT"; }

    @Override
    public boolean supportsRLHF() { return true; }

    @Override
    public boolean supportsFunctionCalling() { return true; }

    @Override
    public int maxContextLength() {
        switch (model) {
            case "gpt-4o":
            case "gpt-4o-mini":
                return 128 * 1024;
            case "gpt-4-turbo":
            case "gpt-4-turbo-2024-04-09":
                return 128 * 1024;
            case "gpt-4":
            case "gpt-4-32k":
                return 32 * 1024;
            case "gpt-3.5-turbo":
            case "gpt-3.5-turbo-16k":
                return 16 * 1024;
            case "o1-preview":
            case "o1-mini":
                return 128 * 1024;
            default:
                return 32 * 1024;
        }
    }

    @Override
    public String[] supportedModalities() {
        if (enableVision || model.contains("4o") || model.contains("4v") || model.contains("vision")) {
            return new String[]{"text", "image"};
        }
        return new String[]{"text"};
    }

    /**
     * Get OpenAI organization ID.
     */
    public String getOrganization() { return organization; }

    /**
     * Check if vision is enabled.
     */
    public boolean hasVisionSupport() { return enableVision; }

    /**
     * Get response format.
     */
    public String getResponseFormat() { return responseFormat; }

    public static Builder builder() { return new Builder(); }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.println("[GPTProvider] Closed: model=" + model);
    }

    public boolean isClosed() { return closed; }

    public static final class Builder {
        private String apiKey;
        private String model = "gpt-4o";
        private String baseUrl = "https://api.openai.com/v1";
        private int maxTokens = 4096;
        private double temperature = 0.7;
        private String organization;
        private boolean enableVision = true;
        private String responseFormat;

        public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }
        public Builder model(String model) { this.model = model; return this; }
        public Builder baseUrl(String baseUrl) { this.baseUrl = baseUrl; return this; }
        public Builder maxTokens(int maxTokens) { this.maxTokens = maxTokens; return this; }
        public Builder temperature(double temperature) { this.temperature = temperature; return this; }
        public Builder organization(String organization) { this.organization = organization; return this; }
        public Builder enableVision(boolean enableVision) { this.enableVision = enableVision; return this; }
        public Builder responseFormat(String responseFormat) { this.responseFormat = responseFormat; return this; }

        public GPTProvider build() {
            return new GPTProvider(apiKey, model, baseUrl, maxTokens, temperature, organization, enableVision, responseFormat);
        }
    }
}
