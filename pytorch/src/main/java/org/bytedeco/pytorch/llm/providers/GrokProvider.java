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
 * Grok (xAI) provider integration.
 *
 * <p>Grok supports:
 * <ul>
 *   <li>Real-time information access</li>
 *   <li>Sarcastic personality</li>
 *   <li>Long context</li>
 *   <li>PPO training</li>
 * </ul>
 *
 * <p>Reference: xAI Grok
 */
public class GrokProvider implements LLMProvider {
    private volatile boolean closed;

    private final String apiKey;
    private final String model;
    private final String baseUrl;
    private final int maxTokens;
    private final double temperature;

    // Grok-specific
    private final boolean enableSearch;
    private final boolean enablePersonality;

    public GrokProvider(String apiKey, String model) {
        this(apiKey, model, null, 4096, 0.7, true, true);
    }

    public GrokProvider(String apiKey, String model, String baseUrl, int maxTokens,
                      double temperature, boolean enableSearch, boolean enablePersonality) {
        this.apiKey = Objects.requireNonNull(apiKey, "apiKey");
        this.model = model != null ? model : "grok-2";
        this.baseUrl = baseUrl != null ? baseUrl : "https://api.x.ai/v1";
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        this.enableSearch = enableSearch;
        this.enablePersonality = enablePersonality;
    }

    @Override
    public String name() { return "Grok"; }

    @Override
    public boolean supportsRLHF() { return true; }

    @Override
    public boolean supportsFunctionCalling() { return true; }

    @Override
    public int maxContextLength() {
        switch (model) {
            case "grok-2":
            case "grok-2-mini":
                return 131072;
            case "grok-1":
            case "grok-1.5":
                return 128 * 1024;
            default:
                return 32 * 1024;
        }
    }

    @Override
    public String[] supportedModalities() {
        return new String[]{"text", "code", "web"};
    }

    /**
     * Check if search capability is enabled.
     */
    public boolean hasSearchSupport() { return enableSearch; }

    /**
     * Check if sarcastic personality is enabled.
     */
    public boolean hasPersonalitySupport() { return enablePersonality; }

    public static Builder builder() { return new Builder(); }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.println("[GrokProvider] Closed: model=" + model);
    }

    public boolean isClosed() { return closed; }

    public static final class Builder {
        private String apiKey;
        private String model = "grok-2";
        private String baseUrl = "https://api.x.ai/v1";
        private int maxTokens = 4096;
        private double temperature = 0.7;
        private boolean enableSearch = true;
        private boolean enablePersonality = true;

        public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }
        public Builder model(String model) { this.model = model; return this; }
        public Builder baseUrl(String baseUrl) { this.baseUrl = baseUrl; return this; }
        public Builder maxTokens(int maxTokens) { this.maxTokens = maxTokens; return this; }
        public Builder temperature(double temperature) { this.temperature = temperature; return this; }
        public Builder enableSearch(boolean enableSearch) { this.enableSearch = enableSearch; return this; }
        public Builder enablePersonality(boolean enablePersonality) { this.enablePersonality = enablePersonality; return this; }

        public GrokProvider build() {
            return new GrokProvider(apiKey, model, baseUrl, maxTokens, temperature, enableSearch, enablePersonality);
        }
    }
}
