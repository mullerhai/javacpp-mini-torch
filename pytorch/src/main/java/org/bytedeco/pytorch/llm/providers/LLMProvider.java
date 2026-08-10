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

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Base interface for LLM provider integration.
 *
 * <p>Each provider (Kimi, DeepSeek, GLM, Claude, Grok, GPT, MiniMax) implements
 * this interface with provider-specific configurations and API calls.
 */
public interface LLMProvider extends AutoCloseable {

    /**
     * Get the provider name.
     */
    String name();

    /**
     * Check if provider supports RLHF training.
     */
    boolean supportsRLHF();

    /**
     * Check if provider supports native function calling.
     */
    boolean supportsFunctionCalling();

    /**
     * Get context window size.
     */
    int maxContextLength();

    /**
     * Get supported input modalities.
     */
    default String[] supportedModalities() {
        return new String[]{"text"};
    }

    /**
     * Provider configuration holder.
     */
    class Config {
        private final String apiKey;
        private final String baseUrl;
        private final String model;
        private final int maxTokens;
        private final double temperature;
        private final Map<String, Object> extraParams;

        public Config(String apiKey, String baseUrl, String model,
                    int maxTokens, double temperature, Map<String, Object> extraParams) {
            this.apiKey = Objects.requireNonNull(apiKey, "apiKey");
            this.baseUrl = baseUrl;
            this.model = model;
            this.maxTokens = maxTokens;
            this.temperature = temperature;
            this.extraParams = extraParams != null ? extraParams : new ConcurrentHashMap<>();
        }

        public String apiKey() { return apiKey; }
        public String baseUrl() { return baseUrl; }
        public String model() { return model; }
        public int maxTokens() { return maxTokens; }
        public double temperature() { return temperature; }
        public Map<String, Object> extraParams() { return extraParams; }

        public static Builder builder() { return new Builder(); }

        public static class Builder {
            private String apiKey;
            private String baseUrl;
            private String model = "default";
            private int maxTokens = 4096;
            private double temperature = 0.7;
            private Map<String, Object> extraParams;

            public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }
            public Builder baseUrl(String baseUrl) { this.baseUrl = baseUrl; return this; }
            public Builder model(String model) { this.model = model; return this; }
            public Builder maxTokens(int maxTokens) { this.maxTokens = maxTokens; return this; }
            public Builder temperature(double temperature) { this.temperature = temperature; return this; }
            public Builder extraParams(Map<String, Object> extraParams) { this.extraParams = extraParams; return this; }

            public Config build() {
                return new Config(apiKey, baseUrl, model, maxTokens, temperature, extraParams);
            }
        }
    }

    /**
     * Create a provider from config.
     */
    static LLMProvider create(Config config) {
        throw new UnsupportedOperationException("Provider creation requires specific implementation");
    }
}
