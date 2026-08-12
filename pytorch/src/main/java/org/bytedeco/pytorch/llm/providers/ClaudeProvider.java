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
 * Claude (Anthropic) provider integration.
 *
 * <p>Claude supports:
 * <ul>
 *   <li>Long context up to 200K tokens</li>
 *   <li>Vision support (Claude 3)</li>
 *   <li>Constitutional AI (CAI)</li>
 *   <li>RLHF and HHH alignment</li>
 * </ul>
 *
 * <p>Reference: Anthropic Claude
 */
public class ClaudeProvider implements LLMProvider {
    private volatile boolean closed;

    private final String apiKey;
    private final String model;
    private final String baseUrl;
    private final int maxTokens;
    private final double temperature;

    // Claude-specific
    private final boolean enableConstitutionalAI;
    private final String anthropicVersion;

    public ClaudeProvider(String apiKey, String model) {
        this(apiKey, model, null, 4096, 0.7, false, "2023-06-01");
    }

    public ClaudeProvider(String apiKey, String model, String baseUrl, int maxTokens,
                        double temperature, boolean enableConstitutionalAI, String anthropicVersion) {
        this.apiKey = Objects.requireNonNull(apiKey, "apiKey");
        this.model = model != null ? model : "claude-3-5-sonnet-20241022";
        this.baseUrl = baseUrl != null ? baseUrl : "https://api.anthropic.com/v1";
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        this.enableConstitutionalAI = enableConstitutionalAI;
        this.anthropicVersion = anthropicVersion;
    }

    @Override
    public String name() { return "Claude"; }

    @Override
    public boolean supportsRLHF() { return true; }

    @Override
    public boolean supportsFunctionCalling() { return true; }

    @Override
    public int maxContextLength() {
        switch (model) {
            case "claude-opus-4-20250514":
                return 200 * 1024;
            case "claude-3-5-sonnet-20241022":
            case "claude-3-5-haiku-20241022":
                return 200 * 1024;
            case "claude-3-opus":
            case "claude-3-sonnet":
            case "claude-3-haiku":
                return 200 * 1024;
            default:
                return 100 * 1024;
        }
    }

    @Override
    public String[] supportedModalities() {
        if (model != null && (model.contains("opus") || model.contains("sonnet") || model.contains("haiku"))) {
            if (!model.contains("3")) {
                return new String[]{"text", "image", "document"};
            }
        }
        return new String[]{"text"};
    }

    /**
     * Check if Constitutional AI is enabled.
     */
    public boolean hasConstitutionalAISupport() { return enableConstitutionalAI; }

    /**
     * Get Anthropic API version.
     */
    public String getAnthropicVersion() { return anthropicVersion; }

    /**
     * Constitutional AI trainer for Claude alignment.
     */
    public static class ConstitutionalAIConfig {
        private final String principles;
        private final double critiqueThreshold;
        private final int revisionSteps;

        public ConstitutionalAIConfig(String principles, double critiqueThreshold, int revisionSteps) {
            this.principles = principles;
            this.critiqueThreshold = critiqueThreshold;
            this.revisionSteps = revisionSteps;
        }

        public String principles() { return principles; }
        public double critiqueThreshold() { return critiqueThreshold; }
        public int revisionSteps() { return revisionSteps; }

        public static Builder builder() { return new Builder(); }

        public static class Builder {
            private String principles = "Helpful, Harmless, Honest";
            private double critiqueThreshold = 0.5;
            private int revisionSteps = 2;

            public Builder principles(String principles) { this.principles = principles; return this; }
            public Builder critiqueThreshold(double critiqueThreshold) { this.critiqueThreshold = critiqueThreshold; return this; }
            public Builder revisionSteps(int revisionSteps) { this.revisionSteps = revisionSteps; return this; }

            public ConstitutionalAIConfig build() {
                return new ConstitutionalAIConfig(principles, critiqueThreshold, revisionSteps);
            }
        }
    }

    public static Builder builder() { return new Builder(); }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.println("[ClaudeProvider] Closed: model=" + model);
    }

    public boolean isClosed() { return closed; }

    public static final class Builder {
        private String apiKey;
        private String model = "claude-3-5-sonnet-20241022";
        private String baseUrl = "https://api.anthropic.com/v1";
        private int maxTokens = 4096;
        private double temperature = 0.7;
        private boolean enableConstitutionalAI = false;
        private String anthropicVersion = "2023-06-01";

        public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }
        public Builder model(String model) { this.model = model; return this; }
        public Builder baseUrl(String baseUrl) { this.baseUrl = baseUrl; return this; }
        public Builder maxTokens(int maxTokens) { this.maxTokens = maxTokens; return this; }
        public Builder temperature(double temperature) { this.temperature = temperature; return this; }
        public Builder enableConstitutionalAI(boolean enableConstitutionalAI) { this.enableConstitutionalAI = enableConstitutionalAI; return this; }
        public Builder anthropicVersion(String anthropicVersion) { this.anthropicVersion = anthropicVersion; return this; }

        public ClaudeProvider build() {
            return new ClaudeProvider(apiKey, model, baseUrl, maxTokens, temperature, enableConstitutionalAI, anthropicVersion);
        }
    }
}
