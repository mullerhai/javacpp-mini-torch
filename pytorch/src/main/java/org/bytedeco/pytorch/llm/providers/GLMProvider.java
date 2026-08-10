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
 * GLM (Zhipu AI) provider integration.
 *
 * <p>GLM supports:
 * <ul>
 *   <li>Long context up to 128K tokens</li>
 *   <li>Vision support (GLM-4V)</li>
 *   <li>Function calling</li>
 *   <li>PPO/DPO training support</li>
 * </ul>
 *
 * <p>Reference: Zhipu AI GLM
 */
public class GLMProvider implements LLMProvider {
    private volatile boolean closed;

    private final String apiKey;
    private final String model;
    private final String baseUrl;
    private final int maxTokens;
    private final double temperature;

    // GLM-specific
    private final boolean enableVision;
    private final boolean enableCPT;  // Custom Prompt Tuning

    public GLMProvider(String apiKey, String model) {
        this(apiKey, model, null, 4096, 0.7, false, false);
    }

    public GLMProvider(String apiKey, String model, String baseUrl, int maxTokens,
                     double temperature, boolean enableVision, boolean enableCPT) {
        this.apiKey = Objects.requireNonNull(apiKey, "apiKey");
        this.model = model != null ? model : "glm-4";
        this.baseUrl = baseUrl != null ? baseUrl : "https://open.bigmodel.cn/api/paas/v4";
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        this.enableVision = enableVision;
        this.enableCPT = enableCPT;
    }

    @Override
    public String name() { return "GLM"; }

    @Override
    public boolean supportsRLHF() { return true; }

    @Override
    public boolean supportsFunctionCalling() { return true; }

    @Override
    public int maxContextLength() {
        switch (model) {
            case "glm-4": return 128 * 1024;
            case "glm-4-flash": return 128 * 1024;
            case "glm-4-plus": return 128 * 1024;
            case "glm-4v": return 8 * 1024;
            case "glm-3-turbo": return 128 * 1024;
            default: return 32 * 1024;
        }
    }

    @Override
    public String[] supportedModalities() {
        if (enableVision || (model != null && model.contains("4v"))) {
            return new String[]{"text", "image"};
        }
        return new String[]{"text"};
    }

    /**
     * Check if vision support is enabled.
     */
    public boolean hasVisionSupport() { return enableVision; }

    /**
     * Check if Custom Prompt Tuning is enabled.
     */
    public boolean hasCPTSupport() { return enableCPT; }

    public static Builder builder() { return new Builder(); }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.println("[GLMProvider] Closed: model=" + model);
    }

    public boolean isClosed() { return closed; }

    public static final class Builder {
        private String apiKey;
        private String model = "glm-4";
        private String baseUrl = "https://open.bigmodel.cn/api/paas/v4";
        private int maxTokens = 4096;
        private double temperature = 0.7;
        private boolean enableVision = false;
        private boolean enableCPT = false;

        public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }
        public Builder model(String model) { this.model = model; return this; }
        public Builder baseUrl(String baseUrl) { this.baseUrl = baseUrl; return this; }
        public Builder maxTokens(int maxTokens) { this.maxTokens = maxTokens; return this; }
        public Builder temperature(double temperature) { this.temperature = temperature; return this; }
        public Builder enableVision(boolean enableVision) { this.enableVision = enableVision; return this; }
        public Builder enableCPT(boolean enableCPT) { this.enableCPT = enableCPT; return this; }

        public GLMProvider build() {
            return new GLMProvider(apiKey, model, baseUrl, maxTokens, temperature, enableVision, enableCPT);
        }
    }
}
