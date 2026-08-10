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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

import java.util.Map;
import java.util.Objects;

/**
 * Kimi (Moonshot AI) provider integration.
 *
 * <p>Kimi supports:
 * <ul>
 *   <li>Long context up to 200K tokens</li>
 *   <li>Native function calling</li>
 *   <li>RLHF training capability</li>
 * </ul>
 *
 * <p>Reference: Moonshot AI Kimi
 */
public class KimiProvider implements LLMProvider {
    private volatile boolean closed;

    private final String apiKey;
    private final String model;
    private final String baseUrl;
    private final int maxTokens;
    private final double temperature;

    // Training support
    private final boolean enableRLHF;
    private final Module rewardModel;

    public KimiProvider(String apiKey, String model) {
        this(apiKey, model, null, 4096, 0.7, false, null);
    }

    public KimiProvider(String apiKey, String model, String baseUrl, int maxTokens,
                       double temperature, boolean enableRLHF, Module rewardModel) {
        this.apiKey = Objects.requireNonNull(apiKey, "apiKey");
        this.model = model != null ? model : "moonshot-v1-8k";
        this.baseUrl = baseUrl != null ? baseUrl : "https://api.moonshot.cn/v1";
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        this.enableRLHF = enableRLHF;
        this.rewardModel = rewardModel;
    }

    @Override
    public String name() { return "Kimi"; }

    @Override
    public boolean supportsRLHF() { return enableRLHF; }

    @Override
    public boolean supportsFunctionCalling() { return true; }

    @Override
    public int maxContextLength() {
        // Kimi supports up to 200K context
        switch (model) {
            case "moonshot-v1-8k": return 8 * 1024;
            case "moonshot-v1-32k": return 32 * 1024;
            case "moonshot-v1-128k": return 128 * 1024;
            case "moonshot-v1-200k": return 200 * 1024;
            default: return 32 * 1024;
        }
    }

    @Override
    public String[] supportedModalities() {
        return new String[]{"text", "code"};
    }

    /**
     * Create a Kimi provider builder.
     */
    public static Builder builder() { return new Builder(); }

    /**
     * Get reward model for RLHF training.
     */
    public Module getRewardModel() { return rewardModel; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.println("[KimiProvider] Closed");
    }

    public boolean isClosed() { return closed; }

    public static final class Builder {
        private String apiKey;
        private String model = "moonshot-v1-32k";
        private String baseUrl = "https://api.moonshot.cn/v1";
        private int maxTokens = 4096;
        private double temperature = 0.7;
        private boolean enableRLHF = false;
        private Module rewardModel;

        public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }
        public Builder model(String model) { this.model = model; return this; }
        public Builder baseUrl(String baseUrl) { this.baseUrl = baseUrl; return this; }
        public Builder maxTokens(int maxTokens) { this.maxTokens = maxTokens; return this; }
        public Builder temperature(double temperature) { this.temperature = temperature; return this; }
        public Builder enableRLHF(boolean enableRLHF) { this.enableRLHF = enableRLHF; return this; }
        public Builder rewardModel(Module rewardModel) { this.rewardModel = rewardModel; return this; }

        public KimiProvider build() {
            return new KimiProvider(apiKey, model, baseUrl, maxTokens, temperature, enableRLHF, rewardModel);
        }
    }
}
