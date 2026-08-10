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
 * MiniMax provider integration.
 *
 * <p>MiniMax supports:
 * <ul>
 *   <li>Long context up to 1M tokens</li>
 *   <li>Speech-to-text</li>
 *   <li>Video generation</li>
 *   <li>Abab series models</li>
 * </ul>
 *
 * <p>Reference: MiniMax AI
 */
public class MiniMaxProvider implements LLMProvider {
    private volatile boolean closed;

    private final String apiKey;
    private final String model;
    private final String baseUrl;
    private final int maxTokens;
    private final double temperature;

    // MiniMax-specific
    private final boolean enableSpeechToText;
    private final boolean enableVideoGeneration;
    private final String groupId;

    public MiniMaxProvider(String apiKey, String model) {
        this(apiKey, model, null, 4096, 0.7, false, false, null);
    }

    public MiniMaxProvider(String apiKey, String model, String baseUrl, int maxTokens,
                         double temperature, boolean enableSpeechToText,
                         boolean enableVideoGeneration, String groupId) {
        this.apiKey = Objects.requireNonNull(apiKey, "apiKey");
        this.model = model != null ? model : "abab6-chat";
        this.baseUrl = baseUrl != null ? baseUrl : "https://api.minimax.chat/v1";
        this.maxTokens = maxTokens;
        this.temperature = temperature;
        this.enableSpeechToText = enableSpeechToText;
        this.enableVideoGeneration = enableVideoGeneration;
        this.groupId = groupId;
    }

    @Override
    public String name() { return "MiniMax"; }

    @Override
    public boolean supportsRLHF() { return true; }

    @Override
    public boolean supportsFunctionCalling() { return true; }

    @Override
    public int maxContextLength() {
        switch (model) {
            case "abab6-chat":
                return 1_000_000;  // 1M tokens!
            case "abab5.5-chat":
                return 256 * 1024;
            case "abab3-chat":
                return 32 * 1024;
            case "abab5-gs":
                return 32 * 1024;
            default:
                return 128 * 1024;
        }
    }

    @Override
    public String[] supportedModalities() {
        if (enableVideoGeneration) {
            return new String[]{"text", "audio", "video"};
        }
        if (enableSpeechToText) {
            return new String[]{"text", "audio"};
        }
        return new String[]{"text"};
    }

    /**
     * Check if speech-to-text is enabled.
     */
    public boolean hasSpeechToTextSupport() { return enableSpeechToText; }

    /**
     * Check if video generation is enabled.
     */
    public boolean hasVideoGenerationSupport() { return enableVideoGeneration; }

    /**
     * Get MiniMax group ID.
     */
    public String getGroupId() { return groupId; }

    public static Builder builder() { return new Builder(); }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.println("[MiniMaxProvider] Closed: model=" + model);
    }

    public boolean isClosed() { return closed; }

    public static final class Builder {
        private String apiKey;
        private String model = "abab6-chat";
        private String baseUrl = "https://api.minimax.chat/v1";
        private int maxTokens = 4096;
        private double temperature = 0.7;
        private boolean enableSpeechToText = false;
        private boolean enableVideoGeneration = false;
        private String groupId;

        public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }
        public Builder model(String model) { this.model = model; return this; }
        public Builder baseUrl(String baseUrl) { this.baseUrl = baseUrl; return this; }
        public Builder maxTokens(int maxTokens) { this.maxTokens = maxTokens; return this; }
        public Builder temperature(double temperature) { this.temperature = temperature; return this; }
        public Builder enableSpeechToText(boolean enableSpeechToText) { this.enableSpeechToText = enableSpeechToText; return this; }
        public Builder enableVideoGeneration(boolean enableVideoGeneration) { this.enableVideoGeneration = enableVideoGeneration; return this; }
        public Builder groupId(String groupId) { this.groupId = groupId; return this; }

        public MiniMaxProvider build() {
            return new MiniMaxProvider(apiKey, model, baseUrl, maxTokens, temperature, enableSpeechToText, enableVideoGeneration, groupId);
        }
    }
}
