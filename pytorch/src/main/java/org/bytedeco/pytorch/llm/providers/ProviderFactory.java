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
 * Factory for creating and managing LLM providers.
 *
 * <p>Supported providers:
 * <ul>
 *   <li>Kimi (Moonshot AI)</li>
 *   <li>DeepSeek</li>
 *   <li>GLM (Zhipu AI)</li>
 *   <li>Claude (Anthropic)</li>
 *   <li>Grok (xAI)</li>
 *   <li>GPT (OpenAI)</li>
 *   <li>MiniMax</li>
 * </ul>
 */
public final class ProviderFactory implements AutoCloseable {
    private volatile boolean closed;
    private final Map<String, LLMProvider> providers = new ConcurrentHashMap<>();

    public ProviderFactory() {}

    /**
     * Create a Kimi provider.
     */
    public KimiProvider createKimi(String apiKey) {
        return createKimi(apiKey, "moonshot-v1-32k");
    }

    public KimiProvider createKimi(String apiKey, String model) {
        KimiProvider provider = new KimiProvider(apiKey, model);
        providers.put("kimi-" + model, provider);
        return provider;
    }

    /**
     * Create a DeepSeek provider.
     */
    public DeepSeekProvider createDeepSeek(String apiKey) {
        return createDeepSeek(apiKey, "deepseek-chat");
    }

    public DeepSeekProvider createDeepSeek(String apiKey, String model) {
        DeepSeekProvider provider = new DeepSeekProvider(apiKey, model);
        providers.put("deepseek-" + model, provider);
        return provider;
    }

    /**
     * Create a GLM provider.
     */
    public GLMProvider createGLM(String apiKey) {
        return createGLM(apiKey, "glm-4");
    }

    public GLMProvider createGLM(String apiKey, String model) {
        GLMProvider provider = new GLMProvider(apiKey, model);
        providers.put("glm-" + model, provider);
        return provider;
    }

    /**
     * Create a Claude provider.
     */
    public ClaudeProvider createClaude(String apiKey) {
        return createClaude(apiKey, "claude-3-5-sonnet-20241022");
    }

    public ClaudeProvider createClaude(String apiKey, String model) {
        ClaudeProvider provider = new ClaudeProvider(apiKey, model);
        providers.put("claude-" + model, provider);
        return provider;
    }

    /**
     * Create a Grok provider.
     */
    public GrokProvider createGrok(String apiKey) {
        return createGrok(apiKey, "grok-2");
    }

    public GrokProvider createGrok(String apiKey, String model) {
        GrokProvider provider = new GrokProvider(apiKey, model);
        providers.put("grok-" + model, provider);
        return provider;
    }

    /**
     * Create a GPT provider.
     */
    public GPTProvider createGPT(String apiKey) {
        return createGPT(apiKey, "gpt-4o");
    }

    public GPTProvider createGPT(String apiKey, String model) {
        GPTProvider provider = new GPTProvider(apiKey, model);
        providers.put("gpt-" + model, provider);
        return provider;
    }

    /**
     * Create a MiniMax provider.
     */
    public MiniMaxProvider createMiniMax(String apiKey) {
        return createMiniMax(apiKey, "abab6-chat");
    }

    public MiniMaxProvider createMiniMax(String apiKey, String model) {
        MiniMaxProvider provider = new MiniMaxProvider(apiKey, model);
        providers.put("minimax-" + model, provider);
        return provider;
    }

    /**
     * Get a provider by name.
     */
    public LLMProvider get(String name) {
        return providers.get(Objects.requireNonNull(name, "name"));
    }

    /**
     * List all registered providers.
     */
    public Map<String, LLMProvider> all() {
        return Map.copyOf(providers);
    }

    /**
     * Check if a provider exists.
     */
    public boolean has(String name) {
        return providers.containsKey(Objects.requireNonNull(name, "name"));
    }

    /**
     * Remove and close a provider.
     */
    public boolean remove(String name) {
        LLMProvider provider = providers.remove(Objects.requireNonNull(name, "name"));
        if (provider != null) {
            try { provider.close(); } catch (Exception ignored) {}
            return true;
        }
        return false;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        for (LLMProvider provider : providers.values()) {
            try { provider.close(); } catch (Exception ignored) {}
        }
        providers.clear();
        System.out.println("[ProviderFactory] Closed: providers=" + providers.size());
    }

    public boolean isClosed() { return closed; }

    /**
     * Get provider statistics.
     */
    public ProviderStats getStats() {
        Map<String, ProviderInfo> info = new ConcurrentHashMap<>();
        for (Map.Entry<String, LLMProvider> e : providers.entrySet()) {
            LLMProvider p = e.getValue();
            info.put(e.getKey(), new ProviderInfo(
                    p.name(),
                    p.supportsRLHF(),
                    p.supportsFunctionCalling(),
                    p.maxContextLength(),
                    p.supportedModalities()
            ));
        }
        return new ProviderStats(providers.size(), info);
    }

    /**
     * Provider statistics.
     */
    public static final class ProviderStats {
        public final int count;
        public final Map<String, ProviderInfo> providers;

        public ProviderStats(int count, Map<String, ProviderInfo> providers) {
            this.count = count;
            this.providers = providers;
        }
    }

    /**
     * Provider information.
     */
    public static final class ProviderInfo {
        public final String name;
        public final boolean supportsRLHF;
        public final boolean supportsFunctionCalling;
        public final int maxContextLength;
        public final String[] modalities;

        public ProviderInfo(String name, boolean supportsRLHF, boolean supportsFunctionCalling,
                         int maxContextLength, String[] modalities) {
            this.name = name;
            this.supportsRLHF = supportsRLHF;
            this.supportsFunctionCalling = supportsFunctionCalling;
            this.maxContextLength = maxContextLength;
            this.modalities = modalities;
        }
    }
}
