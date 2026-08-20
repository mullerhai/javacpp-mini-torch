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
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.transformers.modeling;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import java.util.Objects;

/**
 * Mistral-style causal LM with explicit sliding-window attention and grouped
 * query attention (GQA) parameter layout.
 *
 * <p>Architecture matches HF {@code MistralForCausalLM}:
 * <ul>
 *   <li>Separate q_proj / k_proj / v_proj / o_proj (no qkv fusion, no biases).</li>
 *   <li>SwiGLU MLP: gate_proj / up_proj / down_proj.</li>
 *   <li>RMSNorm pre-norm.</li>
 *   <li>RoPE with theta from config.</li>
 *   <li>{@code sliding_window} attention pattern (config-driven; the
 *       {@link org.bytedeco.pytorch.llm.transformers.generation.cache.SlidingWindowCache}
 *       is the recommended KV cache).</li>
 * </ul>
 *
 * <p>The forward uses the underlying {@link LlamaForCausalLM} graph when
 * sliding window equals full attention (HF does the same — Mistral falls back
 * to full attention for small contexts). When sliding-window is required,
 * the user pairs it with a {@code SlidingWindowCache}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MistralForCausalLM extends Module {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final PretrainedConfig config;
    private final LlamaForCausalLM backbone;
    private final int slidingWindow;
    private final boolean usesGqa;

    public MistralForCausalLM(PretrainedConfig config) {
        super("MistralForCausalLM");
        this.config = Objects.requireNonNull(config);
        this.slidingWindow = config.slidingWindow() > 0 ? config.slidingWindow() : config.maxPositionEmbeddings();
        this.usesGqa = config.numKeyValueHeads() != config.numAttentionHeads();
        // Mistral's parameter names match Llama's exactly — reuse the graph.
        this.backbone = register_module("model", new LlamaForCausalLM(config));
    }

    public static MistralForCausalLM fromConfig(PretrainedConfig config) {
        return new MistralForCausalLM(config);
    }

    @Override
    public Tensor forward(Tensor inputIds) {
        return backbone.forward(inputIds);
    }

    public PretrainedConfig config() { return config; }
    public LlamaForCausalLM backbone() { return backbone; }
    public int slidingWindow() { return slidingWindow; }
    public boolean usesGqa() { return usesGqa; }

    /** Delegate LoRA weld to the Llama backbone (identical HF parameter names). */
    public java.util.List<org.bytedeco.pytorch.llm.peft.LoraLinear> attachLora(
            org.bytedeco.pytorch.llm.peft.LoraConfig config) {
        return backbone.attachLora(config);
    }

    public java.util.Map<String, org.bytedeco.pytorch.llm.peft.LoraLinear> loraAdapters() {
        return backbone.loraAdapters();
    }

    public void freezeBaseModelParameters() {
        backbone.freezeBaseModelParameters();
    }
}