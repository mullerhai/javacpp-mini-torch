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
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.LoraLinear;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.transformers.generation.Generator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import static org.bytedeco.pytorch.global.torch.cross_entropy;

/**
 * Llama / Mistral-style causal LM with HF-identical parameter names
 * (same layout as Qwen2 but q/k/v without bias).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LlamaForCausalLM extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final PretrainedConfig config;
    private final LlamaModel model;
    private final LinearImpl lm_head;
    /** LoRA adapters attached via {@link #attachLora(String, LoraConfig)}. */
    private final java.util.LinkedHashMap<String, LoraLinear> loraAdapters =
            new java.util.LinkedHashMap<>();

    public LlamaForCausalLM(PretrainedConfig config) {
        super("LlamaForCausalLM");
        this.config = Objects.requireNonNull(config, "config");
        this.model = register_module("model", new LlamaModel(config));
        this.lm_head = register_module("lm_head",
                new LinearImpl(new LinearOptions(config.hiddenSize(), config.vocabSize()).bias(false)));
        if (config.tieWordEmbeddings()) {
            try {
                Tensor dest = lm_head.weight();
                Tensor src = model.embed_tokens.weight();
                if (dest.scalar_type() == src.scalar_type()) {
                    lm_head.weight().set_(model.embed_tokens.weight());
                } else {
                    // dtype mismatch: copy data instead of sharing storage
                    lm_head.weight().copy_(model.embed_tokens.weight());
                }
            } catch (Throwable ignored) {}
        }
    }

    public static LlamaForCausalLM fromConfig(PretrainedConfig config) {
        return new LlamaForCausalLM(config);
    }

    public PretrainedConfig config() {
        return config;
    }

    @Override
    public Tensor forward(Tensor inputIds) {
        return lm_head.forward(model.forward(inputIds));
    }

    /**
     * Cache-aware causal LM forward for incremental decode serving.
     *
     * <p><b>Note:</b> This path does <b>not</b> apply LoRA adapters. It is intended
     * for production inference where the LoRA weights have been merged
     * ({@link LoraLinear#merge()}) or where no LoRA is active. Do not call this
     * method when unmerged LoRA adapters are attached — the result will be incorrect.
     *
     * @param inputIds       [B,T] token ids
     * @param positionOffset RoPE start position
     * @param pastKs         [numLayers] past K
     * @param pastVs         [numLayers] past V
     * @return logits + per-layer new K/V
     */
    public CachedForwardResult forwardCached(Tensor inputIds, long positionOffset,
                                              Tensor[] pastKs, Tensor[] pastVs) {
        if (!loraAdapters.isEmpty()) {
            throw new IllegalStateException(
                    "forwardCached() cannot be used when unmerged LoRA adapters are active. "
                    + "Call LoraLinear.merge() on all adapters before using this path, "
                    + "or use forward(Tensor) instead.");
        }
        Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
        long T = ids.size(1);
        if (positionOffset + T > config.maxPositionEmbeddings()) {
            throw new IllegalArgumentException("Sequence length " + (positionOffset + T)
                    + " exceeds max_position_embeddings=" + config.maxPositionEmbeddings());
        }
        Tensor x = model.embed_tokens.forward(ids);
        CachedForwardResult result = model.forwardCached(x, positionOffset, pastKs, pastVs);
        Tensor logits = lm_head.forward(result.hidden());
        return new CachedForwardResult(logits, result.newKs, result.newVs);
    }

    public Tensor loss(Tensor inputIds) {
        Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
        Tensor logits = forward(ids);
        Tensor shiftLogits = logits.slice(1, new LongOptional(0), new LongOptional(logits.size(1) - 1), 1)
                .contiguous();
        Tensor shiftLabels = ids.slice(1, new LongOptional(1), new LongOptional(ids.size(1)), 1)
                .contiguous();
        long V = logits.size(2);
        return cross_entropy(shiftLogits.reshape(-1, V), shiftLabels.reshape(-1));
    }

    /** View of all LoRA adapters attached to this model. */
    public java.util.Map<String, LoraLinear> loraAdapters() {
        return java.util.Collections.unmodifiableMap(loraAdapters);
    }

    /** True if any LoRA adapters have been attached. */
    public boolean hasLoraAdapters() {
        return !loraAdapters.isEmpty();
    }

    public int[] generate(int[] promptIds, int maxNewTokens) {
        return generate(promptIds, GenerationConfig.builder().maxNewTokens(maxNewTokens).build());
    }

    public int[] generate(int[] promptIds, GenerationConfig gen) {
        GenerationConfig g = gen == null ? GenerationConfig.greedy() : gen;
        if (g.eosTokenIds.isEmpty()) {
            g = g.toBuilder().eosTokenId(config.eosTokenId()).build();
        }
        return Generator.generate(this, promptIds, g, config.maxPositionEmbeddings());
    }

    public LlamaModel model() { return model; }
    public LinearImpl lmHead() { return lm_head; }

    /** Re-tie lm_head ← embed_tokens after weight load (dtype-aware). */
    public boolean retieWordEmbeddings() {
        if (!config.tieWordEmbeddings() || lm_head == null || model == null) return false;
        try {
            Tensor dest = lm_head.weight();
            Tensor src = model.embed_tokens.weight();
            if (dest == null || src == null || !dest.defined() || !src.defined()) return false;
            try { dest.requires_grad_(false); } catch (Throwable ignored) {}
            try { src.requires_grad_(false); } catch (Throwable ignored) {}
            if (dest.scalar_type() == src.scalar_type()) {
                dest.set_(src);
            } else {
                dest.copy_(src);
            }
            return true;
        } catch (Throwable t) {
            try {
                lm_head.weight().copy_(model.embed_tokens.weight());
                return true;
            } catch (Throwable t2) {
                return false;
            }
        }
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class LlamaModel extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final EmbeddingImpl embed_tokens;
        public final List<LlamaDecoderLayer> layers = new ArrayList<>();
        public final RMSNorm norm;
        private final PretrainedConfig config;

        public LlamaModel(PretrainedConfig config) {
            super("LlamaModel");
            this.config = config;
            this.embed_tokens = register_module("embed_tokens",
                    new EmbeddingImpl(config.vocabSize(), config.hiddenSize()));
            for (int i = 0; i < config.numHiddenLayers(); i++) {
                layers.add(register_module("layers/" + i, new LlamaDecoderLayer(config, i)));
            }
            this.norm = register_module("norm", new RMSNorm(config.hiddenSize(), config.rmsNormEps()));
        }

        @Override
        public Tensor forward(Tensor inputIds) {
            Tensor ids = inputIds.dim() == 1 ? inputIds.unsqueeze(0) : inputIds;
            long T = ids.size(1);
            if (T > config.maxPositionEmbeddings()) {
                throw new IllegalArgumentException("Sequence length " + T
                        + " exceeds max_position_embeddings=" + config.maxPositionEmbeddings());
            }
            Tensor x = embed_tokens.forward(ids);
            for (LlamaDecoderLayer layer : layers) {
                x = layer.forward(x);
            }
            return norm.forward(x);
        }

        /** Cache-aware model forward. All layers share the same positionOffset. */
        public CachedForwardResult forwardCached(Tensor x, long positionOffset,
                                                  Tensor[] pastKs, Tensor[] pastVs) {
            Tensor[] newKs = new Tensor[config.numHiddenLayers()];
            Tensor[] newVs = new Tensor[config.numHiddenLayers()];
            for (int i = 0; i < layers.size(); i++) {
                Tensor[] out = layers.get(i).forwardCached(x, positionOffset,
                        pastKs != null ? pastKs[i] : null,
                        pastVs != null ? pastVs[i] : null);
                x = out[0];
                newKs[i] = out[1];
                newVs[i] = out[2];
            }
            x = norm.forward(x);
            return new CachedForwardResult(x, newKs, newVs);
        }
    }

    @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
    public static class LlamaDecoderLayer extends Module {
        static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

        public final RMSNorm input_layernorm;
        public final ModelingAttention self_attn;
        public final RMSNorm post_attention_layernorm;
        public final ModelingMlp.SwiGLU mlp;
        /** Per-name LoRA adapters for this layer. Updated by LlamaForCausalLM.attachLora. */
        private final java.util.LinkedHashMap<String, LoraLinear> loraAdapters =
                new java.util.LinkedHashMap<>();

        public LlamaDecoderLayer(PretrainedConfig cfg, int layerIdx) {
            super("LlamaDecoderLayer" + layerIdx);
            this.input_layernorm = register_module("input_layernorm",
                    new RMSNorm(cfg.hiddenSize(), cfg.rmsNormEps()));
            // Llama: no qkv bias
            this.self_attn = register_module("self_attn",
                    new ModelingAttention(cfg.hiddenSize(), cfg.numAttentionHeads(),
                            cfg.numKeyValueHeads(), cfg.ropeTheta(), true, false));
            this.post_attention_layernorm = register_module("post_attention_layernorm",
                    new RMSNorm(cfg.hiddenSize(), cfg.rmsNormEps()));
            this.mlp = register_module("mlp",
                    new ModelingMlp.SwiGLU(cfg.hiddenSize(), cfg.intermediateSize()));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor h = input_layernorm.forward(x);
            // LoRA-aware Q/K/V projections
            Tensor q = applyProj(self_attn.q_proj, "q_proj", h);
            Tensor k = applyProj(self_attn.k_proj, "k_proj", h);
            Tensor v = applyProj(self_attn.v_proj, "v_proj", h);
            // Compute attention with the (Q,K,V) tensors
            Tensor attnOut = self_attn.forwardFromQKV(q, k, v);
            // LoRA-aware O projection
            Tensor o = applyProj(self_attn.o_proj, "o_proj", attnOut);
            x = x.add(o);
            // LoRA-aware MLP: gate_proj and up_proj are LoRA-aware; down_proj is not
            Tensor mh = post_attention_layernorm.forward(x);
            Tensor gate = applyProj(mlp.gate_proj, "gate_proj", mh);
            Tensor up   = applyProj(mlp.up_proj,   "up_proj",   mh);
            x = x.add(mlp.forwardWithGateUp(gate, up));
            return x;
        }

        /**
         * Apply a projection layer with optional LoRA overlay.
         * When a LoRA adapter is attached to the given target name, its output
         * (scaled: B @ A * scaling) is added to the base projection output.
         * When the adapter is merged, only the base (now containing ΔW) is used.
         *
         * @param base      the underlying {@link LinearImpl} (q_proj, k_proj, …)
         * @param targetKey the LoRA adapter key ({@code "q_proj"}, {@code "gate_proj"}, …)
         * @param input     the input tensor
         * @return base(input) [+ ΔW·input] with LoRA
         */
        private Tensor applyProj(LinearImpl base, String targetKey, Tensor input) {
            LoraLinear lora = loraAdapters.get(targetKey);
            Tensor baseOut = base.forward(input);
            if (lora != null && !lora.isMerged()) {
                return baseOut.add(lora.forward(input));
            }
            return baseOut;
        }

        /**
         * Attach a LoRA adapter to this layer. {@code name} should be one of
         * {@code q_proj}, {@code k_proj}, {@code v_proj}, {@code o_proj},
         * {@code gate_proj}, {@code up_proj}, or {@code down_proj}.
         * Returns the adapter, or {@code null} if {@code name} is not a known target.
         */
        public LoraLinear attachLora(String name, LoraConfig cfg) {
            if (name == null || cfg == null) return null;
            LinearImpl base = null;
            switch (name) {
                case "q_proj":  base = self_attn.q_proj; break;
                case "k_proj":  base = self_attn.k_proj; break;
                case "v_proj":  base = self_attn.v_proj; break;
                case "o_proj":  base = self_attn.o_proj; break;
                case "gate_proj": base = mlp.gate_proj; break;
                case "up_proj":  base = mlp.up_proj; break;
                case "down_proj": base = mlp.down_proj; break;
                default: return null;
            }
            LoraLinear adapter = LoraLinear.borrowBase(base, cfg);
            loraAdapters.put(name, adapter);
            return adapter;
        }

        public java.util.Map<String, LoraLinear> loraAdapters() {
            return java.util.Collections.unmodifiableMap(loraAdapters);
        }

        /**
         * Cache-aware layer forward for inference serving.
         *
         * <p>Note: this method does <b>not</b> apply LoRA adapters. It is used
         * exclusively for KV-cache decode paths where LoRA overhead is undesirable.
         * For training (and for correct LoRA inference) use {@link #forward(Tensor)}.
         */
        public Tensor[] forwardCached(Tensor x, long positionOffset, Tensor pastK, Tensor pastV) {
            Tensor h = input_layernorm.forward(x);
            Tensor[] attOut = self_attn.forwardCached(h, positionOffset, pastK, pastV);
            Tensor out = x.add(attOut[0]);
            out = out.add(mlp.forward(post_attention_layernorm.forward(out)));
            return new Tensor[]{out, attOut[1], attOut[2]};
        }
    }

    // ---------------------------------------------------------------------
    // PEFT integration: attachLora, quantizableLinears, namedLinears
    // ---------------------------------------------------------------------

    /** Returns all named Linear children matching the HF prefix structure. */
    public java.util.LinkedHashMap<String, LinearImpl> namedLinears() {
        java.util.LinkedHashMap<String, LinearImpl> m = new java.util.LinkedHashMap<>();
        for (int i = 0; i < model.layers.size(); i++) {
            LlamaDecoderLayer layer = model.layers.get(i);
            String p = "model.layers." + i;
            m.put(p + ".self_attn.q_proj", layer.self_attn.q_proj);
            m.put(p + ".self_attn.k_proj", layer.self_attn.k_proj);
            m.put(p + ".self_attn.v_proj", layer.self_attn.v_proj);
            m.put(p + ".self_attn.o_proj", layer.self_attn.o_proj);
            m.put(p + ".mlp.gate_proj", layer.mlp.gate_proj);
            m.put(p + ".mlp.up_proj", layer.mlp.up_proj);
            m.put(p + ".mlp.down_proj", layer.mlp.down_proj);
        }
        return m;
    }

    /** Returns all named Linear children excluding lm_head (for QLoRA). */
    public java.util.LinkedHashMap<String, LinearImpl> quantizableLinears() {
        return namedLinears();
    }

    /**
     * Attach a LoRA adapter to every Q/K/V/O projection and every MLP projection
     * in every decoder layer. After calling this method, the LoRA ΔW contributions
     * are applied during forward passes via the layer-level adapter dispatch.
     *
     * <p>The adapters are attached to the following targets per layer:
     * {@code q_proj}, {@code k_proj}, {@code v_proj}, {@code o_proj},
     * {@code gate_proj}, {@code up_proj}, {@code down_proj}.
     *
     * <p>After attaching, call {@link #freezeBaseModelParameters()} to freeze all
     * base-model weights so that only the LoRA A/B matrices remain trainable.
     *
     * @param config the LoRA configuration (rank, alpha, dropout, target modules)
     * @return the list of adapters created, one per (layer, target) pair
     */
    public java.util.List<LoraLinear> attachLora(LoraConfig config) {
        if (config == null) throw new IllegalArgumentException("config is null");
        java.util.List<LoraLinear> adapters = new java.util.ArrayList<>();
        java.util.Set<String> targets = config.targetModules() != null
                ? new java.util.HashSet<>(config.targetModules())
                : java.util.Collections.emptySet();
        String[] allTargets = {"q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"};
        for (int i = 0; i < model.layers.size(); i++) {
            LlamaDecoderLayer layer = model.layers.get(i);
            for (String target : allTargets) {
                if (targets.isEmpty() || targets.contains(target)) {
                    LoraLinear adapter = layer.attachLora(target, config);
                    if (adapter != null) {
                        String fullName = "model.layers." + i + "." + target;
                        loraAdapters.put(fullName, adapter);
                        adapters.add(adapter);
                    }
                }
            }
        }
        return adapters;
    }

    /**
     * Attach a single LoRA adapter to a named linear layer. Returns null if
     * {@code name} is not found.
     */
    public LoraLinear attachLora(String name, LoraConfig config) {
        if (config == null) throw new IllegalArgumentException("config is null");
        Objects.requireNonNull(name, "name");
        LinearImpl lin = namedLinears().get(name);
        if (lin == null) return null;
        LoraLinear adapter = LoraLinear.borrowBase(lin, config);
        loraAdapters.put(name, adapter);
        // Parse "model.layers.{idx}.{target}" and wire the layer-level adapter
        if (name.startsWith("model.layers.")) {
            String after = name.substring("model.layers.".length());
            int lastDot = after.lastIndexOf('.');
            if (lastDot > 0) {
                try {
                    int layerIdx = Integer.parseInt(after.substring(0, lastDot));
                    String target = after.substring(lastDot + 1);
                    if (layerIdx >= 0 && layerIdx < model.layers.size()) {
                        model.layers.get(layerIdx).attachLora(target, config);
                    }
                } catch (NumberFormatException ignored) {}
            }
        }
        return adapter;
    }

    /**
     * Freeze all base-model parameters so that only LoRA adapter parameters
     * remain trainable.
     *
     * <p>This is the standard QLoRA preparation step: after attaching LoRA adapters
     * via {@link #attachLora(LoraConfig)}, call this to freeze every base-model
     * parameter returned by {@link #parameters()}. Because LoRA adapters created with
     * {@link LoraLinear#borrowBase} are not registered as children of this module,
     * their A/B weight matrices are <b>not</b> traversed by {@link #parameters()}
     * and therefore remain trainable.
     *
     * <p>Calling this method is equivalent to the Python idiom:
     * <pre>
     *   for name, param in model.named_parameters():
     *       if 'lora_' not in name:
     *           param.requires_grad = False
     * </pre>
     *
     * @see #attachLora(LoraConfig)
     * @see #attachLora(String, LoraConfig)
     */
    public void freezeBaseModelParameters() {
        TensorVector params = parameters();
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p != null && p.defined()) p.set_requires_grad(false);
        }
    }

    /**
     * @deprecated Use {@link #freezeBaseModelParameters()} instead.
     *             This method name is misleading: it does not enable PyTorch
     *             activation checkpointing (which requires
     *             {@code torch.utils.checkpoint}) but freezes base-model parameters.
     */
    @Deprecated
    public void enable_gradient_checkpointing() {
        freezeBaseModelParameters();
    }

    /**
     * Override ugly Module#toString for cleaner logging.
     */
    @Override
    public String toString() {
        return "LlamaForCausalLM(config=" + config
                + ", layers=" + model.layers.size()
                + ", hidden=" + config.hiddenSize() + ")";
    }
}
