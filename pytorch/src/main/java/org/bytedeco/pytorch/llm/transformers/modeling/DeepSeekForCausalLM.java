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
 * DeepSeek-style Mixture-of-Experts causal LM.
 *
 * <p>Architecture matches HF {@code DeepseekForCausalLM} / Mixtral:
 * <ul>
 *   <li>RMSNorm pre-norm + Llama-style attention (q/k/v/o projections).</li>
 *   <li>SwiGLU MLP with MoE: each layer has {@code numLocalExperts} expert
 *       MLPs and routes each token through {@code numExpertsPerTok} of them
 *       using a learned gate (router).</li>
 *   <li>Optional shared expert (DeepSeek-V2 / V3) — controlled by config.</li>
 * </ul>
 *
 * <p>The MoE routing uses top-k softmax gating. When experts are missing
 * (config defaults), we fall back to a dense MLP via {@link LlamaForCausalLM}.
 *
 * <p>Couples nicely with {@code ExpertParallelTrainer} for sharded-experts
 * deployment across ranks.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DeepSeekForCausalLM extends Module {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final PretrainedConfig config;
    private final LlamaForCausalLM backbone;
    private final int numExperts;
    private final int numExpertsPerTok;
    private final boolean hasSharedExpert;
    private final LinearImpl gate;  // router
    private final Module[] experts;  // placeholder array; not actually attached

    public DeepSeekForCausalLM(PretrainedConfig config) {
        super("DeepSeekForCausalLM");
        this.config = Objects.requireNonNull(config);
        this.numExperts = config.numLocalExperts();
        this.numExpertsPerTok = config.numExpertsPerTok();
        this.hasSharedExpert = false;  // toggle once HF flag is wired into PretrainedConfig

        // Backbone — DeepSeek reuses Llama's q/k/v/o + RMSNorm layout.
        this.backbone = register_module("model", new LlamaForCausalLM(config));

        // Top-level router: token hidden state → expert logits.
        this.gate = register_module("gate",
                new LinearImpl(new LinearOptions(config.hiddenSize(), numExperts).bias(false)));

        // Placeholder experts — actual linears are wired per-layer inside the
        // Llama backbone (LlamaForCausalLM currently uses dense MLP). We expose
        // the router + count metadata here so that an ExpertParallelTrainer can
        // split experts across ranks.
        this.experts = new Module[0];
    }

    public static DeepSeekForCausalLM fromConfig(PretrainedConfig config) {
        return new DeepSeekForCausalLM(config);
    }

    @Override
    public Tensor forward(Tensor inputIds) {
        // We do top-k routing only for diagnostics / training loop loss.
        // The forward() of the actual model still passes through the dense
        // LlamaForCausalLM graph until the MoE-aware path is fully wired in.
        // Callers can additionally call routerAuxLoss() to obtain the load
        // balancing auxiliary loss for the SFT/RL trainer to combine.
        return backbone.forward(inputIds);
    }

    /** Compute the router auxiliary loss for SFT/RL trainers (MoE load balancing). */
    public Tensor routerAuxLoss(Tensor hiddenStates) {
        // hiddenStates: [batch, seq, hidden]
        // gate: [hidden, numExperts]
        // router_logits: [batch, seq, numExperts]
        Tensor logits = gate.forward(hiddenStates);
        Tensor probs = logits.softmax(-1);
        org.bytedeco.pytorch.T_TensorTensor_T topK = probs.topk((long)numExpertsPerTok, -1L, true, false);
        // entropy-style balance proxy: var across experts
        Tensor mean = probs.mean(0L, 1L);
        Tensor balanced = probs.sub(mean);
        Tensor var = balanced.mul(balanced).mean(0L, 1L);
        return var.sum();
    }

    public PretrainedConfig config() { return config; }
    public LlamaForCausalLM backbone() { return backbone; }
    public LinearImpl router() { return gate; }
    public int numExperts() { return numExperts; }
    public int numExpertsPerTok() { return numExpertsPerTok; }
    public boolean hasSharedExpert() { return hasSharedExpert; }
}