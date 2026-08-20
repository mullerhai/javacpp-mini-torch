/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option) any later version (collectively, the "License");
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
package org.bytedeco.pytorch.llm.transformers.configuration;

import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import java.util.Map;

/**
 * HuggingFace <code>deepseek_v3Config</code>.
 * Reference: transformers/models/deepseek_v3/configuration_deepseek_v3.py
 */
public final class DeepSeekV3Config extends Config {

    public static final String MODEL_TYPE = "deepseek_v3";

    private final int moeIntermediateSize;
    private final int nSharedExperts;
    private final int nRoutedExperts;
    private final double routedScalingFactor;
    private final int kvLoraRank;
    private final int qLoraRank;
    private final int qkRopeHeadDim;
    private final int vHeadDim;
    private final int qkNopeHeadDim;
    private final int nGroup;
    private final int topkGroup;
    private final int numExpertsPerTok;
    private final int firstKDenseReplace;
    private final boolean normTopkProb;
    private final boolean ropeInterleave;
    private final boolean attentionBias;

    public DeepSeekV3Config(PretrainedConfig base) {
        super(base);
        this.moeIntermediateSize = toInt(base.extra().get("moe_intermediate_size"), 2048);
        this.nSharedExperts = toInt(base.extra().get("n_shared_experts"), 1);
        this.nRoutedExperts = toInt(base.extra().get("n_routed_experts"), 256);
        this.routedScalingFactor = toDouble(base.extra().get("routed_scaling_factor"), 2.5);
        this.kvLoraRank = toInt(base.extra().get("kv_lora_rank"), 512);
        this.qLoraRank = toInt(base.extra().get("q_lora_rank"), 1536);
        this.qkRopeHeadDim = toInt(base.extra().get("qk_rope_head_dim"), 64);
        this.vHeadDim = toInt(base.extra().get("v_head_dim"), 128);
        this.qkNopeHeadDim = toInt(base.extra().get("qk_nope_head_dim"), 128);
        this.nGroup = toInt(base.extra().get("n_group"), 8);
        this.topkGroup = toInt(base.extra().get("topk_group"), 4);
        this.numExpertsPerTok = toInt(base.extra().get("num_experts_per_tok"), 8);
        this.firstKDenseReplace = toInt(base.extra().get("first_k_dense_replace"), 3);
        this.normTopkProb = base.extra().get("norm_topk_prob") == Boolean.TRUE;
        this.ropeInterleave = base.extra().get("rope_interleave") == Boolean.TRUE;
        this.attentionBias = base.extra().get("attention_bias") == Boolean.TRUE;
    }

    public int moeIntermediateSize() { return toInt(base().extra().get("moe_intermediate_size"), 2048); }
    public int nSharedExperts() { return toInt(base().extra().get("n_shared_experts"), 1); }
    public int nRoutedExperts() { return toInt(base().extra().get("n_routed_experts"), 256); }
    public double routedScalingFactor() { return toDouble(base().extra().get("routed_scaling_factor"), 2.5); }
    public int kvLoraRank() { return toInt(base().extra().get("kv_lora_rank"), 512); }
    public int qLoraRank() { return toInt(base().extra().get("q_lora_rank"), 1536); }
    public int qkRopeHeadDim() { return toInt(base().extra().get("qk_rope_head_dim"), 64); }
    public int vHeadDim() { return toInt(base().extra().get("v_head_dim"), 128); }
    public int qkNopeHeadDim() { return toInt(base().extra().get("qk_nope_head_dim"), 128); }
    public int nGroup() { return toInt(base().extra().get("n_group"), 8); }
    public int topkGroup() { return toInt(base().extra().get("topk_group"), 4); }
    public int numExpertsPerTok() { return toInt(base().extra().get("num_experts_per_tok"), 8); }
    public int firstKDenseReplace() { return toInt(base().extra().get("first_k_dense_replace"), 3); }
    public boolean normTopkProb() { return base().extra().get("norm_topk_prob") == Boolean.TRUE; }
    public boolean ropeInterleave() { return base().extra().get("rope_interleave") == Boolean.TRUE; }
    public boolean attentionBias() { return base().extra().get("attention_bias") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return DeepSeekV3Config.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}