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
 * HuggingFace <code>dbrxConfig</code>.
 * Reference: transformers/models/dbrx/configuration_dbrx.py
 */
public final class DbrxConfig extends Config {

    public static final String MODEL_TYPE = "dbrx";

    private final double attnPdrop;
    private final double clipQkv;
    private final int kvNHeads;
    private final int ffnHiddenSize;
    private final int moeNumExperts;
    private final int moeTopK;
    private final double moeJitterEps;
    private final double moeLossWeight;
    private final double moeNormalizeExpertWeights;
    private final double residPdrop;
    private final double embPdrop;
    private final boolean outputRouterLogits;
    private final int dModel;
    private final int nHeads;
    private final int nLayers;
    private final int maxSeqLen;

    public DbrxConfig(PretrainedConfig base) {
        super(base);
        this.attnPdrop = toDouble(base.extra().get("attn_pdrop"), 0.0);
        this.clipQkv = toDouble(base.extra().get("clip_qkv"), 0.0);
        this.kvNHeads = toInt(base.extra().get("kv_n_heads"), 1);
        this.ffnHiddenSize = toInt(base.extra().get("ffn_hidden_size"), 3584);
        this.moeNumExperts = toInt(base.extra().get("moe_num_experts"), 4);
        this.moeTopK = toInt(base.extra().get("moe_top_k"), 1);
        this.moeJitterEps = toDouble(base.extra().get("moe_jitter_eps"), 0.0);
        this.moeLossWeight = toDouble(base.extra().get("moe_loss_weight"), 0.01);
        this.moeNormalizeExpertWeights = toDouble(base.extra().get("moe_normalize_expert_weights"), 1.0);
        this.residPdrop = toDouble(base.extra().get("resid_pdrop"), 0.0);
        this.embPdrop = toDouble(base.extra().get("emb_pdrop"), 0.0);
        this.outputRouterLogits = base.extra().get("output_router_logits") == Boolean.TRUE;
        this.dModel = toInt(base.extra().get("d_model"), 2048);
        this.nHeads = toInt(base.extra().get("n_heads"), 16);
        this.nLayers = toInt(base.extra().get("n_layers"), 24);
        this.maxSeqLen = toInt(base.extra().get("max_seq_len"), 2048);
    }

    public double attnPdrop() { return toDouble(base().extra().get("attn_pdrop"), 0.0); }
    public double clipQkv() { return toDouble(base().extra().get("clip_qkv"), 0.0); }
    public int kvNHeads() { return toInt(base().extra().get("kv_n_heads"), 1); }
    public int ffnHiddenSize() { return toInt(base().extra().get("ffn_hidden_size"), 3584); }
    public int moeNumExperts() { return toInt(base().extra().get("moe_num_experts"), 4); }
    public int moeTopK() { return toInt(base().extra().get("moe_top_k"), 1); }
    public double moeJitterEps() { return toDouble(base().extra().get("moe_jitter_eps"), 0.0); }
    public double moeLossWeight() { return toDouble(base().extra().get("moe_loss_weight"), 0.01); }
    public double moeNormalizeExpertWeights() { return toDouble(base().extra().get("moe_normalize_expert_weights"), 1.0); }
    public double residPdrop() { return toDouble(base().extra().get("resid_pdrop"), 0.0); }
    public double embPdrop() { return toDouble(base().extra().get("emb_pdrop"), 0.0); }
    public boolean outputRouterLogits() { return base().extra().get("output_router_logits") == Boolean.TRUE; }
    public int dModel() { return toInt(base().extra().get("d_model"), 2048); }
    public int nHeads() { return toInt(base().extra().get("n_heads"), 16); }
    public int nLayers() { return toInt(base().extra().get("n_layers"), 24); }
    public int maxSeqLen() { return toInt(base().extra().get("max_seq_len"), 2048); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return DbrxConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}