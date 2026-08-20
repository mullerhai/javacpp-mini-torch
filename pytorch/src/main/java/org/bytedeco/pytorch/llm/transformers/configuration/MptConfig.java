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
 * HuggingFace <code>mptConfig</code>.
 * Reference: transformers/models/mpt/configuration_mpt.py
 */
public final class MptConfig extends Config {

    public static final String MODEL_TYPE = "mpt";

    private final String attnType;
    private final double attnPdrop;
    private final String attnImpl;
    private final double clipQkv;
    private final double softmaxScale;
    private final boolean prefixLm;
    private final boolean qkLn;
    private final boolean attnUsesSequenceId;
    private final boolean alibi;
    private final double alibiBiasMax;
    private final int dModel;
    private final int nHeads;
    private final int nLayers;
    private final int expansionRatio;
    private final int maxSeqLen;
    private final double residPdrop;
    private final double embPdrop;
    private final boolean learnedPosEmb;
    private final double logitScale;
    private final boolean noBias;
    private final double embeddingFraction;
    private final String normType;
    private final boolean useCache;

    public MptConfig(PretrainedConfig base) {
        super(base);
        this.attnType = String.valueOf(base.extra().get("attn_type"));
        this.attnPdrop = toDouble(base.extra().get("attn_pdrop"), 0.0);
        this.attnImpl = String.valueOf(base.extra().get("attn_impl"));
        this.clipQkv = toDouble(base.extra().get("clip_qkv"), 0.0);
        this.softmaxScale = toDouble(base.extra().get("softmax_scale"), 0.0);
        this.prefixLm = base.extra().get("prefix_lm") == Boolean.TRUE;
        this.qkLn = base.extra().get("qk_ln") == Boolean.TRUE;
        this.attnUsesSequenceId = base.extra().get("attn_uses_sequence_id") == Boolean.TRUE;
        this.alibi = base.extra().get("alibi") == Boolean.TRUE;
        this.alibiBiasMax = toDouble(base.extra().get("alibi_bias_max"), 8.0);
        this.dModel = toInt(base.extra().get("d_model"), 2048);
        this.nHeads = toInt(base.extra().get("n_heads"), 16);
        this.nLayers = toInt(base.extra().get("n_layers"), 24);
        this.expansionRatio = toInt(base.extra().get("expansion_ratio"), 4);
        this.maxSeqLen = toInt(base.extra().get("max_seq_len"), 2048);
        this.residPdrop = toDouble(base.extra().get("resid_pdrop"), 0.0);
        this.embPdrop = toDouble(base.extra().get("emb_pdrop"), 0.0);
        this.learnedPosEmb = base.extra().get("learned_pos_emb") == Boolean.TRUE;
        this.logitScale = toDouble(base.extra().get("logit_scale"), 0.0);
        this.noBias = base.extra().get("no_bias") == Boolean.TRUE;
        this.embeddingFraction = toDouble(base.extra().get("embedding_fraction"), 1.0);
        this.normType = String.valueOf(base.extra().get("norm_type"));
        this.useCache = base.extra().get("use_cache") == Boolean.TRUE;
    }

    public String attnType() { Object v = base().extra().get("attn_type"); return v == null ? "multihead_attention" : String.valueOf(v); }
    public double attnPdrop() { return toDouble(base().extra().get("attn_pdrop"), 0.0); }
    public String attnImpl() { Object v = base().extra().get("attn_impl"); return v == null ? "torch" : String.valueOf(v); }
    public double clipQkv() { return toDouble(base().extra().get("clip_qkv"), 0.0); }
    public double softmaxScale() { return toDouble(base().extra().get("softmax_scale"), 0.0); }
    public boolean prefixLm() { return base().extra().get("prefix_lm") == Boolean.TRUE; }
    public boolean qkLn() { return base().extra().get("qk_ln") == Boolean.TRUE; }
    public boolean attnUsesSequenceId() { return base().extra().get("attn_uses_sequence_id") == Boolean.TRUE; }
    public boolean alibi() { return base().extra().get("alibi") == Boolean.TRUE; }
    public double alibiBiasMax() { return toDouble(base().extra().get("alibi_bias_max"), 8.0); }
    public int dModel() { return toInt(base().extra().get("d_model"), 2048); }
    public int nHeads() { return toInt(base().extra().get("n_heads"), 16); }
    public int nLayers() { return toInt(base().extra().get("n_layers"), 24); }
    public int expansionRatio() { return toInt(base().extra().get("expansion_ratio"), 4); }
    public int maxSeqLen() { return toInt(base().extra().get("max_seq_len"), 2048); }
    public double residPdrop() { return toDouble(base().extra().get("resid_pdrop"), 0.0); }
    public double embPdrop() { return toDouble(base().extra().get("emb_pdrop"), 0.0); }
    public boolean learnedPosEmb() { return base().extra().get("learned_pos_emb") == Boolean.TRUE; }
    public double logitScale() { return toDouble(base().extra().get("logit_scale"), 0.0); }
    public boolean noBias() { return base().extra().get("no_bias") == Boolean.TRUE; }
    public double embeddingFraction() { return toDouble(base().extra().get("embedding_fraction"), 1.0); }
    public String normType() { Object v = base().extra().get("norm_type"); return v == null ? "low_precision_layernorm" : String.valueOf(v); }
    public boolean useCache() { return base().extra().get("use_cache") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return MptConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}