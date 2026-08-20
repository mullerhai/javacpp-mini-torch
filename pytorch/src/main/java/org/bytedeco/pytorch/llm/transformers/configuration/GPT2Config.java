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
 * HuggingFace <code>gpt2Config</code>.
 * Reference: transformers/models/gpt2/configuration_gpt2.py
 */
public final class GPT2Config extends Config {

    public static final String MODEL_TYPE = "gpt2";

    private final int nPositions;
    private final int nEmbd;
    private final int nLayer;
    private final int nHead;
    private final int nInner;
    private final String activationFunction;
    private final double residPdrop;
    private final double embdPdrop;
    private final double attnPdrop;
    private final double layerNormEpsilon;
    private final double initializerRange;
    private final String summaryType;
    private final boolean summaryUseProj;
    private final String summaryActivation;
    private final boolean summaryProjToLabels;
    private final double summaryFirstDropout;
    private final boolean scaleAttnWeights;
    private final boolean scaleAttnByInverseLayerIdx;
    private final boolean reorderAndUpcastAttn;
    private final boolean addCrossAttention;

    public GPT2Config(PretrainedConfig base) {
        super(base);
        this.nPositions = toInt(base.extra().get("n_positions"), 1024);
        this.nEmbd = toInt(base.extra().get("n_embd"), 768);
        this.nLayer = toInt(base.extra().get("n_layer"), 12);
        this.nHead = toInt(base.extra().get("n_head"), 12);
        this.nInner = toInt(base.extra().get("n_inner"), 0);
        this.activationFunction = String.valueOf(base.extra().get("activation_function"));
        this.residPdrop = toDouble(base.extra().get("resid_pdrop"), 0.1);
        this.embdPdrop = toDouble(base.extra().get("embd_pdrop"), 0.1);
        this.attnPdrop = toDouble(base.extra().get("attn_pdrop"), 0.1);
        this.layerNormEpsilon = toDouble(base.extra().get("layer_norm_epsilon"), 1e-05);
        this.initializerRange = toDouble(base.extra().get("initializer_range"), 0.02);
        this.summaryType = String.valueOf(base.extra().get("summary_type"));
        this.summaryUseProj = base.extra().get("summary_use_proj") == Boolean.TRUE;
        this.summaryActivation = String.valueOf(base.extra().get("summary_activation"));
        this.summaryProjToLabels = base.extra().get("summary_proj_to_labels") == Boolean.TRUE;
        this.summaryFirstDropout = toDouble(base.extra().get("summary_first_dropout"), 0.1);
        this.scaleAttnWeights = base.extra().get("scale_attn_weights") == Boolean.TRUE;
        this.scaleAttnByInverseLayerIdx = base.extra().get("scale_attn_by_inverse_layer_idx") == Boolean.TRUE;
        this.reorderAndUpcastAttn = base.extra().get("reorder_and_upcast_attn") == Boolean.TRUE;
        this.addCrossAttention = base.extra().get("add_cross_attention") == Boolean.TRUE;
    }

    public int nPositions() { return toInt(base().extra().get("n_positions"), 1024); }
    public int nEmbd() { return toInt(base().extra().get("n_embd"), 768); }
    public int nLayer() { return toInt(base().extra().get("n_layer"), 12); }
    public int nHead() { return toInt(base().extra().get("n_head"), 12); }
    public int nInner() { return toInt(base().extra().get("n_inner"), 0); }
    public String activationFunction() { Object v = base().extra().get("activation_function"); return v == null ? "gelu_new" : String.valueOf(v); }
    public double residPdrop() { return toDouble(base().extra().get("resid_pdrop"), 0.1); }
    public double embdPdrop() { return toDouble(base().extra().get("embd_pdrop"), 0.1); }
    public double attnPdrop() { return toDouble(base().extra().get("attn_pdrop"), 0.1); }
    public double layerNormEpsilon() { return toDouble(base().extra().get("layer_norm_epsilon"), 1e-05); }
    public double initializerRange() { return toDouble(base().extra().get("initializer_range"), 0.02); }
    public String summaryType() { Object v = base().extra().get("summary_type"); return v == null ? "cls_index" : String.valueOf(v); }
    public boolean summaryUseProj() { return base().extra().get("summary_use_proj") == Boolean.TRUE; }
    public String summaryActivation() { Object v = base().extra().get("summary_activation"); return v == null ? null : String.valueOf(v); }
    public boolean summaryProjToLabels() { return base().extra().get("summary_proj_to_labels") == Boolean.TRUE; }
    public double summaryFirstDropout() { return toDouble(base().extra().get("summary_first_dropout"), 0.1); }
    public boolean scaleAttnWeights() { return base().extra().get("scale_attn_weights") == Boolean.TRUE; }
    public boolean scaleAttnByInverseLayerIdx() { return base().extra().get("scale_attn_by_inverse_layer_idx") == Boolean.TRUE; }
    public boolean reorderAndUpcastAttn() { return base().extra().get("reorder_and_upcast_attn") == Boolean.TRUE; }
    public boolean addCrossAttention() { return base().extra().get("add_cross_attention") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return GPT2Config.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}