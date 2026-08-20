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
 * HuggingFace <code>dinov2Config</code>.
 * Reference: transformers/models/dinov2/configuration_dinov2.py
 */
public final class Dinov2Config extends Config {

    public static final String MODEL_TYPE = "dinov2";

    private final int mlpRatio;
    private final String hiddenAct;
    private final double hiddenDropoutProb;
    private final double attentionProbsDropoutProb;
    private final double layerNormEps;
    private final int patchSize;
    private final int numChannels;
    private final boolean qkvBias;
    private final double layerscaleValue;
    private final double dropPathRate;
    private final boolean useSwigluFfn;
    private final boolean applyLayernorm;
    private final boolean reshapeHiddenStates;
    private final boolean useMaskToken;

    public Dinov2Config(PretrainedConfig base) {
        super(base);
        this.mlpRatio = toInt(base.extra().get("mlp_ratio"), 4);
        this.hiddenAct = String.valueOf(base.extra().get("hidden_act"));
        this.hiddenDropoutProb = toDouble(base.extra().get("hidden_dropout_prob"), 0.0);
        this.attentionProbsDropoutProb = toDouble(base.extra().get("attention_probs_dropout_prob"), 0.0);
        this.layerNormEps = toDouble(base.extra().get("layer_norm_eps"), 1e-06);
        this.patchSize = toInt(base.extra().get("patch_size"), 14);
        this.numChannels = toInt(base.extra().get("num_channels"), 3);
        this.qkvBias = base.extra().get("qkv_bias") == Boolean.TRUE;
        this.layerscaleValue = toDouble(base.extra().get("layerscale_value"), 1.0);
        this.dropPathRate = toDouble(base.extra().get("drop_path_rate"), 0.0);
        this.useSwigluFfn = base.extra().get("use_swiglu_ffn") == Boolean.TRUE;
        this.applyLayernorm = base.extra().get("apply_layernorm") == Boolean.TRUE;
        this.reshapeHiddenStates = base.extra().get("reshape_hidden_states") == Boolean.TRUE;
        this.useMaskToken = base.extra().get("use_mask_token") == Boolean.TRUE;
    }

    public int mlpRatio() { return toInt(base().extra().get("mlp_ratio"), 4); }
    public String hiddenAct() { Object v = base().extra().get("hidden_act"); return v == null ? "gelu" : String.valueOf(v); }
    public double hiddenDropoutProb() { return toDouble(base().extra().get("hidden_dropout_prob"), 0.0); }
    public double attentionProbsDropoutProb() { return toDouble(base().extra().get("attention_probs_dropout_prob"), 0.0); }
    public double layerNormEps() { return toDouble(base().extra().get("layer_norm_eps"), 1e-06); }
    public int patchSize() { return toInt(base().extra().get("patch_size"), 14); }
    public int numChannels() { return toInt(base().extra().get("num_channels"), 3); }
    public boolean qkvBias() { return base().extra().get("qkv_bias") == Boolean.TRUE; }
    public double layerscaleValue() { return toDouble(base().extra().get("layerscale_value"), 1.0); }
    public double dropPathRate() { return toDouble(base().extra().get("drop_path_rate"), 0.0); }
    public boolean useSwigluFfn() { return base().extra().get("use_swiglu_ffn") == Boolean.TRUE; }
    public boolean applyLayernorm() { return base().extra().get("apply_layernorm") == Boolean.TRUE; }
    public boolean reshapeHiddenStates() { return base().extra().get("reshape_hidden_states") == Boolean.TRUE; }
    public boolean useMaskToken() { return base().extra().get("use_mask_token") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return Dinov2Config.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}