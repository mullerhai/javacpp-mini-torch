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
 * HuggingFace <code>deitConfig</code>.
 * Reference: transformers/models/deit/configuration_deit.py
 */
public final class DeiTConfig extends Config {

    public static final String MODEL_TYPE = "deit";

    private final double hiddenDropoutProb;
    private final double attentionProbsDropoutProb;
    private final double layerNormEps;
    private final int imageSize;
    private final int patchSize;
    private final int numChannels;
    private final boolean qkvBias;
    private final int encoderStride;
    private final int poolerOutputSize;
    private final String poolerAct;

    public DeiTConfig(PretrainedConfig base) {
        super(base);
        this.hiddenDropoutProb = toDouble(base.extra().get("hidden_dropout_prob"), 0.0);
        this.attentionProbsDropoutProb = toDouble(base.extra().get("attention_probs_dropout_prob"), 0.0);
        this.layerNormEps = toDouble(base.extra().get("layer_norm_eps"), 1e-12);
        this.imageSize = toInt(base.extra().get("image_size"), 224);
        this.patchSize = toInt(base.extra().get("patch_size"), 16);
        this.numChannels = toInt(base.extra().get("num_channels"), 3);
        this.qkvBias = base.extra().get("qkv_bias") == Boolean.TRUE;
        this.encoderStride = toInt(base.extra().get("encoder_stride"), 16);
        this.poolerOutputSize = toInt(base.extra().get("pooler_output_size"), 0);
        this.poolerAct = String.valueOf(base.extra().get("pooler_act"));
    }

    public double hiddenDropoutProb() { return toDouble(base().extra().get("hidden_dropout_prob"), 0.0); }
    public double attentionProbsDropoutProb() { return toDouble(base().extra().get("attention_probs_dropout_prob"), 0.0); }
    public double layerNormEps() { return toDouble(base().extra().get("layer_norm_eps"), 1e-12); }
    public int imageSize() { return toInt(base().extra().get("image_size"), 224); }
    public int patchSize() { return toInt(base().extra().get("patch_size"), 16); }
    public int numChannels() { return toInt(base().extra().get("num_channels"), 3); }
    public boolean qkvBias() { return base().extra().get("qkv_bias") == Boolean.TRUE; }
    public int encoderStride() { return toInt(base().extra().get("encoder_stride"), 16); }
    public int poolerOutputSize() { return toInt(base().extra().get("pooler_output_size"), 0); }
    public String poolerAct() { Object v = base().extra().get("pooler_act"); return v == null ? "tanh" : String.valueOf(v); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return DeiTConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}