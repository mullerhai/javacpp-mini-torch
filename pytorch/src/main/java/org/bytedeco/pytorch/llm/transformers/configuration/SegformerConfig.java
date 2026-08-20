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
 * HuggingFace <code>SegformerConfig</code>.
 * Reference: transformers/models/segformer/configuration_segformer.py
 */
public final class SegformerConfig extends Config {

    public static final String MODEL_TYPE = "segformer";

    private final int numChannels;
    private final int patchSize;
    private final int imageSize;
    private final int decoderHiddenSize;
    private final String hiddenAct;
    private final double layerNormEps;
    private final double attentionDropout;
    private final double hiddenDropoutProb;

    public SegformerConfig(PretrainedConfig base) {
        super(base);
        this.numChannels = toInt(base.extra().get("num_channels"), 3);
        this.patchSize = toInt(base.extra().get("patch_size"), 4);
        this.imageSize = toInt(base.extra().get("image_size"), 512);
        this.decoderHiddenSize = toInt(base.extra().get("decoder_hidden_size"), 768);
        this.hiddenAct = String.valueOf(base.extra().get("hidden_act"));
        this.layerNormEps = toDouble(base.extra().get("layer_norm_eps"), 1e-12);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.hiddenDropoutProb = toDouble(base.extra().get("hidden_dropout_prob"), 0.0);
    }

    public int numChannels() { return toInt(base().extra().get("num_channels"), 3); }
    public int patchSize() { return toInt(base().extra().get("patch_size"), 4); }
    public int imageSize() { return toInt(base().extra().get("image_size"), 512); }
    public int decoderHiddenSize() { return toInt(base().extra().get("decoder_hidden_size"), 768); }
    public String hiddenAct() { Object v = base().extra().get("hidden_act"); return v == null ? "silu" : String.valueOf(v); }
    public double layerNormEps() { return toDouble(base().extra().get("layer_norm_eps"), 1e-12); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public double hiddenDropoutProb() { return toDouble(base().extra().get("hidden_dropout_prob"), 0.0); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return SegformerConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}
