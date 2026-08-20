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
 * HuggingFace <code>Pix2StructConfig</code>.
 * Reference: transformers/models/pix2struct/configuration_pix2struct.py
 */
public final class Pix2StructConfig extends Config {

    public static final String MODEL_TYPE = "pix2struct";

    private final int imageSize;
    private final int patchSize;
    private final int numChannels;
    private final int imageHiddenSize;
    private final int seqLen;
    private final double attentionDropout;
    private final double hiddenDropoutProb;
    private final double layerNormEps;
    private final boolean isGumbel;
    private final boolean useCache;

    public Pix2StructConfig(PretrainedConfig base) {
        super(base);
        this.imageSize = toInt(base.extra().get("image_size"), 224);
        this.patchSize = toInt(base.extra().get("patch_size"), 16);
        this.numChannels = toInt(base.extra().get("num_channels"), 3);
        this.imageHiddenSize = toInt(base.extra().get("image_hidden_size"), 768);
        this.seqLen = toInt(base.extra().get("seq_len"), 4096);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.hiddenDropoutProb = toDouble(base.extra().get("hidden_dropout_prob"), 0.0);
        this.layerNormEps = toDouble(base.extra().get("layer_norm_eps"), 1e-12);
        this.isGumbel = base.extra().get("is_gumbel") == Boolean.TRUE;
        this.useCache = base.extra().get("use_cache") == Boolean.TRUE;
    }

    public int imageSize() { return toInt(base().extra().get("image_size"), 224); }
    public int patchSize() { return toInt(base().extra().get("patch_size"), 16); }
    public int numChannels() { return toInt(base().extra().get("num_channels"), 3); }
    public int imageHiddenSize() { return toInt(base().extra().get("image_hidden_size"), 768); }
    public int seqLen() { return toInt(base().extra().get("seq_len"), 4096); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public double hiddenDropoutProb() { return toDouble(base().extra().get("hidden_dropout_prob"), 0.0); }
    public double layerNormEps() { return toDouble(base().extra().get("layer_norm_eps"), 1e-12); }
    public boolean isGumbel() { return base().extra().get("is_gumbel") == Boolean.TRUE; }
    public boolean useCache() { return base().extra().get("use_cache") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return Pix2StructConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}
