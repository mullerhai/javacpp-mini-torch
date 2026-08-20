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
 * HuggingFace <code>bloomConfig</code>.
 * Reference: transformers/models/bloom/configuration_bloom.py
 */
public final class BloomConfig extends Config {

    public static final String MODEL_TYPE = "bloom";

    private final int nLayer;
    private final int nHead;
    private final double layerNormEpsilon;
    private final boolean applyResidualConnectionPostLayernorm;
    private final double hiddenDropout;
    private final double attentionDropout;
    private final int pretrainingTp;
    private final boolean slowButExact;

    public BloomConfig(PretrainedConfig base) {
        super(base);
        this.nLayer = toInt(base.extra().get("n_layer"), 2);
        this.nHead = toInt(base.extra().get("n_head"), 8);
        this.layerNormEpsilon = toDouble(base.extra().get("layer_norm_epsilon"), 1e-05);
        this.applyResidualConnectionPostLayernorm = base.extra().get("apply_residual_connection_post_layernorm") == Boolean.TRUE;
        this.hiddenDropout = toDouble(base.extra().get("hidden_dropout"), 0.0);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.pretrainingTp = toInt(base.extra().get("pretraining_tp"), 1);
        this.slowButExact = base.extra().get("slow_but_exact") == Boolean.TRUE;
    }

    public int nLayer() { return toInt(base().extra().get("n_layer"), 2); }
    public int nHead() { return toInt(base().extra().get("n_head"), 8); }
    public double layerNormEpsilon() { return toDouble(base().extra().get("layer_norm_epsilon"), 1e-05); }
    public boolean applyResidualConnectionPostLayernorm() { return base().extra().get("apply_residual_connection_post_layernorm") == Boolean.TRUE; }
    public double hiddenDropout() { return toDouble(base().extra().get("hidden_dropout"), 0.0); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public int pretrainingTp() { return toInt(base().extra().get("pretraining_tp"), 1); }
    public boolean slowButExact() { return base().extra().get("slow_but_exact") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return BloomConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}