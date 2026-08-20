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
 * HuggingFace <code>gptjConfig</code>.
 * Reference: transformers/models/gptj/configuration_gptj.py
 */
public final class GPTJConfig extends Config {

    public static final String MODEL_TYPE = "gptj";

    private final int nPositions;
    private final int nEmbd;
    private final int nLayer;
    private final int nHead;
    private final int rotaryDim;
    private final int nInner;
    private final String activationFunction;
    private final double residPdrop;
    private final double embdPdrop;
    private final double attnPdrop;
    private final double layerNormEpsilon;

    public GPTJConfig(PretrainedConfig base) {
        super(base);
        this.nPositions = toInt(base.extra().get("n_positions"), 2048);
        this.nEmbd = toInt(base.extra().get("n_embd"), 4096);
        this.nLayer = toInt(base.extra().get("n_layer"), 28);
        this.nHead = toInt(base.extra().get("n_head"), 16);
        this.rotaryDim = toInt(base.extra().get("rotary_dim"), 64);
        this.nInner = toInt(base.extra().get("n_inner"), 0);
        this.activationFunction = String.valueOf(base.extra().get("activation_function"));
        this.residPdrop = toDouble(base.extra().get("resid_pdrop"), 0.0);
        this.embdPdrop = toDouble(base.extra().get("embd_pdrop"), 0.0);
        this.attnPdrop = toDouble(base.extra().get("attn_pdrop"), 0.0);
        this.layerNormEpsilon = toDouble(base.extra().get("layer_norm_epsilon"), 1e-05);
    }

    public int nPositions() { return toInt(base().extra().get("n_positions"), 2048); }
    public int nEmbd() { return toInt(base().extra().get("n_embd"), 4096); }
    public int nLayer() { return toInt(base().extra().get("n_layer"), 28); }
    public int nHead() { return toInt(base().extra().get("n_head"), 16); }
    public int rotaryDim() { return toInt(base().extra().get("rotary_dim"), 64); }
    public int nInner() { return toInt(base().extra().get("n_inner"), 0); }
    public String activationFunction() { Object v = base().extra().get("activation_function"); return v == null ? "gelu_new" : String.valueOf(v); }
    public double residPdrop() { return toDouble(base().extra().get("resid_pdrop"), 0.0); }
    public double embdPdrop() { return toDouble(base().extra().get("embd_pdrop"), 0.0); }
    public double attnPdrop() { return toDouble(base().extra().get("attn_pdrop"), 0.0); }
    public double layerNormEpsilon() { return toDouble(base().extra().get("layer_norm_epsilon"), 1e-05); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return GPTJConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}