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
 * HuggingFace <code>gpt_neoxConfig</code>.
 * Reference: transformers/models/gpt_neox/configuration_gpt_neox.py
 */
public final class GPTNeoXConfig extends Config {

    public static final String MODEL_TYPE = "gpt_neox";

    private final String hiddenAct;
    private final double attentionDropout;
    private final double hiddenDropout;
    private final double classifierDropout;
    private final boolean useParallelResidual;
    private final boolean attentionBias;
    private final boolean isDecoder;

    public GPTNeoXConfig(PretrainedConfig base) {
        super(base);
        this.hiddenAct = String.valueOf(base.extra().get("hidden_act"));
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.hiddenDropout = toDouble(base.extra().get("hidden_dropout"), 0.0);
        this.classifierDropout = toDouble(base.extra().get("classifier_dropout"), 0.1);
        this.useParallelResidual = base.extra().get("use_parallel_residual") == Boolean.TRUE;
        this.attentionBias = base.extra().get("attention_bias") == Boolean.TRUE;
        this.isDecoder = base.extra().get("is_decoder") == Boolean.TRUE;
    }

    public String hiddenAct() { Object v = base().extra().get("hidden_act"); return v == null ? "gelu" : String.valueOf(v); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public double hiddenDropout() { return toDouble(base().extra().get("hidden_dropout"), 0.0); }
    public double classifierDropout() { return toDouble(base().extra().get("classifier_dropout"), 0.1); }
    public boolean useParallelResidual() { return base().extra().get("use_parallel_residual") == Boolean.TRUE; }
    public boolean attentionBias() { return base().extra().get("attention_bias") == Boolean.TRUE; }
    public boolean isDecoder() { return base().extra().get("is_decoder") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return GPTNeoXConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}