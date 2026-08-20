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
 * HuggingFace <code>debertaConfig</code>.
 * Reference: transformers/models/deberta/configuration_deberta.py
 */
public final class DebertaConfig extends Config {

    public static final String MODEL_TYPE = "deberta";

    private final double hiddenDropoutProb;
    private final double attentionProbsDropoutProb;
    private final int typeVocabSize;
    private final boolean relativeAttention;
    private final int maxRelativePositions;
    private final boolean positionBiasedInput;
    private final String posAttType;
    private final double poolerDropout;
    private final String poolerHiddenAct;
    private final boolean legacy;

    public DebertaConfig(PretrainedConfig base) {
        super(base);
        this.hiddenDropoutProb = toDouble(base.extra().get("hidden_dropout_prob"), 0.1);
        this.attentionProbsDropoutProb = toDouble(base.extra().get("attention_probs_dropout_prob"), 0.1);
        this.typeVocabSize = toInt(base.extra().get("type_vocab_size"), 0);
        this.relativeAttention = base.extra().get("relative_attention") == Boolean.TRUE;
        this.maxRelativePositions = toInt(base.extra().get("max_relative_positions"), -1);
        this.positionBiasedInput = base.extra().get("position_biased_input") == Boolean.TRUE;
        this.posAttType = String.valueOf(base.extra().get("pos_att_type"));
        this.poolerDropout = toDouble(base.extra().get("pooler_dropout"), 0.0);
        this.poolerHiddenAct = String.valueOf(base.extra().get("pooler_hidden_act"));
        this.legacy = base.extra().get("legacy") == Boolean.TRUE;
    }

    public double hiddenDropoutProb() { return toDouble(base().extra().get("hidden_dropout_prob"), 0.1); }
    public double attentionProbsDropoutProb() { return toDouble(base().extra().get("attention_probs_dropout_prob"), 0.1); }
    public int typeVocabSize() { return toInt(base().extra().get("type_vocab_size"), 0); }
    public boolean relativeAttention() { return base().extra().get("relative_attention") == Boolean.TRUE; }
    public int maxRelativePositions() { return toInt(base().extra().get("max_relative_positions"), -1); }
    public boolean positionBiasedInput() { return base().extra().get("position_biased_input") == Boolean.TRUE; }
    public String posAttType() { Object v = base().extra().get("pos_att_type"); return v == null ? null : String.valueOf(v); }
    public double poolerDropout() { return toDouble(base().extra().get("pooler_dropout"), 0.0); }
    public String poolerHiddenAct() { Object v = base().extra().get("pooler_hidden_act"); return v == null ? "gelu" : String.valueOf(v); }
    public boolean legacy() { return base().extra().get("legacy") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return DebertaConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}