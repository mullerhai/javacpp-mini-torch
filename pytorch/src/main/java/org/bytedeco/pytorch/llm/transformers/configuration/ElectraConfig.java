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
 * HuggingFace <code>electraConfig</code>.
 * Reference: transformers/models/electra/configuration_electra.py
 */
public final class ElectraConfig extends Config {

    public static final String MODEL_TYPE = "electra";

    private final int embeddingSize;
    private final double hiddenDropoutProb;
    private final double attentionProbsDropoutProb;
    private final int typeVocabSize;
    private final double layerNormEps;
    private final String summaryType;
    private final boolean summaryUseProj;
    private final String summaryActivation;
    private final double summaryLastDropout;
    private final double classifierDropout;
    private final boolean isDecoder;
    private final boolean addCrossAttention;

    public ElectraConfig(PretrainedConfig base) {
        super(base);
        this.embeddingSize = toInt(base.extra().get("embedding_size"), 128);
        this.hiddenDropoutProb = toDouble(base.extra().get("hidden_dropout_prob"), 0.1);
        this.attentionProbsDropoutProb = toDouble(base.extra().get("attention_probs_dropout_prob"), 0.1);
        this.typeVocabSize = toInt(base.extra().get("type_vocab_size"), 2);
        this.layerNormEps = toDouble(base.extra().get("layer_norm_eps"), 1e-12);
        this.summaryType = String.valueOf(base.extra().get("summary_type"));
        this.summaryUseProj = base.extra().get("summary_use_proj") == Boolean.TRUE;
        this.summaryActivation = String.valueOf(base.extra().get("summary_activation"));
        this.summaryLastDropout = toDouble(base.extra().get("summary_last_dropout"), 0.1);
        this.classifierDropout = toDouble(base.extra().get("classifier_dropout"), 0.0);
        this.isDecoder = base.extra().get("is_decoder") == Boolean.TRUE;
        this.addCrossAttention = base.extra().get("add_cross_attention") == Boolean.TRUE;
    }

    public int embeddingSize() { return toInt(base().extra().get("embedding_size"), 128); }
    public double hiddenDropoutProb() { return toDouble(base().extra().get("hidden_dropout_prob"), 0.1); }
    public double attentionProbsDropoutProb() { return toDouble(base().extra().get("attention_probs_dropout_prob"), 0.1); }
    public int typeVocabSize() { return toInt(base().extra().get("type_vocab_size"), 2); }
    public double layerNormEps() { return toDouble(base().extra().get("layer_norm_eps"), 1e-12); }
    public String summaryType() { Object v = base().extra().get("summary_type"); return v == null ? "first" : String.valueOf(v); }
    public boolean summaryUseProj() { return base().extra().get("summary_use_proj") == Boolean.TRUE; }
    public String summaryActivation() { Object v = base().extra().get("summary_activation"); return v == null ? "gelu" : String.valueOf(v); }
    public double summaryLastDropout() { return toDouble(base().extra().get("summary_last_dropout"), 0.1); }
    public double classifierDropout() { return toDouble(base().extra().get("classifier_dropout"), 0.0); }
    public boolean isDecoder() { return base().extra().get("is_decoder") == Boolean.TRUE; }
    public boolean addCrossAttention() { return base().extra().get("add_cross_attention") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return ElectraConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}