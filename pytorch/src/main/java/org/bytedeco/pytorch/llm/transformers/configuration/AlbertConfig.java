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
 * HuggingFace <code>albertConfig</code>.
 * Reference: transformers/models/albert/configuration_albert.py
 */
public final class AlbertConfig extends Config {

    public static final String MODEL_TYPE = "albert";

    private final int embeddingSize;
    private final int numHiddenGroups;
    private final int innerGroupNum;
    private final String hiddenAct;
    private final double hiddenDropoutProb;
    private final double attentionProbsDropoutProb;
    private final int typeVocabSize;
    private final double layerNormEps;
    private final double classifierDropoutProb;

    public AlbertConfig(PretrainedConfig base) {
        super(base);
        this.embeddingSize = toInt(base.extra().get("embedding_size"), 128);
        this.numHiddenGroups = toInt(base.extra().get("num_hidden_groups"), 1);
        this.innerGroupNum = toInt(base.extra().get("inner_group_num"), 1);
        this.hiddenAct = String.valueOf(base.extra().get("hidden_act"));
        this.hiddenDropoutProb = toDouble(base.extra().get("hidden_dropout_prob"), 0.0);
        this.attentionProbsDropoutProb = toDouble(base.extra().get("attention_probs_dropout_prob"), 0.0);
        this.typeVocabSize = toInt(base.extra().get("type_vocab_size"), 2);
        this.layerNormEps = toDouble(base.extra().get("layer_norm_eps"), 1e-12);
        this.classifierDropoutProb = toDouble(base.extra().get("classifier_dropout_prob"), 0.1);
    }

    public int embeddingSize() { return toInt(base().extra().get("embedding_size"), 128); }
    public int numHiddenGroups() { return toInt(base().extra().get("num_hidden_groups"), 1); }
    public int innerGroupNum() { return toInt(base().extra().get("inner_group_num"), 1); }
    public String hiddenAct() { Object v = base().extra().get("hidden_act"); return v == null ? "gelu_new" : String.valueOf(v); }
    public double hiddenDropoutProb() { return toDouble(base().extra().get("hidden_dropout_prob"), 0.0); }
    public double attentionProbsDropoutProb() { return toDouble(base().extra().get("attention_probs_dropout_prob"), 0.0); }
    public int typeVocabSize() { return toInt(base().extra().get("type_vocab_size"), 2); }
    public double layerNormEps() { return toDouble(base().extra().get("layer_norm_eps"), 1e-12); }
    public double classifierDropoutProb() { return toDouble(base().extra().get("classifier_dropout_prob"), 0.1); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return AlbertConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}