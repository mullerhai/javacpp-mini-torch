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
 * HuggingFace <code>falconConfig</code>.
 * Reference: transformers/models/falcon/configuration_falcon.py
 */
public final class FalconConfig extends Config {

    public static final String MODEL_TYPE = "falcon";

    private final int numLnInParallelAttn;
    private final double hiddenDropout;
    private final double attentionDropout;
    private final int numKvHeads;
    private final boolean alibi;
    private final boolean newDecoderArchitecture;
    private final boolean multiQuery;
    private final boolean parallelAttn;
    private final boolean bias;
    private final int ffnHiddenSize;
    private final String activation;

    public FalconConfig(PretrainedConfig base) {
        super(base);
        this.numLnInParallelAttn = toInt(base.extra().get("num_ln_in_parallel_attn"), 0);
        this.hiddenDropout = toDouble(base.extra().get("hidden_dropout"), 0.0);
        this.attentionDropout = toDouble(base.extra().get("attention_dropout"), 0.0);
        this.numKvHeads = toInt(base.extra().get("num_kv_heads"), 0);
        this.alibi = base.extra().get("alibi") == Boolean.TRUE;
        this.newDecoderArchitecture = base.extra().get("new_decoder_architecture") == Boolean.TRUE;
        this.multiQuery = base.extra().get("multi_query") == Boolean.TRUE;
        this.parallelAttn = base.extra().get("parallel_attn") == Boolean.TRUE;
        this.bias = base.extra().get("bias") == Boolean.TRUE;
        this.ffnHiddenSize = toInt(base.extra().get("ffn_hidden_size"), 0);
        this.activation = String.valueOf(base.extra().get("activation"));
    }

    public int numLnInParallelAttn() { return toInt(base().extra().get("num_ln_in_parallel_attn"), 0); }
    public double hiddenDropout() { return toDouble(base().extra().get("hidden_dropout"), 0.0); }
    public double attentionDropout() { return toDouble(base().extra().get("attention_dropout"), 0.0); }
    public int numKvHeads() { return toInt(base().extra().get("num_kv_heads"), 0); }
    public boolean alibi() { return base().extra().get("alibi") == Boolean.TRUE; }
    public boolean newDecoderArchitecture() { return base().extra().get("new_decoder_architecture") == Boolean.TRUE; }
    public boolean multiQuery() { return base().extra().get("multi_query") == Boolean.TRUE; }
    public boolean parallelAttn() { return base().extra().get("parallel_attn") == Boolean.TRUE; }
    public boolean bias() { return base().extra().get("bias") == Boolean.TRUE; }
    public int ffnHiddenSize() { return toInt(base().extra().get("ffn_hidden_size"), 0); }
    public String activation() { Object v = base().extra().get("activation"); return v == null ? "gelu" : String.valueOf(v); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return FalconConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}