/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
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

import java.util.List;

/**
 * HuggingFace {@code Gemma3Config}.
 *
 * <p>Adds an explicit per-layer attention type pattern
 * (e.g. {@code ["sliding_attention", "full_attention", ...]}),
 * RoPE scaling for extended context, and {@code query_pre_attn_scalar}.
 */
public final class Gemma3Config extends Config {

    public static final String MODEL_TYPE = "gemma3";

    public Gemma3Config(PretrainedConfig base) {
        super(base);
    }

    public List<String> layerTypes() { return base().layerTypes(); }
    public int queryPreAttnScalar() { return base().queryPreAttnScalar(); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return Gemma3Config.class; }
}
