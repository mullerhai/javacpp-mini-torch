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
package org.bytedeco.pytorch.llm.peft.utils;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.utils.constants}.
 *
 * <p>Centralised file-name and layer-name constants for enterprise PEFT.
 */
public final class PeftConstants {

    private PeftConstants() {}

    /** Pickle-based weights filename (rarely used; safetensors preferred). */
    public static final String WEIGHTS_NAME = "adapter_model.bin";

    /** Safetensors weights filename written by {@code PeftModel.save_pretrained}. */
    public static final String SAFETENSORS_WEIGHTS_NAME = "adapter_model.safetensors";

    /** Adapter config filename. */
    public static final String CONFIG_NAME = "adapter_config.json";

    /** Tokenizer config filename (used by Auto* factories). */
    public static final String TOKENIZER_CONFIG_NAME = "tokenizer_config.json";

    /** Model card README filename. */
    public static final String MODEL_CARD_NAME = "README.md";

    /** Default module names that count as embedding layers (saved when {@code save_embedding_layers=True}). */
    public static final List<String> EMBEDDING_LAYER_NAMES =
            Collections.unmodifiableList(Arrays.asList("embed_tokens", "lm_head"));

    /** Default sequence/head names for classification task families. */
    public static final List<String> SEQ_CLS_HEAD_NAMES =
            Collections.unmodifiableList(Arrays.asList("score", "classifier"));

    /** Sentinel value for {@code target_modules="all-linear"}. */
    public static final String INCLUDE_LINEAR_LAYERS_SHORTHAND = "all-linear";

    /** Sentinel value used to disable target module matching. */
    public static final String DUMMY_TARGET_MODULES = "dummy-target-modules";

    /** Default placeholder for {@code model.config} when not a real transformers config. */
    public static final Map<String, Object> DUMMY_MODEL_CONFIG;
    static {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("model_type", "custom");
        DUMMY_MODEL_CONFIG = Collections.unmodifiableMap(m);
    }

    /** Below this many target modules, skip layer-status reporting optimisations. */
    public static final int MIN_TARGET_MODULES_FOR_OPTIMIZATION = 20;

    /** FP8 dtypes eligible for upcast to fp32 in {@code cast_adapter_dtype}. */
    public static final List<String> UPCAST_DTYPES = Collections.unmodifiableList(Arrays.asList(
            "float8_e4m3fn", "float8_e4m3fnuz",
            "float8_e5m2", "float8_e5m2fnuz",
            "float8_e8m0fnu"));

    /** Standard safetensors meta key for the producer library. */
    public static final String META_PRODUCER = "producer";
    /** Standard safetensors meta key for the merged state. */
    public static final String META_MERGED = "merged";
    /** Standard safetensors meta key for the format version. */
    public static final String META_FORMAT = "format";
}