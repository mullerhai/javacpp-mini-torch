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
package org.bytedeco.pytorch.llm.peft.mapping;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * HuggingFace {@code TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING}.
 *
 * <p>Used only when the caller left {@code target_modules} empty and did not
 * pass {@code "all-linear"}. Tutorial scripts always set targets explicitly.
 */
public final class PeftTargetModules {

    private PeftTargetModules() {}

    public static final List<String> LLAMA_ATTN = List.of("q_proj", "k_proj", "v_proj", "o_proj");
    public static final List<String> LLAMA_ATTN_MLP = List.of(
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj");
    public static final List<String> GPT2 = List.of("c_attn", "c_proj");

    public static List<String> forModelType(String modelType) {
        if (modelType == null || modelType.isBlank()) return LLAMA_ATTN;
        String t = modelType.toLowerCase(Locale.ROOT);
        if (t.contains("gpt2") || t.contains("gpt-2") || t.contains("causal")) return GPT2;
        if (t.contains("llama") || t.contains("mistral") || t.contains("qwen")
                || t.contains("smol") || t.contains("gemma") || t.contains("phi")
                || t.contains("falcon")) {
            return LLAMA_ATTN;
        }
        return LLAMA_ATTN;
    }

    public static Map<String, List<String>> mapping() {
        return Map.of(
                "llama", LLAMA_ATTN,
                "mistral", LLAMA_ATTN,
                "qwen2", LLAMA_ATTN,
                "smollm", LLAMA_ATTN,
                "gpt2", GPT2);
    }
}
