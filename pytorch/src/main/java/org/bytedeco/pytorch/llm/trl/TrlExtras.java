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
package org.bytedeco.pytorch.llm.trl;

import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;

/**
 * HuggingFace TRL helpers from {@code trl.trainer.utils}:
 * {@code get_peft_config}, {@code get_quantization_config}, {@code get_kbit_device_map}.
 */
public final class TrlExtras {

    private TrlExtras() {}

    /** Python {@code get_peft_config(model_args)} — {@code null} when PEFT is off. */
    public static LoraConfig getPeftConfig(ModelConfig args) {
        return get_peft_config(args);
    }

    public static LoraConfig get_peft_config(ModelConfig args) {
        if (args == null || !args.usePeft()) return null;
        return LoraConfig.builder()
                .r(args.loraR())
                .lora_alpha(args.loraAlpha())
                .lora_dropout(args.loraDropout())
                .targetModules(args.loraTargetModules())
                .bias("none")
                .task_type("CAUSAL_LM")
                .build();
    }

    /** Python {@code get_quantization_config(model_args)}. */
    public static BitsAndBytesConfig getQuantizationConfig(ModelConfig args) {
        return get_quantization_config(args);
    }

    public static BitsAndBytesConfig get_quantization_config(ModelConfig args) {
        if (args == null) return null;
        if (args.loadIn4Bit()) {
            return BitsAndBytesConfig.builder()
                    .load_in_4bit(true)
                    .bnb_4bit_quant_type("nf4")
                    .bnb_4bit_compute_dtype(args.torchDtype() == null ? "bfloat16" : args.torchDtype())
                    .bnb_4bit_use_double_quant(true)
                    .build();
        }
        if (args.loadIn8Bit()) {
            return BitsAndBytesConfig.builder().load_in_8bit(true).build();
        }
        return null;
    }

    /** Python {@code get_kbit_device_map()} → {@code {"": "cuda:0"}} / {@code "auto"}. */
    public static String getKbitDeviceMap() {
        return get_kbit_device_map();
    }

    public static String get_kbit_device_map() {
        return "auto";
    }
}
