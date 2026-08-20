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
package org.bytedeco.pytorch.llm.peft;

/** PEFT adapter family (mirrors Hugging Face {@code peft.utils.peft_types.PeftType}, v0.20.0). */
public enum PeftType {
    // Prompt learning
    PROMPT_TUNING,
    MULTITASK_PROMPT_TUNING,
    P_TUNING,
    PREFIX_TUNING,
    CPT,
    CARTRIDGE,
    ADAPTION_PROMPT,
    TRAINABLE_TOKENS,

    // Adapter family
    LORA,
    QLORA,
    ADALORA,
    ADAMSS,
    IA3,
    BOFT,
    OFT,
    HRA,
    HIRA,
    SHIRA,
    ROAD,
    LOHA,
    LOKR,
    LILY,
    VERA,
    PVERA,
    FROD,
    DELORA,
    DEFT,
    C3A,
    GLORA,
    GRALORA,
    RANDLORA,
    VBLORA,
    UNILORA,
    PEANUT,
    PSOFT,
    OSF,
    MISS,
    BEFT,
    FOURIERFT,
    WAVEFT,
    TINYLORA,
    XLORA,
    POLY,
    LN_TUNING;

    /** HuggingFace parameter prefix for this adapter family. */
    public String prefix() {
        switch (this) {
            // LoRA family
            case LORA: case QLORA: case ADALORA: case ADAMSS: case DELORA:
            case C3A: case DEFT: case GLORA: case GRALORA: case LILY:
            case MISS: case OSF: case PEANUT: case PSOFT: case RANDLORA:
            case SHIRA: case TINYLORA: case UNILORA: case VBLORA:
                return "lora_";
            // IA3 family
            case IA3:
                return "ia3_";
            // OFT family
            case OFT:
                return "oft_";
            case BOFT:
                return "boft_";
            // Lycoris family (LoHa / LoKr)
            case LOHA:
                return "hada_";
            case LOKR:
                return "lokr_";
            // VeRA / PVeRA / FROD
            case VERA: case PVERA: case FROD:
                return "vera_lambda_";
            // Prompt learning
            case PROMPT_TUNING: case MULTITASK_PROMPT_TUNING: case CPT:
                return "prompt_embeddings_";
            case P_TUNING:
                return "prompt_encoder_";
            case PREFIX_TUNING:
                return "prefix_encoder_";
            case CARTRIDGE:
                return "cartridge_";
            case ADAPTION_PROMPT:
                return "adaption_prompt_";
            case TRAINABLE_TOKENS:
                return "trainable_tokens_";
            // Single-purpose
            case HRA:
                return "hra_";
            case HIRA:
                return "hira_";
            case ROAD:
                return "road_";
            case BEFT:
                return "beft_";
            case FOURIERFT:
                return "fourierft_";
            case WAVEFT:
                return "waveft_";
            case XLORA:
                return "xlora_";
            case POLY:
                return "poly_";
            case LN_TUNING:
                return "ln_tuning_";
        }
        return name().toLowerCase() + "_";
    }

    /** Tuners compatible with {@code PeftMixedModel} (HF {@code COMPATIBLE_TUNER_TYPES}). */
    public boolean isMixedCompatible() {
        switch (this) {
            case LORA: case LOHA: case LOKR: case ADALORA: case OFT: case SHIRA:
                return true;
            default:
                return false;
        }
    }

    /** Prompt-learning families (need {@code PromptLearningConfig} + prompt encoder). */
    public boolean isPromptLearning() {
        switch (this) {
            case PROMPT_TUNING: case PREFIX_TUNING: case P_TUNING:
            case MULTITASK_PROMPT_TUNING: case CPT:
                return true;
            default:
                return false;
        }
    }

    /** True for adaption-prompt (carries its own ModuleDict; not a target-module swap). */
    public boolean isAdaptionPrompt() {
        return this == ADAPTION_PROMPT;
    }
}
