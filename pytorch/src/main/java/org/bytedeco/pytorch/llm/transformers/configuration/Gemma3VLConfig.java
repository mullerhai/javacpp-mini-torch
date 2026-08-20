/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
 * HuggingFace {@code Gemma3VLConfig} — multimodal (vision + language) variant of Gemma 3.
 */
public final class Gemma3VLConfig extends Config {

    public static final String MODEL_TYPE = "gemma3_vl";

    public Gemma3VLConfig(PretrainedConfig base) {
        super(base);
    }

    public PretrainedConfig visionConfig() {
        Map<String, Object> extra = base().extra();
        Object vc = extra.get("vision_config");
        if (vc instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> vmap = (Map<String, Object>) vc;
            return PretrainedConfig.fromMap(vmap);
        }
        return null;
    }

    public boolean isVisionModel() {
        return Boolean.TRUE.equals(base().extra().get("is_vision_model"));
    }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return Gemma3VLConfig.class; }
}
