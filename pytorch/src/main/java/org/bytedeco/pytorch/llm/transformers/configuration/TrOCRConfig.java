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
 * HuggingFace <code>TrOCRConfig</code>.
 * Reference: transformers/models/trocr/configuration_trocr.py
 */
public final class TrOCRConfig extends Config {

    public static final String MODEL_TYPE = "trocr";

    private final String encoderModelType;
    private final String decoderModelType;
    private final Map<String, Object> visionConfig;
    private final int imageSize;
    private final int decoderStartTokenId;

    public TrOCRConfig(PretrainedConfig base) {
        super(base);
        this.encoderModelType = String.valueOf(base.extra().get("encoder_model_type"));
        this.decoderModelType = String.valueOf(base.extra().get("decoder_model_type"));
        this.visionConfig = base.extra();
        this.imageSize = toInt(base.extra().get("image_size"), 384);
        this.decoderStartTokenId = toInt(base.extra().get("decoder_start_token_id"), 0);
    }

    public String encoderModelType() { Object v = base().extra().get("encoder_model_type"); return v == null ? "vit" : String.valueOf(v); }
    public String decoderModelType() { Object v = base().extra().get("decoder_model_type"); return v == null ? "roberta" : String.valueOf(v); }
    public Map<String, Object> visionConfig() { return base().extra(); }
    public int imageSize() { return toInt(base().extra().get("image_size"), 384); }
    public int decoderStartTokenId() { return toInt(base().extra().get("decoder_start_token_id"), 0); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return TrOCRConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}
