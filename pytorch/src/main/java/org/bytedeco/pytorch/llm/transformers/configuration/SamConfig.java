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
 * HuggingFace <code>SamConfig</code>.
 * Reference: transformers/models/sam/configuration_sam.py
 */
public final class SamConfig extends Config {

    public static final String MODEL_TYPE = "sam";

    private final Map<String, Object> visionConfig;
    private final Map<String, Object> promptEncoderConfig;
    private final Map<String, Object> maskDecoderConfig;
    private final boolean pixelMeanAsImageNet;
    private final boolean pixelStdAsImageNet;

    public SamConfig(PretrainedConfig base) {
        super(base);
        this.visionConfig = base.extra();
        this.promptEncoderConfig = base.extra();
        this.maskDecoderConfig = base.extra();
        this.pixelMeanAsImageNet = base.extra().get("pixel_mean_as_image_net") == Boolean.TRUE;
        this.pixelStdAsImageNet = base.extra().get("pixel_std_as_image_net") == Boolean.TRUE;
    }

    public Map<String, Object> visionConfig() { return base().extra(); }
    public Map<String, Object> promptEncoderConfig() { return base().extra(); }
    public Map<String, Object> maskDecoderConfig() { return base().extra(); }
    public boolean pixelMeanAsImageNet() { return base().extra().get("pixel_mean_as_image_net") == Boolean.TRUE; }
    public boolean pixelStdAsImageNet() { return base().extra().get("pixel_std_as_image_net") == Boolean.TRUE; }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return SamConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}
