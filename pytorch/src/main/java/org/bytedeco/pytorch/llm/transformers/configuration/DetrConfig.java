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

/**
 * HuggingFace <code>DetrConfig</code> — DEtection TRansformer.
 * Reference: transformers/models/detr/configuration_detr.py
 */
public final class DetrConfig extends Config {

    public static final String MODEL_TYPE = "detr";

    private final int numLabels;
    private final int imageSize;
    private final int usePretrainedBackbone;
    private final double lossBeta1;
    private final double lossBeta2;

    public DetrConfig(PretrainedConfig base) {
        super(base);
        this.numLabels = toInt(base.extra().get("num_labels"), 91);
        this.imageSize = toInt(base.extra().get("image_size"), 800);
        this.usePretrainedBackbone = toInt(base.extra().get("use_pretrained_backbone"), 1);
        this.lossBeta1 = toDouble(base.extra().get("loss_beta1"), 0.9);
        this.lossBeta2 = toDouble(base.extra().get("loss_beta2"), 0.999);
    }

    public int numLabels() { return toInt(base().extra().get("num_labels"), 91); }
    public int imageSize() { return toInt(base().extra().get("image_size"), 800); }
    public boolean usePretrainedBackbone() { return base().extra().get("use_pretrained_backbone") == Boolean.TRUE; }
    public double lossBeta1() { return toDouble(base().extra().get("loss_beta1"), 0.9); }
    public double lossBeta2() { return toDouble(base().extra().get("loss_beta2"), 0.999); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return DetrConfig.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}
