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
package org.bytedeco.pytorch.llm.transformers.auto;

import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.nn.Module;

import java.io.IOException;
import java.nio.file.Path;

/**
 * HF {@code AutoModelForTextToImage.from_pretrained} entry point.
 * Delegates to Diffusers for Flux-based text-to-image generation.
 */
public final class AutoModelForTextToImage {

    private AutoModelForTextToImage() {}

    public static final class Bundle {
        public final Module model;
        public final PretrainedConfig config;
        public final Path snapshot;
        public final String modelType;

        public Bundle(Module model, PretrainedConfig config, Path snapshot, String modelType) {
            this.model = model;
            this.config = config;
            this.snapshot = snapshot;
            this.modelType = modelType;
        }
    }

    public static Bundle fromPretrained(String modelId, HfHub hub) throws IOException {
        throw new UnsupportedOperationException(
                "Text-to-image (Flux) is handled via Diffusers pipeline; " +
                "use org.bytedeco.pytorch.llm.transformers.diffusers.FluxPipeline instead.");
    }

    public static Bundle fromDirectory(Path dir) throws IOException {
        throw new UnsupportedOperationException(
                "Text-to-image (Flux) is handled via Diffusers pipeline; " +
                "use org.bytedeco.pytorch.llm.transformers.diffusers.FluxPipeline instead.");
    }
}
