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
import org.bytedeco.pytorch.llm.transformers.processor.ImageProcessor;
import org.bytedeco.pytorch.llm.transformers.processor.ImageProcessorFactory;

import java.io.IOException;
import java.nio.file.Path;

/**
 * HF {@code AutoImageProcessor.from_pretrained} entry point.
 *
 * <p>Wraps {@link ImageProcessor#fromPretrained(Path)} for directory-based loading.
 * For Hub models, downloads the snapshot first then delegates to {@code fromDirectory}.
 */
public final class AutoImageProcessor {

    private AutoImageProcessor() {}

    /**
     * Load an image processor from a HuggingFace Hub model id.
     *
     * @param modelId HF model id (e.g. {@code "microsoft/resnet-50"})
     * @param hub      Hub client (credentials, mirrors)
     * @return loaded {@link ImageProcessor}
     */
    public static ImageProcessor fromPretrained(String modelId, HfHub hub) throws IOException {
        Path snap = hub.snapshotDownload(modelId, "main", "models", java.util.List.of(
                "preprocessor_config.json", "processor_config.json",
                "config.json", "tokenizer.json"));
        return ImageProcessorFactory.fromPretrained(snap);
    }

    /**
     * Load an image processor from a local directory containing
     * {@code preprocessor_config.json} or {@code config.json}.
     *
     * @param dir directory containing processor artifacts
     * @return loaded {@link ImageProcessor}
     */
    public static ImageProcessor fromDirectory(Path dir) throws IOException {
        return ImageProcessorFactory.fromPretrained(dir);
    }
}
