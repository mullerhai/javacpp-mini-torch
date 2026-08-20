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
package org.bytedeco.pytorch.llm.transformers.processor;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Bridges the {@link ImageProcessor} to the {@link AutoProcessor} loader.
 *
 * <p>Since {@code ImageProcessor.fromPretrained(Path)} does not exist as a static
 * factory, this helper resolves the directory through {@link AutoProcessor} and
 * extracts (or synthesises) the image processor.
 */
public final class ImageProcessorFactory {

    private ImageProcessorFactory() {}

    /**
     * Load (or synthesise) an {@link ImageProcessor} from a model directory.
     *
     * <ol>
     *   <li>Try {@link AutoProcessor#fromPretrained(Path)} and extract the
     *       {@code ImageProcessor} component if present.</li>
     *   <li>Fallback to {@link ImageProcessor#createImageNet()}.</li>
     * </ol>
     *
     * @param dir model directory containing {@code config.json} /
     *            {@code preprocessor_config.json}
     * @return an {@link ImageProcessor} ready for inference
     */
    public static ImageProcessor fromPretrained(Path dir) {
        try {
            Processor p = AutoProcessor.fromPretrained(dir);
            if (p instanceof ImageProcessor ip) {
                return ip;
            }
            if (p instanceof org.bytedeco.pytorch.llm.transformers.processor.Qwen2VLProcessor q) {
                // Qwen2VLProcessor exposes an image processor internally
                return ImageProcessor.createQwen2VL();
            }
            if (p instanceof org.bytedeco.pytorch.llm.transformers.processor.MiniMaxVLProcessor m) {
                return ImageProcessor.createMiniMaxVL();
            }
        } catch (IOException ignored) {
            // fall through to default
        }
        return ImageProcessor.createImageNet();
    }
}
