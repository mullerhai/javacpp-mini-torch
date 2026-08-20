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

import java.nio.file.Path;

/**
 * HF {@code AutoFeatureExtractor.from_pretrained} entry point.
 *
 * <p>Wraps generic audio feature extractors (e.g. {@code ASTFeatureExtractor},
 * {@code Wav2Vec2FeatureExtractor}). This stub delegates to the audio processor
 * once the feature-extractor wiring is in place.
 */
public final class AutoFeatureExtractor {

    private AutoFeatureExtractor() {}

    /**
     * Load a feature extractor from a HuggingFace Hub model id.
     *
     * @param <T>  concrete feature-extractor type
     * @param id   HF model id
     * @return loaded extractor
     */
    public static <T> T fromPretrained(String id) {
        throw new UnsupportedOperationException(
                "Audio feature extractor not yet wired; " +
                "use AutoProcessor or AudioProcessor directly.");
    }

    /**
     * Load a feature extractor from a local directory.
     *
     * @param <T>  concrete feature-extractor type
     * @param dir  local directory
     * @return loaded extractor
     */
    public static <T> T fromDirectory(Path dir) {
        throw new UnsupportedOperationException(
                "Audio feature extractor not yet wired; " +
                "use AutoProcessor or AudioProcessor directly.");
    }
}
