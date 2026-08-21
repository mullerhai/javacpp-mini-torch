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
package org.bytedeco.pytorch.llm.transformers.pipeline;

import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForMaskedImageModeling;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("masked-image-modeling")}.
 *
 * <p>Takes an image, corrupts it with random masking, and predicts the missing
 * pixel / latent values. Returns top-K reconstruction candidates with confidence
 * scores.
 *
 * <pre>{@code
 * MaskedImageModelingPipeline pipe = MaskedImageModelingPipeline.fromPretrained(
 *     "facebook/vit-masked-latent-patch16", hub);
 * Map<String, Object> result = pipe.call(image, 5);
 * }</pre>
 */
public final class MaskedImageModelingPipeline {

    private final AutoModelForMaskedImageModeling.Bundle bundle;

    public MaskedImageModelingPipeline(AutoModelForMaskedImageModeling.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static MaskedImageModelingPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new MaskedImageModelingPipeline(AutoModelForMaskedImageModeling.fromPretrained(modelId, hub));
    }

    public static MaskedImageModelingPipeline fromDirectory(Path dir) throws IOException {
        return new MaskedImageModelingPipeline(AutoModelForMaskedImageModeling.fromDirectory(dir));
    }

    public AutoModelForMaskedImageModeling.Bundle bundle() {
        return bundle;
    }

    /** Returns a map containing predicted masks and per-patch confidence scores. */
    public Map<String, Object> call(Object image, int topK) {
        // TODO: apply random masking, forward the corrupted image through the MAE / VQ-VAE
        // model, and return {masks, scores} for the top-K predicted patches.
        return Map.of();
    }

    public void close() {
        try { bundle.model.close(); } catch (Throwable ignored) {}
    }
}
