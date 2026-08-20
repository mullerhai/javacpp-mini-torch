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
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForZeroShotImageClassification;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("zero-shot-image-classification")}.
 *
 * <p>Takes an image and a list of free-form candidate labels (typically via a
 * CLIP-style dual encoder) and returns labels ranked by similarity score.
 *
 * <pre>{@code
 * ZeroShotImageClassificationPipeline pipe = ZeroShotImageClassificationPipeline.fromPretrained(
 *     "openai/clip-vit-base-patch32", hub);
 * List<Map<String, Object>> out = pipe.call(image, List.of("a cat", "a dog"));
 * }</pre>
 */
public final class ZeroShotImageClassificationPipeline {

    private final AutoModelForZeroShotImageClassification.Bundle bundle;

    public ZeroShotImageClassificationPipeline(AutoModelForZeroShotImageClassification.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static ZeroShotImageClassificationPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new ZeroShotImageClassificationPipeline(AutoModelForZeroShotImageClassification.fromPretrained(modelId, hub));
    }

    public static ZeroShotImageClassificationPipeline fromDirectory(Path dir) throws IOException {
        return new ZeroShotImageClassificationPipeline(AutoModelForZeroShotImageClassification.fromDirectory(dir));
    }

    public AutoModelForZeroShotImageClassification.Bundle bundle() {
        return bundle;
    }

    /** Returns labels ranked by similarity to the image. */
    public List<Map<String, Object>> call(Object image, List<String> candidateLabels) {
        // TODO: compute image embedding, encode each candidate, and return ranked
        // {label, score} pairs sorted descending by score.
        return List.of();
    }

    public void close() {
        try { bundle.model.close(); } catch (Throwable ignored) {}
    }
}
