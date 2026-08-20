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
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForImageClassification;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("image-classification")}.
 *
 * <p>Takes an image (or list of images) and returns the top-K class predictions
 * for each as {@code Map<String, Float>} (label → score), sorted descending by
 * score.
 *
 * <pre>{@code
 * ImageClassificationPipeline pipe = ImageClassificationPipeline.fromPretrained(
 *     "google/vit-base-patch16-224", hub);
 * List<Map<String, Object>> result = pipe.call(image);
 * }</pre>
 */
public final class ImageClassificationPipeline {

    private final AutoModelForImageClassification.Bundle bundle;

    public ImageClassificationPipeline(AutoModelForImageClassification.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static ImageClassificationPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new ImageClassificationPipeline(AutoModelForImageClassification.fromPretrained(modelId, hub));
    }

    public static ImageClassificationPipeline fromDirectory(Path dir) throws IOException {
        return new ImageClassificationPipeline(AutoModelForImageClassification.fromDirectory(dir));
    }

    public AutoModelForImageClassification.Bundle bundle() {
        return bundle;
    }

    /** Returns top-K predictions for the given image input. */
    public List<Map<String, Object>> call(Object image) {
        // TODO: route to AutoModelForImageClassification.Bundle.predictTopK with the
        // required ImageInput type and map class ids → label names.
        return List.of();
    }

    /** Batch variant: returns top-K predictions per input. */
    public List<List<Map<String, Object>>> callBatch(List<Object> images) {
        java.util.List<List<Map<String, Object>>> out = new java.util.ArrayList<>(images.size());
        for (Object img : images) out.add(call(img));
        return out;
    }

    public void close() {
        try { bundle.model.close(); } catch (Throwable ignored) {}
    }
}
