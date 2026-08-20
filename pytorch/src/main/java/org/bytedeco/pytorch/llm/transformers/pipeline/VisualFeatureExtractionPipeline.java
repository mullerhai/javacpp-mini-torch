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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForImageClassification;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("visual-feature-extraction")} (a.k.a. "image-to-feature" or
 * "embedding").
 *
 * <p>Runs a vision encoder in {@code eval()} mode and returns the per-patch or
 * pooled visual embedding tensor. Useful for retrieval, similarity search, or
 * as input to a downstream head.
 *
 * <pre>{@code
 * VisualFeatureExtractionPipeline pipe = VisualFeatureExtractionPipeline.fromPretrained(
 *     "google/siglip-base-patch16-224", hub);
 * Tensor features = pipe.call(image);
 * }</pre>
 */
public final class VisualFeatureExtractionPipeline {

    private final AutoModelForImageClassification.Bundle bundle;

    public VisualFeatureExtractionPipeline(AutoModelForImageClassification.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static VisualFeatureExtractionPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new VisualFeatureExtractionPipeline(AutoModelForImageClassification.fromPretrained(modelId, hub));
    }

    public static VisualFeatureExtractionPipeline fromDirectory(Path dir) throws IOException {
        return new VisualFeatureExtractionPipeline(AutoModelForImageClassification.fromDirectory(dir));
    }

    public AutoModelForImageClassification.Bundle bundle() {
        return bundle;
    }

    /** Returns the visual embedding tensor for the input image. */
    public Tensor call(Object image) {
        // TODO: forward the processed image through the ViT / CLIP vision encoder and
        // return the pooled or per-patch hidden states as the embedding tensor.
        return null;
    }

    public void close() {
        try { bundle.model.close(); } catch (Throwable ignored) {}
    }
}
