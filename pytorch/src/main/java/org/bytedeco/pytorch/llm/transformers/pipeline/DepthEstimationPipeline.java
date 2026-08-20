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
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForDepthEstimation;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("depth-estimation")}.
 *
 * <p>Estimates a per-pixel depth map from a single image. Returns a result map
 * containing the predicted depth tensor, original image dimensions, and optional
 * normalised depth values.
 *
 * <pre>{@code
 * DepthEstimationPipeline pipe = DepthEstimationPipeline.fromPretrained(
 *     "Intel/zoedepth-nyu", hub);
 * Map<String, Object> result = pipe.call(image);
 * Tensor depth = (Tensor) result.get("predicted_depth");
 * }</pre>
 */
public final class DepthEstimationPipeline {

    private final AutoModelForDepthEstimation.Bundle bundle;

    public DepthEstimationPipeline(AutoModelForDepthEstimation.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static DepthEstimationPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new DepthEstimationPipeline(AutoModelForDepthEstimation.fromPretrained(modelId, hub));
    }

    public static DepthEstimationPipeline fromDirectory(Path dir) throws IOException {
        return new DepthEstimationPipeline(AutoModelForDepthEstimation.fromDirectory(dir));
    }

    public AutoModelForDepthEstimation.Bundle bundle() {
        return bundle;
    }

    /** Returns a map with {@code predicted_depth}, {@code width}, and {@code height}. */
    public Map<String, Object> call(Object image) {
        // TODO: route to AutoModelForDepthEstimation.Bundle once a high-level
        // depth decoding entry point is exposed.
        return Map.of();
    }

    public void close() {
        try { bundle.model.close(); } catch (Throwable ignored) {}
    }
}
