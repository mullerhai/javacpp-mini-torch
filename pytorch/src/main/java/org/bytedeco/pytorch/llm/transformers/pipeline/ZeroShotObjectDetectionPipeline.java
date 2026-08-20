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
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForZeroShotObjectDetection;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("zero-shot-object-detection")}.
 *
 * <p>Takes an image, a list of candidate labels (text prompts) and a confidence
 * threshold; returns detections ({@code boxes}, {@code labels}, {@code scores})
 * matching the prompts above the threshold. Typically driven by GroundingDINO or
 * Owlv2.
 *
 * <pre>{@code
 * ZeroShotObjectDetectionPipeline pipe = ZeroShotObjectDetectionPipeline.fromPretrained(
 *     "google/owlv2-base-patch16-ensemble", hub);
 * List<Map<String, Object>> dets = pipe.call(image, List.of("cat", "dog"), 0.3f);
 * }</pre>
 */
public final class ZeroShotObjectDetectionPipeline {

    private final AutoModelForZeroShotObjectDetection.Bundle bundle;

    public ZeroShotObjectDetectionPipeline(AutoModelForZeroShotObjectDetection.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static ZeroShotObjectDetectionPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new ZeroShotObjectDetectionPipeline(AutoModelForZeroShotObjectDetection.fromPretrained(modelId, hub));
    }

    public static ZeroShotObjectDetectionPipeline fromDirectory(Path dir) throws IOException {
        return new ZeroShotObjectDetectionPipeline(AutoModelForZeroShotObjectDetection.fromDirectory(dir));
    }

    public AutoModelForZeroShotObjectDetection.Bundle bundle() {
        return bundle;
    }

    /** Returns a list of {@code {boxes, labels, scores}} detections above the threshold. */
    public List<Map<String, Object>> call(Object image, List<String> candidateLabels, float threshold) {
        // TODO: route to AutoModelForZeroShotObjectDetection.Bundle once a high-level
        // text-conditioned detection entry point is exposed.
        return List.of();
    }

    public void close() {
        try { bundle.model.close(); } catch (Throwable ignored) {}
    }
}
