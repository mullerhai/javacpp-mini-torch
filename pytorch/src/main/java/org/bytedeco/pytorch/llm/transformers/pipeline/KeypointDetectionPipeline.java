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
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForKeypointDetection;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("keypoint-detection")}.
 *
 * <p>Detects human body keypoints (e.g. from a MoveNet or OpenPose model) and
 * returns a list of keypoint maps per detected person. Each map contains
 * {@code (key, score)} pairs.
 *
 * <pre>{@code
 * KeypointDetectionPipeline pipe = KeypointDetectionPipeline.fromPretrained(
 *     "google/movenet-multipose", hub);
 * List<Map<String, Object>> keypoints = pipe.call(imageTensor);
 * }</pre>
 */
public final class KeypointDetectionPipeline {

    private final AutoModelForKeypointDetection.Bundle bundle;

    public KeypointDetectionPipeline(AutoModelForKeypointDetection.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static KeypointDetectionPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new KeypointDetectionPipeline(AutoModelForKeypointDetection.fromPretrained(modelId, hub));
    }

    public static KeypointDetectionPipeline fromDirectory(Path dir) throws IOException {
        return new KeypointDetectionPipeline(AutoModelForKeypointDetection.fromDirectory(dir));
    }

    public AutoModelForKeypointDetection.Bundle bundle() {
        return bundle;
    }

    /** Returns a list of keypoint maps (one per detected person). */
    public List<Map<String, Object>> call(Object image) {
        // TODO: route to AutoModelForKeypointDetection.Bundle once a high-level
        // keypoint heatmap decoding entry point is exposed.
        return List.of();
    }

    public void close() {
        try { bundle.model.close(); } catch (Throwable ignored) {}
    }
}
