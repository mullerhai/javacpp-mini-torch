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
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForPoseEstimation;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("pose-estimation")}.
 *
 * <p>Detects people and their full body pose (keypoints + optionally segmentation masks).
 * Returns a list of per-person maps containing keypoints, scores, and optionally a
 * segmentation mask.
 *
 * <pre>{@code
 * PoseEstimationPipeline pipe = PoseEstimationPipeline.fromPretrained(
 *     "DavidLiSpan/rt-detr-kapao", hub);
 * List<Map<String, Object>> poses = pipe.call(image);
 * }</pre>
 */
public final class PoseEstimationPipeline {

    private final AutoModelForPoseEstimation.Bundle bundle;

    public PoseEstimationPipeline(AutoModelForPoseEstimation.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static PoseEstimationPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new PoseEstimationPipeline(AutoModelForPoseEstimation.fromPretrained(modelId, hub));
    }

    public static PoseEstimationPipeline fromDirectory(Path dir) throws IOException {
        return new PoseEstimationPipeline(AutoModelForPoseEstimation.fromDirectory(dir));
    }

    public AutoModelForPoseEstimation.Bundle bundle() {
        return bundle;
    }

    /** Returns a list of per-person pose maps containing keypoints, scores, and optional mask. */
    public List<Map<String, Object>> call(Object image) {
        // TODO: route to AutoModelForPoseEstimation.Bundle once a high-level
        // pose-decoding entry point is exposed.
        return List.of();
    }

    public void close() {
        try { bundle.model.close(); } catch (Throwable ignored) {}
    }
}
