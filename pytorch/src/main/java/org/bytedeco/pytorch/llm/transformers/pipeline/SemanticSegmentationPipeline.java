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
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForSemanticSegmentation;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("image-segmentation")} for semantic segmentation.
 *
 * <p>Returns a list of {@code (label, mask)} pairs per input image — each mask is
 * a {@link org.bytedeco.pytorch.Tensor} of shape {@code [H, W]} with the predicted
 * class per pixel.
 *
 * <pre>{@code
 * SemanticSegmentationPipeline pipe = SemanticSegmentationPipeline.fromPretrained(
 *     "nvidia/segformer-b0-finetuned-ade-512-512", hub);
 * List<Map<String, Object>> segs = pipe.call(image);
 * }</pre>
 */
public final class SemanticSegmentationPipeline {

    private final AutoModelForSemanticSegmentation.Bundle bundle;

    public SemanticSegmentationPipeline(AutoModelForSemanticSegmentation.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static SemanticSegmentationPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new SemanticSegmentationPipeline(AutoModelForSemanticSegmentation.fromPretrained(modelId, hub));
    }

    public static SemanticSegmentationPipeline fromDirectory(Path dir) throws IOException {
        return new SemanticSegmentationPipeline(AutoModelForSemanticSegmentation.fromDirectory(dir));
    }

    public AutoModelForSemanticSegmentation.Bundle bundle() {
        return bundle;
    }

    /** Returns a list of {@code {label, mask}} maps per input image. */
    public List<Map<String, Object>> call(Object image) {
        // TODO: route to AutoModelForSemanticSegmentation.Bundle once a high-level
        // argmax-over-classes entry point is exposed.
        return List.of();
    }

    public List<List<Map<String, Object>>> callBatch(List<Object> images) {
        java.util.List<List<Map<String, Object>>> out = new java.util.ArrayList<>(images.size());
        for (Object img : images) out.add(call(img));
        return out;
    }

    public void close() {
        try { bundle.model.close(); } catch (Throwable ignored) {}
    }
}
