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
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.transformers.image_processing;

import org.bytedeco.pytorch.Tensor;

import java.util.List;

/**
 * HuggingFace {@code image_processing_utils.ImageFeatureExtractionMixin} port.
 *
 * <p>Provides the {@code extractFeatures} entry point used by models that
 * expose intermediate image embeddings rather than (or in addition to)
 * classification logits (e.g. DINOv2, SigLIP, BEiT).
 *
 * <p>Concrete implementations decide how many layers of features to return
 * and whether to include attention maps / patch embeddings.
 */
public interface ImageFeatureExtractionMixin {

    /**
     * Extract intermediate feature representations from a raw image tensor.
     *
     * @param image CHW or NCHW float tensor in {@code [0, 1]}
     * @return list of feature tensors; first element is typically the CLS
     *         token / pooled output, subsequent elements are per-layer
     *         intermediate activations
     */
    List<Tensor> extractFeatures(Tensor image);

    /**
     * Extract a single scalar feature map (convenience for single-output
     * models such as standard ViT).
     *
     * @param image CHW or NCHW float tensor
     * @return the extracted feature tensor
     */
    default Tensor extractFeaturesSingle(Tensor image) {
        List<Tensor> features = extractFeatures(image);
        if (features == null || features.isEmpty()) {
            throw new IllegalStateException("extractFeatures returned empty list");
        }
        return features.get(0);
    }
}
