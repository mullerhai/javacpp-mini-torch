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

import java.util.Map;

/**
 * HuggingFace {@code image_processing_utils.ImageProcessingMixin} port.
 *
 * <p>Mirrors the public surface used by {@code BaseImageProcessor} subclasses
 * such as {@code CLIPImageProcessor}, {@code ViTImageProcessor} and
 * {@code DINOv2ImageProcessor}: a single {@link #process(Tensor, Map)}
 * entry point that returns the model-ready {@code pixel_values} tensor.
 */
public interface ImageProcessingMixin {

    /**
     * Apply the configured preprocessing pipeline to a single image tensor.
     *
     * @param image  image as a CHW float tensor (typical), NHWC also supported
     * @param kwargs optional overrides (e.g. {@code do_normalize=false})
     * @return processed {@code pixel_values} tensor, typically NCHW
     */
    Tensor process(Tensor image, Map<String, Object> kwargs);

    /** Convenience overload with no kwargs. */
    default Tensor process(Tensor image) {
        return process(image, java.util.Collections.emptyMap());
    }
}