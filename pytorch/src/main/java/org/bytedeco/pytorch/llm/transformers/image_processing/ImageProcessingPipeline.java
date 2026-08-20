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
import org.bytedeco.pytorch.llm.transformers.image_processing.transforms.Transform;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A composable image-processing pipeline that applies a list of
 * {@link Transform}s sequentially.
 *
 * <p>Modelled after {@code torchvision.transforms.Compose} and the
 * HF {@code ImageProcessingMixin} pipeline pattern.
 *
 * <pre>{@code
 * ImageProcessingPipeline pipeline = new ImageProcessingPipeline(List.of(
 *     new Resize(256, 256, InterpolationMode.BILINEAR.value()),
 *     new CenterCrop(224, 224),
 *     new Normalize(new float[]{0.485f, 0.456f, 0.406f},
 *                  new float[]{0.229f, 0.224f, 0.225f})
 * ));
 * Tensor pixelValues = pipeline.apply(rawImage);
 * }</pre>
 */
public class ImageProcessingPipeline {

    private final List<Transform> transforms;

    /**
     * Build a pipeline from the given transforms.
     *
     * @param transforms non-null list; each transform is applied in order
     */
    public ImageProcessingPipeline(List<Transform> transforms) {
        Objects.requireNonNull(transforms, "transforms");
        this.transforms = Collections.unmodifiableList(new ArrayList<>(transforms));
    }

    /** Convenience varargs overload. */
    public ImageProcessingPipeline(Transform... transforms) {
        this(List.of(transforms));
    }

    /**
     * Apply the entire transform list to the input tensor.
     *
     * @param t input image tensor (CHW or NCHW)
     * @return transformed tensor
     */
    public Tensor apply(Tensor t) {
        Tensor result = t;
        for (Transform transform : transforms) {
            result = transform.apply(result);
        }
        return result;
    }

    /** Alias matching HF Python naming. */
    public Tensor forward(Tensor t) {
        return apply(t);
    }

    /** Read-only view of the pipeline transforms. */
    public List<Transform> transforms() {
        return transforms;
    }

    /**
     * Return a new pipeline that appends {@code other} after the current list.
     * The new pipeline is immutable; neither original pipeline is modified.
     */
    public ImageProcessingPipeline then(Transform other) {
        List<Transform> combined = new ArrayList<>(transforms);
        combined.add(Objects.requireNonNull(other, "other"));
        return new ImageProcessingPipeline(combined);
    }

    /**
     * Return a new pipeline that concatenates {@code second} after the current list.
     * Useful for composing two pre-built pipelines.
     */
    public ImageProcessingPipeline then(ImageProcessingPipeline second) {
        Objects.requireNonNull(second, "second");
        List<Transform> combined = new ArrayList<>(transforms);
        combined.addAll(second.transforms);
        return new ImageProcessingPipeline(combined);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("ImageProcessingPipeline[");
        for (int i = 0; i < transforms.size(); i++) {
            if (i > 0) sb.append(" -> ");
            sb.append(transforms.get(i).getClass().getSimpleName());
        }
        sb.append(']');
        return sb.toString();
    }
}
