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
package org.bytedeco.pytorch.llm.transformers.image_processing.transforms;

import org.bytedeco.pytorch.Tensor;

/**
 * Base class for all image-processing transforms operating on {@link Tensor}s.
 *
 * <p>Mirrors the {@code torchvision.transforms.Transform} contract and
 * HF {@code image_transforms} design: a transform is a pure function
 * {@code Tensor → Tensor} that may optionally keep state (for random
 * augmentations).
 *
 * <p>Implementations should honour the contract of {@link #then(Transform)}
 * so that pipelines compose cleanly.
 */
public abstract class Transform {

    /**
     * Apply this transform to the input image tensor.
     *
     * @param t input tensor, typically CHW or NCHW float in {@code [0, 1]}
     * @return transformed tensor; must not be {@code null}
     */
    public abstract Tensor apply(Tensor t);

    /**
     * Return a new transform that first applies {@code this} then {@code other}.
     *
     * <p>Composition is right-associative:
     * {@code a.then(b).then(c).apply(x) == c.apply(b.apply(a.apply(x)))}
     */
    public Transform then(Transform other) {
        if (other == null) return this;
        Transform self = this;
        return new Transform() {
            @Override
            public Tensor apply(Tensor t) {
                return other.apply(self.apply(t));
            }
        };
    }

    /**
     * Alias for {@link #apply(Tensor)} matching the torchvision naming.
     */
    public Tensor forward(Tensor t) {
        return apply(t);
    }

    /** Human-readable name used by {@link #toString()}. */
    protected String name() {
        return getClass().getSimpleName();
    }

    @Override
    public String toString() {
        return name();
    }
}
