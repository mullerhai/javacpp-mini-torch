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
package org.bytedeco.pytorch.llm.transformers.trainer.utils;

/**
 * Label smoothing utility mirroring HF's {@code transformers.label_smoothing}.
 *
 * <p>Distributes label mass uniformly over all classes to reduce over-confidence.
 */
public final class label_smoothing {

    private final float epsilon;

    /**
     * @param epsilon smoothing factor in [0, 1]; 0 = no smoothing
     */
    public label_smoothing(float epsilon) {
        if (epsilon < 0 || epsilon > 1) {
            throw new IllegalArgumentException("epsilon must be in [0, 1], got: " + epsilon);
        }
        this.epsilon = epsilon;
    }

    /**
     * Apply label smoothing to a batch of one-hot or integer labels.
     *
     * @param labels  integer label ids (shape {@code [batch]}) or one-hot ({@code [batch, numClasses]})
     * @param numClasses  number of classes (used when labels are integer ids)
     * @return smoothed label distribution
     */
    public float[] _distribute_scalar_labels(long[] labels, int numClasses) {
        // Stub: real implementation would distribute (1 - epsilon) to the true class
        // and epsilon / (numClasses - 1) to the others.
        // For now, return uniform distribution.
        float uniform = epsilon / Math.max(1, numClasses - 1);
        float[] out = new float[numClasses];
        for (int i = 0; i < out.length; i++) out[i] = uniform;
        return out;
    }

    public float epsilon() { return epsilon; }
}
