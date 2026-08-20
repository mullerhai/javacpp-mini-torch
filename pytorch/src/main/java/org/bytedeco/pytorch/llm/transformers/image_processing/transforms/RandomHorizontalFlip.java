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
import org.bytedeco.pytorch.global.torch;

import java.util.Random;

/**
 * Horizontally flip an image tensor with probability {@code p}.
 *
 * <p>If {@code p == 1.0} the flip is always applied; if {@code p == 0.0}
 * it is never applied. Uses {@link torch#flip(Tensor, long[])} along
 * the width axis (last spatial dimension).
 */
public class RandomHorizontalFlip extends Transform {

    private final double p;
    private final Random rng;

    /**
     * Default flip probability 0.5.
     */
    public RandomHorizontalFlip() {
        this(0.5, new Random());
    }

    /**
     * Explicit flip probability with a new {@link Random}.
     */
    public RandomHorizontalFlip(double p) {
        this(p, new Random());
    }

    /**
     * Explicit flip probability with a caller-supplied {@link Random}.
     */
    public RandomHorizontalFlip(double p, long seed) {
        this(p, new Random(seed));
    }

    private RandomHorizontalFlip(double p, Random rng) {
        if (p < 0 || p > 1) {
            throw new IllegalArgumentException("p must be in [0, 1]");
        }
        this.p = p;
        this.rng = rng != null ? rng : new Random();
    }

    @Override
    public Tensor apply(Tensor t) {
        if (t == null) return null;
        if (rng.nextDouble() < p) {
            // Flip along width axis (last spatial dimension).
            int wDim = t.dim() - 1;
            return torch.flip(t, new long[]{wDim});
        }
        return t;
    }

    public double probability() { return p; }

    @Override
    protected String name() {
        return "RandomHorizontalFlip[p=" + p + "]";
    }
}
