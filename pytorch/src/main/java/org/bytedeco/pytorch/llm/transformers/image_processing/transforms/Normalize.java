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

import java.util.Objects;

/**
 * Per-channel normalization: {@code output = (x - mean) / std}.
 *
 * <p>Mean and std are 3-element arrays for the R, G, B channels.
 * They are broadcast to the tensor shape so both CHW and NCHW tensors
 * are supported.
 */
public class Normalize extends Transform {

    private final float[] mean;
    private final float[] std;

    /**
     * Build with the given mean and std arrays.
     *
     * @param mean channel means
     * @param std  channel standard deviations
     */
    public Normalize(float[] mean, float[] std) {
        Objects.requireNonNull(mean, "mean");
        Objects.requireNonNull(std, "std");
        if (mean.length != std.length) {
            throw new IllegalArgumentException("mean and std must have the same length");
        }
        this.mean = mean.clone();
        this.std = std.clone();
    }

    /**
     * Build with the given mean and std doubles (cast to float).
     */
    public Normalize(double[] mean, double[] std) {
        Objects.requireNonNull(mean, "mean");
        Objects.requireNonNull(std, "std");
        if (mean.length != std.length) {
            throw new IllegalArgumentException("mean and std must have the same length");
        }
        this.mean = new float[mean.length];
        this.std = new float[std.length];
        for (int i = 0; i < mean.length; i++) {
            this.mean[i] = (float) mean[i];
            this.std[i] = (float) std[i];
        }
    }

    @Override
    public Tensor apply(Tensor t) {
        Objects.requireNonNull(t, "tensor");
        Tensor meanT = torch.tensor(mean).reshape(1, mean.length, 1, 1)
                .to(t.scalar_type());
        Tensor stdT = torch.tensor(std).reshape(1, std.length, 1, 1)
                .to(t.scalar_type());
        Tensor result = t.sub(meanT).div(stdT);
        meanT.close();
        stdT.close();
        return result;
    }

    public float[] mean() { return mean.clone(); }
    public float[] std()  { return std.clone(); }

    @Override
    protected String name() {
        return "Normalize[mean=" + arr(mean) + ", std=" + arr(std) + "]";
    }

    private static String arr(float[] a) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < a.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.4f", a[i]));
        }
        return sb.append(']').toString();
    }
}
