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
package org.bytedeco.pytorch.llm.transformers.image_processing.utils;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.Objects;

/**
 * Per-channel image normalization: {@code output = (x - mean) / std}.
 *
 * <p>Mirrors the behaviour of {@code torchvision.transforms.Normalize} and
 * HF {@code image_utils.normalize}. Both inputs are broadcast over the
 * spatial dimensions so the same call works for CHW, NCHW, HWC, and NHWC
 * tensors — provided the channel axis is correctly specified.
 *
 * <p>For most HF vision models (ViT, CLIP, DINOv2) the default mean/std
 * are:
 * <pre>{@code
 * mean = [0.485, 0.456, 0.406]
 * std  = [0.229, 0.224, 0.225]
 * }</pre>
 *
 * <p>For ImageNet-standard normalization (CLIP-style):
 * <pre>{@code
 * mean = [0.48145466, 0.4578275, 0.40821073]
 * std  = [0.26862954, 0.26130258, 0.27577711]
 * }</pre>
 */
public final class ImageNormalizer {

    private ImageNormalizer() {}

    /**
     * Normalize a CHW / NCHW image tensor: {@code (x - mean) / std}.
     *
     * <p>The mean and std arrays are reshaped to {@code (1, C, 1, 1)} so they
     * broadcast correctly over N (batch) and H/W dimensions.
     *
     * @param t    input tensor; must be at least 3-dimensional with a 3-element channel axis
     * @param mean channel means (length 3 for RGB)
     * @param std  channel standard deviations (length 3 for RGB)
     * @return normalized tensor; the input is NOT closed
     */
    public static Tensor normalize(Tensor t, float[] mean, float[] std) {
        return normalize(t, mean, std, ChannelDimension.FIRST);
    }

    /**
     * Normalize with explicit channel dimension.
     */
    public static Tensor normalize(Tensor t, float[] mean, float[] std,
                                   ChannelDimension channelDim) {
        Objects.requireNonNull(t, "tensor");
        Objects.requireNonNull(mean, "mean");
        Objects.requireNonNull(std, "std");
        Objects.requireNonNull(channelDim, "channelDim");
        if (mean.length != std.length) {
            throw new IllegalArgumentException("mean and std must have the same length");
        }
        if (mean.length != 3) {
            throw new IllegalArgumentException(
                    "Expected 3 channels (RGB), got " + mean.length);
        }

        int cDim = channelDim.axis(t.dim());
        if (cDim < 0) {
            throw new IllegalArgumentException(
                    "Cannot determine channel axis for tensor with " + t.dim() + " dims");
        }
        long c = t.size(cDim);
        if (c != mean.length) {
            throw new IllegalArgumentException(
                    "Channel dimension has " + c + " channels, but mean/std have " +
                    mean.length + " entries");
        }

        // Build (1, C, 1, 1) broadcast tensors — works for CHW, NCHW, etc.
        long[] meanShape = makeShape(t.dim(), cDim, mean.length);
        long[] stdShape  = makeShape(t.dim(), cDim, mean.length);

        Tensor meanT = torch.tensor(mean).reshape(meanShape).to(t.scalar_type());
        Tensor stdT  = torch.tensor(std).reshape(stdShape).to(t.scalar_type());
        Tensor result = t.sub(meanT).div(stdT);
        meanT.close();
        stdT.close();
        return result;
    }

    /**
     * Convenience: normalize using double arrays (cast to float internally).
     */
    public static Tensor normalize(Tensor t, double[] mean, double[] std,
                                   ChannelDimension channelDim) {
        float[] fm = new float[mean.length];
        float[] fs = new float[std.length];
        for (int i = 0; i < mean.length; i++) {
            fm[i] = (float) mean[i];
            fs[i] = (float) std[i];
        }
        return normalize(t, fm, fs, channelDim);
    }

    // ---------------------------------------------------------------------
    // Preset configurations
    // ---------------------------------------------------------------------

    /** Standard ImageNet mean/std. */
    public static float[] imagenetMean() {
        return new float[]{0.485f, 0.456f, 0.406f};
    }

    public static float[] imagenetStd() {
        return new float[]{0.229f, 0.224f, 0.225f};
    }

    /** CLIP / HuggingFace default mean/std. */
    public static float[] clipMean() {
        return new float[]{0.48145466f, 0.4578275f, 0.40821073f};
    }

    public static float[] clipStd() {
        return new float[]{0.26862954f, 0.26130258f, 0.27577711f};
    }

    /** Simple [0.5, 0.5, 0.5] normalize. */
    public static Tensor normalizeHalf(Tensor t) {
        return normalize(t, DEFAULT_MEAN, DEFAULT_STD, ChannelDimension.FIRST);
    }

    private static final float[] DEFAULT_MEAN = {0.5f, 0.5f, 0.5f};
    private static final float[] DEFAULT_STD  = {0.5f, 0.5f, 0.5f};

    // ---------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------

    /**
     * Build a shape array of length {@code n} with value {@code v} at index {@code cDim}.
     * All other positions are filled with {@code 1}.
     * Example: makeShape(4, 1, 3) → [1, 3, 1, 1]
     */
    private static long[] makeShape(int n, int cDim, int c) {
        long[] shape = new long[n];
        for (int i = 0; i < n; i++) {
            shape[i] = (i == cDim) ? c : 1;
        }
        return shape;
    }
}
