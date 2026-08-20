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

/**
 * Describes the position of the channel dimension in an image tensor.
 *
 * <p>Matches the HF {@code image_utils.ChannelDimension} enum used across
 * ViT, CLIP, DINOv2, and other vision processors to handle both
 * CHW (channels-first) and HWC (channels-last) layouts without ambiguity.
 *
 * <p>Usage when inferring from a tensor shape:
 * <pre>{@code
 * ChannelDimension dim = inferChannelDimension(t);
 * int cDim = dim.axis(t.dim());
 * }</pre>
 */
public enum ChannelDimension {

    /**
     * Channel dimension is the first axis after batch (CHW or NCHW).
     * PyTorch convention.
     */
    FIRST,

    /**
     * Channel dimension is the last axis (HWC or NHWC).
     * NumPy / TensorFlow / OpenCV convention.
     */
    LAST,

    /**
     * Ambiguous or single-channel image; callers must infer from context.
     */
    ANY;

    /**
     * Return the concrete axis index for this dimension given a tensor
     * of {@code n} dimensions.
     *
     * <ul>
     *   <li>{@code FIRST}: returns {@code n > 2 ? 1 : 0}</li>
     *   <li>{@code LAST}:  returns {@code n - 1}</li>
     *   <li>{@code ANY}:  returns {@code -1} (unknown)</li>
     * </ul>
     *
     * @param n total number of dimensions ({@code >= 2})
     */
    public int axis(int n) {
        if (n < 2) return -1;
        switch (this) {
            case FIRST: return n > 2 ? 1 : 0;
            case LAST:  return n - 1;
            case ANY:
            default:    return -1;
        }
    }

    /**
     * Infer the channel dimension from a tensor's shape.
     *
     * <p>Convention:
     * <ul>
     *   <li>3-D tensor: assumes {@code FIRST} (CHW)</li>
     *   <li>4-D tensor: assumes {@code FIRST} (NCHW)</li>
     * </ul>
     * Throws {@link IllegalArgumentException} for 2-D (no channel) tensors.
     */
    public static ChannelDimension fromShape(long[] shape) {
        if (shape == null || shape.length < 3) {
            throw new IllegalArgumentException(
                    "Cannot infer channel dimension from shape of length " +
                    (shape == null ? "null" : shape.length));
        }
        // Default assumption: CHW / NCHW (PyTorch convention)
        return FIRST;
    }

    /**
     * Infer the channel dimension from a tensor.
     *
     * @param t non-null tensor with at least 3 dimensions
     */
    public static ChannelDimension fromTensor(Tensor t) {
        if (t == null) throw new IllegalArgumentException("tensor is null");
        return fromShape(new long[]{
                t.size(0), t.size(1),
                t.size(t.dim() > 2 ? 2 : 1)
        });
    }

    /**
     * Return the dimension string used in HF JSON configs
     * ({@code "channels_first"} or {@code "channels_last"}).
     */
    public String configString() {
        switch (this) {
            case FIRST: return "channels_first";
            case LAST:  return "channels_last";
            default:    return "channels_any";
        }
    }

    /**
     * Parse a HF config string to a {@link ChannelDimension}.
     *
     * @param s string such as {@code "channels_first"}, {@code "channels_last"}
     * @return matching enum, or {@link #ANY} if unknown
     */
    public static ChannelDimension fromConfigString(String s) {
        if ("channels_first".equalsIgnoreCase(s)) return FIRST;
        if ("channels_last".equalsIgnoreCase(s))   return LAST;
        return ANY;
    }
}
