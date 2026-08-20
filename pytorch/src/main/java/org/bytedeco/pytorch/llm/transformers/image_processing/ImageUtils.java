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
import org.bytedeco.pytorch.global.torch;

import java.util.Objects;
import org.bytedeco.pytorch.LongOptional;

/**
 * Static utility methods for image tensors.
 *
 * <p>Mirrors the helpers found in
 * {@code transformers.image_processing_utils} and
 * {@code torchvision.utils}.
 */
public final class ImageUtils {

    private ImageUtils() {}

    // ---------------------------------------------------------------------
    // Validation
    // ---------------------------------------------------------------------

    /**
     * Return {@code true} if {@code t} has a valid image shape
     * (at least 2 dimensions, all dims > 0, channels in {1, 3, 4}).
     */
    public static boolean isValidImage(Tensor t) {
        if (t == null) return false;
        if (t.dim() < 2) return false;
        if (t.numel() == 0) return false;
        long h = t.size(t.dim() - 2);
        long w = t.size(t.dim() - 1);
        if (h <= 0 || w <= 0) return false;
        if (t.dim() >= 3) {
            long c = t.size(t.dim() - 3);
            if (c != 1 && c != 3 && c != 4) return false;
        }
        return true;
    }

    // ---------------------------------------------------------------------
    // Channel manipulation
    // ---------------------------------------------------------------------

    /**
     * Ensure the tensor has 3 channels by replicating a single channel
     * or dropping an alpha channel.
     *
     * <p>If {@code t} is CHW with C=1 → returns C=3 (R=G=B).
     * If {@code t} is CHW with C=4 → returns C=3 (drops alpha).
     * If {@code t} is NCHW with C=1 or C=4 → same per-batch.
     * If already C=3 → returns the tensor unchanged.
     */
    public static Tensor convertToRGB(Tensor t) {
        Objects.requireNonNull(t, "t");
        if (t.dim() < 3) return t;
        long c = t.size(t.dim() - 3);
        if (c == 3) return t;
        if (c == 1) {
            // Replicate gray channel 3 times.
            Tensor gray = t.squeeze(LongOptional.of(t.dim() - 3));
            return torch.stack(gray, gray, gray, /*dim=*/0);
        }
        if (c == 4) {
            // Drop alpha.
            return t.slice(t.dim() - 3, 0, 3, 1);
        }
        return t;
    }

    /**
     * Normalize a single channel along the inferred channel axis:
     * {@code result = (x - mean) / std}.
     *
     * @param t     input tensor (must have a channel dimension)
     * @param axis  the axis that represents channels ({@code t.dim() - 3} for CHW,
     *              {@code t.dim() - 1} for HWC). Pass {@code -1} to auto-detect.
     */
    public static Tensor normalizeChannel(Tensor t, int axis) {
        return normalizeChannel(t, axis, 0.5f, 0.5f);
    }

    /**
     * Normalize a single channel with custom mean/std.
     */
    public static Tensor normalizeChannel(Tensor t, int axis, float mean, float std) {
        Objects.requireNonNull(t, "t");
        if (std == 0f) throw new IllegalArgumentException("std must be non-zero");
        int effectiveAxis = resolveChannelAxis(t, axis);
        long c = t.size(effectiveAxis);
        if (c != 1) {
            throw new IllegalArgumentException(
                    "normalizeChannel expects single-channel tensor, got C=" + c);
        }
        return t.sub(new org.bytedeco.pytorch.Scalar(mean))
                .div(new org.bytedeco.pytorch.Scalar(std));
    }

    /**
     * Infer which axis is the channel axis based on tensor shape conventions.
     *
     * <ul>
     *   <li>CHW  → axis 0</li>
     *   <li>NCHW → axis 1</li>
     *   <li>HWC  → axis 2</li>
     *   <li>NHWC → axis 3</li>
     * </ul>
     *
     * @param t non-null tensor
     * @return inferred channel axis, or {@code fallback} if ambiguous
     */
    public static int inferChannelAxis(Tensor t, int fallback) {
        Objects.requireNonNull(t, "t");
        int dim = t.dim();
        if (dim == 3) return 0;          // CHW
        if (dim == 4) return 1;          // NCHW
        if (dim == 2) return fallback;   // HW — ambiguous
        return fallback;
    }

    /** Convenience: infer with fallback = -1 (auto mode). */
    public static int inferChannelAxis(Tensor t) {
        return inferChannelAxis(t, -1);
    }

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    private static int resolveChannelAxis(Tensor t, int axis) {
        if (axis >= 0) return axis;
        int inferred = inferChannelAxis(t);
        if (inferred < 0) {
            throw new IllegalArgumentException(
                    "Cannot infer channel axis for tensor of dim " + t.dim() +
                    ". Please specify axis explicitly.");
        }
        return inferred;
    }

    /** Convert tensor to float type without changing values. */
    public static Tensor toFloat(Tensor t) {
        return t.to(torch.ScalarType.Float);
    }

    /** Rescale tensor from [0, 255] to [0, 1]. */
    public static Tensor rescale01(Tensor t) {
        return t.div(new org.bytedeco.pytorch.Scalar(255.0));
    }

    /** Rescale tensor from [0, 1] to [0, 255] and convert to byte. */
    public static Tensor toUint8(Tensor t) {
        return t.mul(new org.bytedeco.pytorch.Scalar(255.0))
                .clamp(0.0, 255.0)
                .to(torch.ScalarType.Byte);
    }
}
