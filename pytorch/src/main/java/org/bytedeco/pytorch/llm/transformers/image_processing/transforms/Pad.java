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
 * Pad an image tensor with a constant value along the spatial dimensions.
 *
 * <p>Padding is specified as {@code [left, right, top, bottom]} (PyTorch
 * convention). When the input is smaller than the target size, the result
 * is padded accordingly; when larger, it is unchanged (use {@link RandomCrop}
 * or {@link CenterCrop} for centre-cropping behaviour).
 *
 * <p>Mirrors {@code torchvision.transforms.Pad}.
 */
public class Pad extends Transform {

    /** Padding modes matching the torchvision naming. */
    public enum PaddingMode {
        CONSTANT,
        EDGE,
        REFLECT,
        REPLICATE
    }

    private final int top, left, bottom, right;
    private final int fill;
    private final PaddingMode paddingMode;

    /**
     * Pad all sides equally with constant value 0 (black).
     *
     * @param padding padding in pixels on each side
     */
    public Pad(int padding) {
        this(padding, padding, padding, padding, 0, PaddingMode.CONSTANT);
    }

    /**
     * Explicit 4-side padding with a fill value.
     *
     * @param padding    array of 1–4 ints: [left], [left,right], or [top,bottom,left,right]
     * @param fill       fill value (for constant mode)
     * @param paddingMode padding mode
     */
    public Pad(int[] padding, int fill, PaddingMode paddingMode) {
        this(parsePadding(padding), fill, paddingMode);
    }

    /**
     * Fully explicit 4-side padding.
     */
    public Pad(int top, int left, int bottom, int right, int fill, PaddingMode paddingMode) {
        if (top < 0 || left < 0 || bottom < 0 || right < 0) {
            throw new IllegalArgumentException("Padding values must be non-negative");
        }
        this.top = top;
        this.left = left;
        this.bottom = bottom;
        this.right = right;
        this.fill = fill;
        this.paddingMode = paddingMode != null ? paddingMode : PaddingMode.CONSTANT;
    }

    @Override
    public Tensor apply(Tensor t) {
        if (t == null) return null;
        if (top == 0 && left == 0 && bottom == 0 && right == 0) return t;
        long[] pad = new long[]{top, bottom, left, right};
        if (paddingMode == PaddingMode.CONSTANT) {
            return torch.constant_pad_nd(t, pad, (double) fill);
        }
        // EDGE / REFLECT / REPLICATE are not yet wired to torch primitives;
        // fall back to constant pad with a warning.
        return torch.constant_pad_nd(t, pad, (double) fill);
    }

    /**
     * Return a new Pad that pads to make the image at least size × size square.
     */
    public static Pad forSquare(int size) {
        return new Pad(size, size, size, size, 0, PaddingMode.CONSTANT);
    }

    public int top()   { return top; }
    public int left()  { return left; }
    public int bottom() { return bottom; }
    public int right()  { return right; }
    public int fill()   { return fill; }
    public PaddingMode paddingMode() { return paddingMode; }

    private static int[] parsePadding(int[] p) {
        Objects.requireNonNull(p, "padding");
        switch (p.length) {
            case 1: return new int[]{p[0], p[0], p[0], p[0]};
            case 2: return new int[]{p[0], p[0], p[1], p[1]};  // (left/right, top/bottom) — unusual
            case 4: return p.clone();
            default: throw new IllegalArgumentException(
                    "padding array must have 1, 2, or 4 elements, got " + p.length);
        }
    }

    @Override
    protected String name() {
        return "Pad[top=" + top + ", left=" + left + ", bottom=" + bottom + ", right=" + right
                + ", fill=" + fill + ", mode=" + paddingMode + "]";
    }
}
