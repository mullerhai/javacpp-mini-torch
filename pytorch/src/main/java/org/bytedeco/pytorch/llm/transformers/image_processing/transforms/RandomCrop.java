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
import java.util.concurrent.ThreadLocalRandom;

/**
 * Randomly crop a patch of the given size from the input image tensor.
 *
 * <p>If the input is smaller than the crop size in either dimension,
 * zero-padding is applied before cropping (mirrors torchvision behaviour).
 */
public class RandomCrop extends Transform {

    private final int height;
    private final int width;
    private final Random rng;

    /**
     * Square random crop with a new {@link Random} instance.
     *
     * @param size crop height and width
     */
    public RandomCrop(int size) {
        this(size, size, new Random());
    }

    /**
     * Random crop with explicit dimensions and a new {@link Random}.
     */
    public RandomCrop(int height, int width) {
        this(height, width, new Random());
    }

    /**
     * Random crop with a caller-supplied {@link Random} for reproducibility.
     */
    public RandomCrop(int height, int width, Random rng) {
        if (height <= 0 || width <= 0) {
            throw new IllegalArgumentException("height and width must be positive");
        }
        this.height = height;
        this.width = width;
        this.rng = rng != null ? rng : new Random();
    }

    @Override
    public Tensor apply(Tensor t) {
        if (t == null) return null;
        int dim = t.dim();
        int spatialOffset = dim - 2;
        long hDim = dim - 2;
        long wDim = dim - 1;
        int h = (int) t.size(hDim);
        int w = (int) t.size(wDim);

        // Zero-pad if image is smaller than the crop size.
        Tensor padded = t;
        int padTop = 0, padLeft = 0;
        if (h < height || w < width) {
            int padH = Math.max(0, height - h);
            int padW = Math.max(0, width - w);
            padTop = padH / 2;
            int padBottom = padH - padTop;
            padLeft = padW / 2;
            int padRight = padW - padLeft;
            padded = torch.constant_pad_nd(t, new long[]{padTop, padBottom, padLeft, padRight}, 0.0);
        }

        // Recompute dimensions after padding.
        h = (int) padded.size(hDim);
        w = (int) padded.size(wDim);

        int top = h > height ? rng.nextInt(h - height + 1) : 0;
        int left = w > width ? rng.nextInt(w - width + 1) : 0;

        // Slice out the crop window.
        return padded.slice(hDim, top, top + height, 1)
                .slice(wDim, left, left + width, 1);
    }

    public int height() { return height; }
    public int width()  { return width; }

    @Override
    protected String name() {
        return "RandomCrop[" + height + "x" + width + "]";
    }
}
