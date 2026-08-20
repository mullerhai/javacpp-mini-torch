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
import org.bytedeco.pytorch.vision.opencv.OpenCVIO;

import java.util.Objects;

/**
 * Crop the centre of an image tensor to the given height and width.
 *
 * <p>Input may be CHW or NCHW; the crop is applied to the spatial dimensions.
 */
public class CenterCrop extends Transform {

    private final int height;
    private final int width;

    /**
     * Square centre crop.
     *
     * @param size crop height and width
     */
    public CenterCrop(int size) {
        this(size, size);
    }

    /**
     * Centre crop to explicit dimensions.
     *
     * @param height crop height
     * @param width  crop width
     */
    public CenterCrop(int height, int width) {
        if (height <= 0 || width <= 0) {
            throw new IllegalArgumentException("height and width must be positive");
        }
        this.height = height;
        this.width = width;
    }

    @Override
    public Tensor apply(Tensor t) {
        Objects.requireNonNull(t, "tensor");
        return OpenCVIO.centerCrop(t, height, width);
    }

    public int height() { return height; }
    public int width()  { return width; }

    @Override
    protected String name() {
        return "CenterCrop[" + height + "x" + width + "]";
    }
}
