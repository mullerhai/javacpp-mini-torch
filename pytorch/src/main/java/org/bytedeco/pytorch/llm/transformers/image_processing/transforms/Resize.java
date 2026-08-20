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
 * Resize an image tensor to the given height and width using OpenCV.
 *
 * <p>Wraps {@link OpenCVIO#resize(Tensor, int, int)}. Input may be CHW
 * or NCHW; the output preserves the same layout with updated spatial dims.
 */
public class Resize extends Transform {

    private final int height;
    private final int width;
    private final int interpolation;

    /**
     * Square resize with default bilinear interpolation.
     *
     * @param size target width and height
     */
    public Resize(int size) {
        this(size, size, InterpolationMode.BILINEAR.value());
    }

    /**
     * Resize to explicit dimensions with default bilinear interpolation.
     */
    public Resize(int height, int width) {
        this(height, width, InterpolationMode.BILINEAR.value());
    }

    /**
     * Resize to explicit dimensions with the given interpolation mode value.
     *
     * @param height        target height in pixels
     * @param width         target width in pixels
     * @param interpolation interpolation mode ordinal ({@code InterpolationMode.value()})
     */
    public Resize(int height, int width, int interpolation) {
        if (height <= 0 || width <= 0) {
            throw new IllegalArgumentException("height and width must be positive");
        }
        this.height = height;
        this.width = width;
        this.interpolation = interpolation;
    }

    @Override
    public Tensor apply(Tensor t) {
        Objects.requireNonNull(t, "tensor");
        return OpenCVIO.resize(t, height, width);
    }

    public int height() { return height; }
    public int width()  { return width; }
    public int interpolation() { return interpolation; }

    @Override
    protected String name() {
        return "Resize[" + height + "x" + width + "]";
    }
}
