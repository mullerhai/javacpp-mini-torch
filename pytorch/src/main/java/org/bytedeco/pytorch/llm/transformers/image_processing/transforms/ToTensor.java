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
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.vision.utils.ImageTensors;
import org.bytedeco.pytorch.global.torch;

import java.awt.image.BufferedImage;
import java.util.Objects;

/**
 * Convert a generic image object to a float CHW tensor in {@code [0, 1]}.
 *
 * <p>Supported input types:
 * <ul>
 *   <li>{@link BufferedImage} — converted via {@link ImageTensors#toTensor}</li>
 *   <li>{@link ImageData} — converted to {@link BufferedImage} then to tensor</li>
 *   <li>{@link Tensor} — returned unchanged (assumed already CHW float)</li>
 * </ul>
 *
 * <p>Mirrors the torchvision {@code ToTensor} transform.
 */
public class ToTensor extends Transform {

    /**
     * Convert any supported image type to a CHW float tensor in {@code [0, 1]}.
     */
    @Override
    public Tensor apply(Tensor t) {
        // If already a float tensor in [0,1], return as-is.
        if (t != null) return t;
        throw new IllegalArgumentException(
                "ToTensor does not accept null; use a supported image type " +
                "(BufferedImage, ImageData, or a non-null Tensor)");
    }

    /**
     * Convert a {@link BufferedImage} to a CHW float tensor.
     */
    public static Tensor fromBufferedImage(BufferedImage image) {
        Objects.requireNonNull(image, "image");
        Tensor t = ImageTensors.toTensor(image);
        return t;
    }

    /**
     * Convert an {@link ImageData} to a CHW float tensor.
     */
    public static Tensor fromImageData(ImageData image) {
        Objects.requireNonNull(image, "image");
        BufferedImage bi = image.toBufferedImage();
        return fromBufferedImage(bi);
    }

    /**
     * Convert a generic object to a tensor if possible.
     * Throws {@link IllegalArgumentException} for unsupported types.
     */
    public static Tensor fromObject(Object o) {
        if (o instanceof Tensor t) return t;
        if (o instanceof BufferedImage bi) return fromBufferedImage(bi);
        if (o instanceof ImageData id) return fromImageData(id);
        throw new IllegalArgumentException(
                "Unsupported image type: " + (o == null ? "null" : o.getClass().getName()));
    }

    @Override
    protected String name() {
        return "ToTensor";
    }
}
