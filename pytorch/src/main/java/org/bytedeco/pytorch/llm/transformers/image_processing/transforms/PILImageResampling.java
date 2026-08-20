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

/**
 * PIL / Pillow-compatible resampling mode constants.
 *
 * <p>This enum mirrors the {@code PIL.Image} resampling attributes
 * ({@code PIL.Image.NEAREST}, {@code PIL.Image.BILINEAR}, etc.) so that
 * Java code that reads a HuggingFace config JSON can map the
 * {@code "resample"} string to a usable integer without depending on
 * the Python PIL package directly.
 *
 * <p>Values are compatible with Pillow's integer constants:
 * <pre>{@code
 * PIL.Image.NEAREST  = 0
 * PIL.Image.LANCZOS  = 1   (called ANTIALIAS in older Pillow)
 * PIL.Image.BILINEAR = 2
 * PIL.Image.BICUBIC  = 3
 * PIL.Image.BOX      = 4  (added in Pillow 5.0)
 * PIL.Image.HAMMING  = 5  (added in Pillow 5.0)
 * }</pre>
 *
 * <p>Note: Pillow LANCZOS = 1 while OpenCV LANCZOS = 6;
 * use {@link InterpolationMode} when you need OpenCV codes.
 */
public enum PILImageResampling {

    /** Nearest-neighbour (fastest). */
    NEAREST(0),

    /**
     * Lanczos windowed sinc. Note: Pillow value = 1 but OpenCV value = 6.
     * Use {@link InterpolationMode#LANCZOS} when calling OpenCV.
     */
    LANCZOS(1),

    /** Bilinear interpolation (default for most HF image processors). */
    BILINEAR(2),

    /** Bicubic interpolation. */
    BICUBIC(3),

    /** Box sampling (good for downscaling). */
    BOX(4),

    /** Hamming windowed sinc (sharp, fast). */
    HAMMING(5);

    private final int value;

    PILImageResampling(int value) {
        this.value = value;
    }

    /** The PIL integer code. */
    public int value() {
        return value;
    }

    /**
     * Convert a PIL integer code to the corresponding enum.
     *
     * @param code PIL resample value
     * @return matching enum, or {@link #NEAREST} if unknown
     */
    public static PILImageResampling from(int code) {
        for (PILImageResampling r : values()) {
            if (r.value == code) return r;
        }
        return NEAREST;
    }

    /**
     * Convert a PIL code to an {@link InterpolationMode}.
     * Handles the LANCZOS value mismatch (PIL=1, OpenCV=6).
     */
    public static InterpolationMode toInterpolationMode(int pilCode) {
        if (pilCode == LANCZOS.value) return InterpolationMode.LANCZOS;
        return InterpolationMode.from(pilCode);
    }
}
