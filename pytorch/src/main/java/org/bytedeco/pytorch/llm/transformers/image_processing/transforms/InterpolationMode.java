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
 * Interpolation modes used by image resizing transforms.
 *
 * <p>Values match {@code torchvision.transforms.InterpolationMode} and the
 * underlying OpenCV integer codes used by {@code cv2.resize}.
 *
 * <p>Usage:
 * <pre>{@code
 * Resize resize = new Resize(256, 256, InterpolationMode.BILINEAR.value());
 * }</pre>
 */
public enum InterpolationMode {

    /** Nearest-neighbour (fastest, aliased). */
    NEAREST(0),

    /** High-quality nearest neighbour (pixel-area, anti-aliased). */
    NEAREST_EXACT(1),

    /** Bilinear interpolation (default in most HF processors). */
    BILINEAR(2),

    /** Bicubic interpolation (smoother, slower). */
    BICUBIC(3),

    /** Area-based interpolation (best for downscaling). */
    BOX(4),

    /** Hamming windowed sinc (fast, sharper than bilinear). */
    HAMMING(5),

    /** Lanczos windowed sinc (highest quality, slowest). */
    LANCZOS(6);

    private final int value;

    InterpolationMode(int value) {
        this.value = value;
    }

    /** The integer code used by OpenCV / PIL resample functions. */
    public int value() {
        return value;
    }

    /**
     * Resolve a raw integer code to an {@link InterpolationMode}.
     *
     * @param code integer value
     * @return the corresponding enum constant, or {@link #NEAREST} if unknown
     */
    public static InterpolationMode from(int code) {
        for (InterpolationMode m : values()) {
            if (m.value == code) return m;
        }
        return NEAREST;
    }

    /** Return the mode name in lowercase (e.g. {@code "bilinear"}). */
    public String nameLower() {
        return name().toLowerCase();
    }
}
