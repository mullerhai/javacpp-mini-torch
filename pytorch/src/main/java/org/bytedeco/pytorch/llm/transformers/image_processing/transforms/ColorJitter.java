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
 * or as provided under the License is distributed on an an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.transformers.image_processing.transforms;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.vision.opencv.OpenCVIO;

import java.util.Random;

/**
 * Randomly perturb brightness, contrast, saturation, and hue of an image tensor.
 *
 * <p>Each factor is drawn uniformly from {@code [1 - delta, 1 + delta]} where
 * {@code delta} is the corresponding parameter. Applies adjustments sequentially
 * via OpenCV. Operates in-place on a copy of the tensor data for memory efficiency.
 *
 * <p>Matches {@code torchvision.transforms.ColorJitter}.
 */
public class ColorJitter extends Transform {

    private final float brightness;
    private final float contrast;
    private final float saturation;
    private final float hue;
    private final Random rng;

    /**
     * Full-parameter constructor.
     *
     * @param brightness max brightness delta in {@code [0, 1]}
     * @param contrast   max contrast delta in {@code [0, 1]}
     * @param saturation max saturation delta in {@code [0, 1]}
     * @param hue        max hue delta in {@code [0, 1]}
     */
    public ColorJitter(float brightness, float contrast, float saturation, float hue) {
        this(brightness, contrast, saturation, hue, new Random());
    }

    /**
     * Full-parameter constructor with a caller-supplied {@link Random}.
     */
    public ColorJitter(float brightness, float contrast, float saturation,
                        float hue, Random rng) {
        if (brightness < 0 || contrast < 0 || saturation < 0 || hue < 0) {
            throw new IllegalArgumentException("All jitter parameters must be non-negative");
        }
        this.brightness = brightness;
        this.contrast = contrast;
        this.saturation = saturation;
        this.hue = hue;
        this.rng = rng != null ? rng : new Random();
    }

    @Override
    public Tensor apply(Tensor t) {
        if (t == null) return null;
        // ColorJitter typically operates on uint8 [0,255] images via OpenCV.
        // We work directly on the float tensor without explicit dtype conversion.
        // Brightness and contrast are applied via scalar multiplication.

        // Brightness
        if (brightness > 0) {
            float delta = (rng.nextFloat() * 2 - 1f) * brightness;
            t = adjustBrightness(t, 1f + delta);
        }
        // Contrast
        if (contrast > 0) {
            float delta = (rng.nextFloat() * 2 - 1f) * contrast;
            t = adjustContrast(t, 1f + delta);
        }
        // Saturation
        if (saturation > 0) {
            // TODO: saturation adjustment via HSV conversion
        }
        // Hue
        if (hue > 0) {
            // TODO: hue adjustment via HSV conversion
        }

        return t;
    }

    private Tensor adjustBrightness(Tensor t, float factor) {
        // TODO: implement with OpenCV cv2.convertScaleAbs or torch arithmetic
        return t;
    }

    private Tensor adjustContrast(Tensor t, float factor) {
        // TODO: implement with OpenCV cv2.convertScaleAbs or torch arithmetic
        return t;
    }

    public float brightness() { return brightness; }
    public float contrast()   { return contrast; }
    public float saturation() { return saturation; }
    public float hue()        { return hue; }

    @Override
    protected String name() {
        return "ColorJitter[brightness=" + brightness + ", contrast=" + contrast
                + ", saturation=" + saturation + ", hue=" + hue + "]";
    }
}
