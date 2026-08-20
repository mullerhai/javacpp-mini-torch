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

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.transformers.utils.T;
import org.bytedeco.pytorch.vision.opencv.OpenCVIO;
import org.bytedeco.pytorch.global.torch;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * HuggingFace {@code image_processing_utils.BaseImageProcessor} port.
 *
 * <p>Holds the canonical preprocessing configuration that every HF vision
 * processor derives from: target {@code size}, {@code resample} filter,
 * {@code do_resize}/{@code do_normalize} flags, and the {@code image_mean}
 * / {@code image_std} triple used to standardize pixel values.
 *
 * <p>Subclasses (e.g. {@link ImageProcessor}, ViT, CLIP) refine the
 * {@link #process(Tensor, Map)} method but inherit the helper pipeline
 * implemented here.
 */
public abstract class BaseImageProcessor implements ImageProcessingMixin {

    /** Default resample value (2 = bilinear). */
    public static final int DEFAULT_RESAMPLE = 2;

    /** Default normalization mean (HuggingFace ImageNet-style placeholder). */
    public static final float[] DEFAULT_MEAN = {0.5f, 0.5f, 0.5f};
    /** Default normalization std. */
    public static final float[] DEFAULT_STD = {0.5f, 0.5f, 0.5f};

    protected Map<String, Integer> size;
    protected int resample;
    protected boolean doResize;
    protected boolean doNormalize;
    protected float[] imageMean;
    protected float[] imageStd;

    protected BaseImageProcessor() {
        this(new LinkedHashMap<>(), DEFAULT_RESAMPLE, true, true,
                DEFAULT_MEAN.clone(), DEFAULT_STD.clone());
    }

    protected BaseImageProcessor(Map<String, Integer> size,
                                  int resample,
                                  boolean doResize,
                                  boolean doNormalize,
                                  float[] imageMean,
                                  float[] imageStd) {
        this.size = size == null ? new LinkedHashMap<>() : new LinkedHashMap<>(size);
        this.resample = resample;
        this.doResize = doResize;
        this.doNormalize = doNormalize;
        this.imageMean = imageMean == null ? DEFAULT_MEAN.clone() : imageMean.clone();
        this.imageStd = imageStd == null ? DEFAULT_STD.clone() : imageStd.clone();
    }

    // ---------------------------------------------------------------------
    // Getters / setters (HF-style)
    // ---------------------------------------------------------------------

    public Map<String, Integer> size() { return Collections.unmodifiableMap(size); }
    public int resample() { return resample; }
    public boolean doResize() { return doResize; }
    public boolean doNormalize() { return doNormalize; }
    public float[] imageMean() { return imageMean.clone(); }
    public float[] imageStd() { return imageStd.clone(); }

    public void size(Map<String, Integer> size) {
        this.size = new LinkedHashMap<>(size);
    }
    public void resample(int resample) { this.resample = resample; }
    public void doResize(boolean v) { this.doResize = v; }
    public void doNormalize(boolean v) { this.doNormalize = v; }
    public void imageMean(float[] v) { this.imageMean = v.clone(); }
    public void imageStd(float[] v) { this.imageStd = v.clone(); }

    // ---------------------------------------------------------------------
    // Helpers called from the subclass process() pipeline
    // ---------------------------------------------------------------------

    /**
     * Resize the input tensor to the configured {@code size} using
     * {@link OpenCVIO#resize(Tensor, int, int)}.
     */
    public Tensor resize(Tensor image) {
        Objects.requireNonNull(image, "image");
        if (!doResize) return image;
        int h = size.getOrDefault("height", 224);
        int w = size.getOrDefault("width", h);
        return OpenCVIO.resize(image, h, w);
    }

    /**
     * Center-crop to the configured {@code size} after a resize. Mirrors
     * HF {@code center_crop} behaviour for square targets.
     */
    public Tensor centerCrop(Tensor image) {
        Objects.requireNonNull(image, "image");
        int h = size.getOrDefault("height", 224);
        int w = size.getOrDefault("width", h);
        return OpenCVIO.centerCrop(image, h, w);
    }

    /**
     * Standard HF normalize step: {@code (x - mean) / std} per channel.
     *
     * <p>Operates on CHW (or NCHW) float tensors. Reshapes the mean/std
     * broadcasts to {@code (1, C, 1, 1)} so a batch of N images is handled.
     */
    public Tensor normalize(Tensor input) {
        if (!doNormalize) return input;
        Tensor mean = torch.tensor(imageMean).reshape(1, imageMean.length, 1, 1)
                .to(input.scalar_type());
        Tensor std = torch.tensor(imageStd).reshape(1, imageStd.length, 1, 1)
                .to(input.scalar_type());
        Tensor result = input.sub(mean).div(std);
        mean.close();
        std.close();
        return result;
    }

    /**
     * Convert an image tensor to the final NCHW {@code pixel_values} layout,
     * applying the resize + crop + normalize pipeline. Returns a new tensor;
     * the input is not closed.
     */
    public Tensor to_pixel_values(Tensor image) {
        Objects.requireNonNull(image, "image");
        Tensor out = resize(image);
        out = centerCrop(out);
        out = normalize(out);
        // Ensure leading batch dim.
        if (out.dim() == 3) {
            out = out.unsqueeze(0);
        }
        return out;
    }

    // ---------------------------------------------------------------------
    // Convenience constants
    // ---------------------------------------------------------------------

    /** ImageNet mean (standard for many HF vision models). */
    public static float[] imagenetMean() {
        return new float[]{0.485f, 0.456f, 0.406f};
    }

    /** ImageNet std. */
    public static float[] imagenetStd() {
        return new float[]{0.229f, 0.224f, 0.225f};
    }

    /** Whether {@code x} should be treated as a valid image tensor. */
    public static boolean isValid(Tensor x) {
        return x != null && x.dim() >= 2 && x.numel() > 0;
    }

    /** Build the channel-axis mean broadcast tensor (helper used by subclasses). */
    public Tensor meanTensor(Tensor prototype) {
        return torch.tensor(imageMean).reshape(1, imageMean.length, 1, 1)
                .to(prototype.scalar_type());
    }

    /** Build the channel-axis std broadcast tensor (helper used by subclasses). */
    public Tensor stdTensor(Tensor prototype) {
        return torch.tensor(imageStd).reshape(1, imageStd.length, 1, 1)
                .to(prototype.scalar_type());
    }

    /** Convert an int to a float scalar for arithmetic. */
    public static Scalar fScalar(float v) {
        return new Scalar(v);
    }

    /** Convert mean/std arrays to tensors in a single pass (for caching). */
    public Tensor[] normalizationTensors(Tensor prototype) {
        return new Tensor[]{meanTensor(prototype), stdTensor(prototype)};
    }
}
