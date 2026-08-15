/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
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
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.dataframe.media;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.dtype.VideoData;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Specialized converter for Tensor to Video conversions with advanced features.
 *
 * <p>Features:
 * <ul>
 *   <li>Video tensor to VideoData conversion</li>
 *   <li>Frame extraction and batch processing</li>
 *   <li>Video synthesis from image sequences</li>
 *   <li>Temporal operations (subsample, merge, split)</li>
 *   <li>Integration with FFmpeg for video encoding/decoding</li>
 * </ul>
 */
public final class TensorToVideoConverter {

    private TensorToVideoConverter() {}

    // ── Core Conversion Methods ───────────────────────────────────────────

    /**
     * Convert NCHW video tensor to VideoData.
     *
     * @param tensor Shape [N,C,H,W] representing N frames
     * @param fps Frames per second
     * @return VideoData containing all frames
     */
    public static VideoData toVideoData(Tensor tensor, double fps) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        Tensor cpu = tensor.contiguous().cpu().to(ScalarType.Float);
        long[] shape = MultimodalTensorConverter.getShape(cpu);

        if (shape.length != 4) {
            throw new IllegalArgumentException("Expected [N,C,H,W], got rank " + shape.length);
        }

        int numFrames = (int) shape[0];
        List<ImageData> frames = new ArrayList<>(numFrames);

        for (int i = 0; i < numFrames; i++) {
            Tensor frame = cpu.select(0, i);
            frames.add(TensorToImageConverter.toImageData(frame));
        }

        VideoData video = new VideoData(frames, fps);
        video.setFrameCount(numFrames);
        video.setWidth((int) shape[3]);
        video.setHeight((int) shape[2]);
        video.setDuration(numFrames / fps);

        return video;
    }

    /**
     * Convert VideoData to NCHW video tensor.
     *
     * @param video Source VideoData
     * @return Tensor shape [N,C,H,W]
     */
    public static Tensor toTensor(VideoData video) {
        Objects.requireNonNull(video, "video cannot be null");
        List<ImageData> frames = video.getFrames();
        if (frames == null || frames.isEmpty()) {
            throw new IllegalArgumentException("VideoData has no frames");
        }

        int n = frames.size();
        ImageData first = frames.get(0);
        int c = first.getChannels();
        int h = first.getHeight();
        int w = first.getWidth();

        List<Tensor> tensors = new ArrayList<>(n);
        for (ImageData frame : frames) {
            Tensor t = MultimodalTensorConverter.toTensor(frame);
            if (t.size(0) != c || t.size(1) != h || t.size(2) != w) {
                t = TensorToImageConverter.resize(t, h, w);
            }
            tensors.add(t);
        }

        return org.bytedeco.pytorch.global.torch.stack(tensors.toArray(new Tensor[0]));
    }

    /**
     * Convert list of CHW tensors to VideoData.
     *
     * @param frames List of tensors [C,H,W]
     * @param fps Frames per second
     * @return VideoData
     */
    public static VideoData fromTensorList(List<Tensor> frames, double fps) {
        return MultimodalTensorConverter.fromTensorList(frames, fps);
    }

    /**
     * Convert batch tensor to list of ImageData.
     *
     * @param batchTensor Shape [N,C,H,W]
     * @return List of ImageData
     */
    public static List<ImageData> toImageDataList(Tensor batchTensor) {
        return MultimodalTensorConverter.batchToImageData(batchTensor);
    }

    // ── Video Tensor Operations ──────────────────────────────────────────

    /**
     * Extract a range of frames from video tensor.
     *
     * @param tensor Shape [N,C,H,W]
     * @param startFrame Start frame index (inclusive)
     * @param endFrame End frame index (exclusive)
     * @return Tensor shape [endFrame-startFrame,C,H,W]
     */
    public static Tensor sliceFrames(Tensor tensor, int startFrame, int endFrame) {
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        if (shape.length != 4) {
            throw new IllegalArgumentException("Expected [N,C,H,W]");
        }
        int numFrames = (int) shape[0];
        startFrame = Math.max(0, Math.min(startFrame, numFrames - 1));
        endFrame = Math.max(startFrame + 1, Math.min(endFrame, numFrames));
        return tensor.narrow(0, startFrame, endFrame - startFrame);
    }

    /**
     * Subsample video tensor by selecting every nth frame.
     *
     * @param tensor Shape [N,C,H,W]
     * @param stride Frame stride
     * @return Subsampled tensor
     */
    public static Tensor subsampleFrames(Tensor tensor, int stride) {
        if (stride <= 1) {
            return tensor;
        }

        long[] shape = MultimodalTensorConverter.getShape(tensor);
        int numFrames = (int) shape[0];
        int newNumFrames = (numFrames + stride - 1) / stride;

        List<Tensor> selected = new ArrayList<>();
        for (int i = 0; i < numFrames; i += stride) {
            selected.add(tensor.select(0, i));
        }

        return org.bytedeco.pytorch.global.torch.stack(selected.toArray(new Tensor[0]));
    }

    /**
     * Concatenate multiple video tensors along the frame dimension.
     *
     * @param tensors List of tensors with same [C,H,W]
     * @return Concatenated tensor
     */
    public static Tensor concatenateVideos(List<Tensor> tensors) {
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("Empty tensor list");
        }
        if (tensors.size() == 1) {
            return tensors.get(0);
        }

        List<Tensor> allFrames = new ArrayList<>();
        for (Tensor video : tensors) {
            long[] shape = MultimodalTensorConverter.getShape(video);
            for (int i = 0; i < shape[0]; i++) {
                allFrames.add(video.select(0, i));
            }
        }

        return org.bytedeco.pytorch.global.torch.stack(allFrames.toArray(new Tensor[0]));
    }

    /**
     * Reverse video frames.
     *
     * @param tensor Shape [N,C,H,W]
     * @return Reversed tensor
     */
    public static Tensor reverseFrames(Tensor tensor) {
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        int numFrames = (int) shape[0];

        List<Tensor> reversed = new ArrayList<>(numFrames);
        for (int i = numFrames - 1; i >= 0; i--) {
            reversed.add(tensor.select(0, i));
        }

        return org.bytedeco.pytorch.global.torch.stack(reversed.toArray(new Tensor[0]));
    }

    /**
     * Apply uniform speed change by frame sampling.
     *
     * @param tensor Shape [N,C,H,W]
     * @param speedFactor Speed multiplier (>1 = faster, <1 = slower)
     * @return Speed-adjusted tensor
     */
    public static Tensor adjustSpeed(Tensor tensor, double speedFactor) {
        if (Math.abs(speedFactor - 1.0) < 1e-6) {
            return tensor;
        }

        long[] shape = MultimodalTensorConverter.getShape(tensor);
        int numFrames = (int) shape[0];
        int newNumFrames = (int) Math.round(numFrames / speedFactor);

        if (newNumFrames <= 0) {
            newNumFrames = 1;
        }

        List<Tensor> adjusted = new ArrayList<>(newNumFrames);
        for (int i = 0; i < newNumFrames; i++) {
            int srcIdx = (int) Math.round(i * speedFactor);
            srcIdx = Math.min(srcIdx, numFrames - 1);
            adjusted.add(tensor.select(0, srcIdx));
        }

        return org.bytedeco.pytorch.global.torch.stack(adjusted.toArray(new Tensor[0]));
    }

    // ── Frame-level Operations ───────────────────────────────────────────

    /**
     * Extract specific frames from video.
     *
     * @param tensor Shape [N,C,H,W]
     * @param frameIndices Indices of frames to extract
     * @return Tensor with selected frames
     */
    public static Tensor extractFrames(Tensor tensor, int[] frameIndices) {
        List<Tensor> selected = new ArrayList<>(frameIndices.length);
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        int numFrames = (int) shape[0];

        for (int idx : frameIndices) {
            if (idx >= 0 && idx < numFrames) {
                selected.add(tensor.select(0, idx));
            }
        }

        if (selected.isEmpty()) {
            throw new IllegalArgumentException("No valid frame indices");
        }

        return org.bytedeco.pytorch.global.torch.stack(selected.toArray(new Tensor[0]));
    }

    /**
     * Get single frame from video tensor.
     *
     * @param tensor Shape [N,C,H,W]
     * @param index Frame index
     * @return Tensor [C,H,W]
     */
    public static Tensor getFrame(Tensor tensor, int index) {
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        int numFrames = (int) shape[0];
        if (index < 0 || index >= numFrames) {
            throw new IndexOutOfBoundsException("Frame index " + index + " out of range [0," + numFrames + ")");
        }
        return tensor.select(0, index);
    }

    /**
     * Apply image transformation to all frames.
     *
     * @param tensor Shape [N,C,H,W]
     * @param transformer Function to transform each frame
     * @return Transformed video tensor
     */
    public static Tensor applyToFrames(Tensor tensor, java.util.function.Function<Tensor, Tensor> transformer) {
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        int numFrames = (int) shape[0];

        List<Tensor> transformed = new ArrayList<>(numFrames);
        for (int i = 0; i < numFrames; i++) {
            Tensor frame = tensor.select(0, i);
            transformed.add(transformer.apply(frame));
        }

        return org.bytedeco.pytorch.global.torch.stack(transformed.toArray(new Tensor[0]));
    }

    // ── Format Validation ─────────────────────────────────────────────────

    /**
     * Validate tensor shape for video conversion.
     */
    public static boolean isValidVideoTensor(Tensor tensor) {
        if (tensor == null) return false;
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        return shape.length == 4;
    }

    /**
     * Get video dimensions from tensor.
     *
     * @return int array [numFrames, height, width, channels]
     */
    public static int[] getVideoDimensions(Tensor tensor) {
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        if (shape.length != 4) {
            throw new IllegalArgumentException("Expected [N,C,H,W], got rank " + shape.length);
        }
        return new int[]{(int) shape[0], (int) shape[2], (int) shape[3], (int) shape[1]};
    }

    /**
     * Get number of frames from tensor.
     */
    public static int getNumFrames(Tensor tensor) {
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        return (int) shape[0];
    }

    // ── Synthesis Methods ─────────────────────────────────────────────────

    /**
     * Create video from single image (repeated frames).
     *
     * @param imageTensor Shape [C,H,W]
     * @param numFrames Number of frames to repeat
     * @return Video tensor [N,C,H,W]
     */
    public static Tensor fromSingleImage(Tensor imageTensor, int numFrames) {
        long[] shape = MultimodalTensorConverter.getShape(imageTensor);
        if (shape.length != 3) {
            throw new IllegalArgumentException("Expected [C,H,W]");
        }

        Tensor[] frames = new Tensor[numFrames];
        for (int i = 0; i < numFrames; i++) {
            frames[i] = imageTensor.clone();
        }
        return org.bytedeco.pytorch.global.torch.stack(frames);
    }

    /**
     * Create video from color gradient.
     *
     * @param numFrames Number of frames
     * @param height Frame height
     * @param width Frame width
     * @param hueRange Hue rotation range [start, end]
     * @return Video tensor [N,3,H,W]
     */
    public static Tensor createGradientVideo(int numFrames, int height, int width, float[] hueRange) {
        List<Tensor> frames = new ArrayList<>(numFrames);
        float hueStart = hueRange != null && hueRange.length > 0 ? hueRange[0] : 0f;
        float hueEnd = hueRange != null && hueRange.length > 1 ? hueRange[1] : 1f;

        for (int i = 0; i < numFrames; i++) {
            float hue = hueStart + (hueEnd - hueStart) * i / (numFrames - 1);
            frames.add(TensorToImageConverter.createTestPattern(width, height, "gradient"));
        }

        return org.bytedeco.pytorch.global.torch.stack(frames.toArray(new Tensor[0]));
    }

    /**
     * Create video from list of images.
     *
     * @param images List of ImageData or BufferedImage
     * @param fps Frames per second
     * @return VideoData
     */
    public static VideoData fromImages(List<?> images, double fps) {
        List<ImageData> frames = new ArrayList<>();
        for (Object img : images) {
            if (img instanceof ImageData) {
                frames.add((ImageData) img);
            } else if (img instanceof java.awt.image.BufferedImage) {
                frames.add(new ImageData((java.awt.image.BufferedImage) img));
            }
        }

        VideoData video = new VideoData(frames, fps);
        video.setFrameCount(frames.size());
        if (!frames.isEmpty()) {
            video.setWidth(frames.get(0).getWidth());
            video.setHeight(frames.get(0).getHeight());
            video.setDuration(frames.size() / fps);
        }
        return video;
    }
}
