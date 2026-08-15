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
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.dtype.VideoData;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.vision.utils.ImageTensors;
import org.bytedeco.pytorch.audio.utils.AudioTensors;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Objects;

/**
 * Unified multimodal tensor converter for converting between PyTorch Tensors
 * and various media formats (images, audio, video).
 *
 * <p>This class provides enterprise-grade conversion utilities with support for:
 * <ul>
 *   <li>Tensor to ImageData/BufferedImage conversion (CHW/NCHW formats)</li>
 *   <li>Tensor to AudioData/waveform conversion (channels-first layout)</li>
 *   <li>Tensor to VideoData/frame sequence conversion (NCHW tensor)</li>
 *   <li>Bidirectional conversion with automatic format detection</li>
 *   <li>Batch processing support for efficient pipeline operations</li>
 * </ul>
 *
 * <pre>{@code
 * // Image conversion
 * Tensor imageTensor = torch.randn(3, 224, 224);
 * BufferedImage bi = MultimodalTensorConverter.toBufferedImage(imageTensor);
 * ImageData imgData = MultimodalTensorConverter.toImageData(imageTensor);
 *
 * // Audio conversion
 * Tensor waveform = torch.randn(2, 16000);  // [channels, samples]
 * AudioData audio = MultimodalTensorConverter.toAudioData(waveform, 16000);
 *
 * // Video conversion
 * Tensor videoTensor = torch.randn(16, 3, 224, 224);  // [N, C, H, W]
 * VideoData video = MultimodalTensorConverter.toVideoData(videoTensor, 30.0);
 * }</pre>
 */
public final class MultimodalTensorConverter {

    private MultimodalTensorConverter() {}

    // ── Image Conversion ────────────────────────────────────────────────────

    /**
     * Convert a CHW/NCHW tensor to BufferedImage.
     *
     * @param tensor Shape [C,H,W] or [N,C,H,W] (first batch item is used)
     * @return RGB BufferedImage (or grayscale if C=1)
     * @throws IllegalArgumentException if tensor shape is invalid
     */
    public static BufferedImage toBufferedImage(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return ImageTensors.toBufferedImage(tensor);
    }

    /**
     * Convert a CHW/NCHW tensor to ImageData.
     *
     * @param tensor Shape [C,H,W] or [N,C,H,W]
     * @return ImageData containing the converted image
     */
    public static ImageData toImageData(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        BufferedImage bi = toBufferedImage(tensor);
        return new ImageData(bi);
    }

    /**
     * Convert ImageData to CHW tensor in [0,1] range.
     *
     * @param image Source ImageData
     * @return Tensor shape [C,H,W] float32 in [0,1]
     */
    public static Tensor toTensor(ImageData image) {
        Objects.requireNonNull(image, "image cannot be null");
        return MediaBridge.imageToTensor(image);
    }

    /**
     * Convert BufferedImage to CHW tensor in [0,1] range.
     *
     * @param image Source BufferedImage
     * @return Tensor shape [C,H,W] float32 in [0,1]
     */
    public static Tensor toTensor(BufferedImage image) {
        Objects.requireNonNull(image, "image cannot be null");
        return ImageTensors.toTensor(image);
    }

    /**
     * Convert tensor to OpenCV-style [0,255] range.
     *
     * @param tensor Input tensor [0,1] range
     * @return Tensor scaled to [0,255]
     */
    public static Tensor toRange255(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return tensor.mul(new org.bytedeco.pytorch.Scalar(255.0));
    }

    /**
     * Convert tensor from [0,255] to [0,1] range.
     *
     * @param tensor Input tensor [0,255] range
     * @return Tensor scaled to [0,1]
     */
    public static Tensor toRange01(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return tensor.div(new org.bytedeco.pytorch.Scalar(255.0));
    }

    // ── Audio Conversion ────────────────────────────────────────────────────

    /**
     * Convert waveform tensor to AudioData.
     *
     * @param tensor Shape [T] (mono), [C,T] (multi-channel), or [B,C,T] (batch)
     * @param sampleRate Sample rate in Hz
     * @return AudioData with decoded samples
     */
    public static AudioData toAudioData(Tensor tensor, int sampleRate) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return AudioTensors.toAudioData(tensor, sampleRate);
    }

    /**
     * Convert AudioData to waveform tensor [C,T].
     *
     * @param audio Source AudioData
     * @return Tensor shape [C,T] or [T] for mono
     */
    public static Tensor toTensor(AudioData audio) {
        Objects.requireNonNull(audio, "audio cannot be null");
        return AudioTensors.toTensor(audio);
    }

    /**
     * Convert waveform tensor to float array.
     *
     * @param tensor Waveform tensor
     * @return Interleaved float samples
     */
    public static float[] toFloatArray(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return AudioTensors.fromTensor(tensor);
    }

    /**
     * Create waveform tensor from float samples.
     *
     * @param samples Interleaved float samples
     * @param channels Number of channels
     * @return Tensor shape [C,T]
     */
    public static Tensor fromFloatArray(float[] samples, int channels) {
        Objects.requireNonNull(samples, "samples cannot be null");
        return AudioTensors.toTensor(samples, channels);
    }

    /**
     * Infer channel count from waveform tensor shape.
     *
     * @param tensor Waveform tensor
     * @return Number of channels
     */
    public static int inferAudioChannels(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return AudioTensors.inferChannels(tensor);
    }

    /**
     * Infer time samples from waveform tensor shape.
     *
     * @param tensor Waveform tensor
     * @return Number of time samples
     */
    public static int inferAudioTime(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return AudioTensors.inferTime(tensor);
    }

    // ── Video Conversion ────────────────────────────────────────────────────

    /**
     * Convert NCHW tensor to VideoData.
     *
     * @param tensor Shape [N,C,H,W] representing N frames
     * @param fps Frames per second for the video
     * @return VideoData containing extracted frames
     */
    public static VideoData toVideoData(Tensor tensor, double fps) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return MediaBridge.tensorToVideoData(tensor, fps);
    }

    /**
     * Convert VideoData to NCHW tensor [N,C,H,W].
     *
     * @param video Source VideoData
     * @return Tensor shape [N,C,H,W]
     */
    public static Tensor toTensor(VideoData video) {
        Objects.requireNonNull(video, "video cannot be null");
        return MediaBridge.videoToTensor(video);
    }

    /**
     * Convert list of ImageData to NCHW tensor.
     *
     * @param frames List of image frames
     * @return Tensor shape [N,C,H,W]
     */
    public static Tensor toTensor(List<ImageData> frames) {
        Objects.requireNonNull(frames, "frames cannot be null");
        return MediaBridge.stackImages(frames);
    }

    /**
     * Convert list of tensors to VideoData.
     *
     * @param tensors List of CHW tensors
     * @param fps Frames per second
     * @return VideoData
     */
    public static VideoData fromTensorList(List<Tensor> tensors, double fps) {
        Objects.requireNonNull(tensors, "tensors cannot be null");
        List<ImageData> frames = tensors.stream()
                .map(MultimodalTensorConverter::toImageData)
                .toList();
        VideoData video = new VideoData(frames, fps);
        if (!frames.isEmpty()) {
            video.setWidth(frames.get(0).getWidth());
            video.setHeight(frames.get(0).getHeight());
            video.setFrameCount(frames.size());
            video.setDuration(frames.size() / fps);
        }
        return video;
    }

    // ── Batch Operations ───────────────────────────────────────────────────

    /**
     * Convert batch of tensors to list of BufferedImages.
     *
     * @param batchTensors Shape [N,C,H,W]
     * @return List of BufferedImages
     */
    public static List<BufferedImage> batchToBufferedImage(Tensor batchTensors) {
        Objects.requireNonNull(batchTensors, "batchTensors cannot be null");
        Tensor cpu = batchTensors.contiguous().cpu().to(ScalarType.Float);
        long[] shape = cpu.sizes().stream().mapToLong(Long::longValue).toArray();

        if (shape.length != 4) {
            throw new IllegalArgumentException("Expected [N,C,H,W], got rank " + shape.length);
        }

        int n = (int) shape[0];
        List<BufferedImage> results = new java.util.ArrayList<>(n);

        for (int i = 0; i < n; i++) {
            Tensor frame = cpu.select(0, i);
            results.add(toBufferedImage(frame));
        }
        return results;
    }

    /**
     * Convert batch of tensors to list of ImageData.
     *
     * @param batchTensors Shape [N,C,H,W]
     * @return List of ImageData
     */
    public static List<ImageData> batchToImageData(Tensor batchTensors) {
        Objects.requireNonNull(batchTensors, "batchTensors cannot be null");
        return batchToBufferedImage(batchTensors).stream()
                .map(ImageData::new)
                .toList();
    }

    // ── Utility Methods ─────────────────────────────────────────────────────

    /**
     * Get tensor shape as long array.
     *
     * @param tensor Input tensor
     * @return Shape array
     */
    public static long[] getShape(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        long ndim = tensor.dim();
        long[] shape = new long[(int) ndim];
        for (int i = 0; i < ndim; i++) {
            shape[i] = tensor.size(i);
        }
        return shape;
    }

    /**
     * Check if tensor represents a single image [C,H,W].
     */
    public static boolean isImageShape(long[] shape) {
        return shape.length == 3;
    }

    /**
     * Check if tensor represents a batch of images [N,C,H,W].
     */
    public static boolean isBatchImageShape(long[] shape) {
        return shape.length == 4;
    }

    /**
     * Check if tensor represents audio waveform [T] or [C,T].
     */
    public static boolean isAudioShape(long[] shape) {
        return shape.length == 1 || shape.length == 2;
    }

    /**
     * Check if tensor represents video [N,C,H,W].
     */
    public static boolean isVideoShape(long[] shape) {
        return shape.length == 4;
    }

    /**
     * Detect tensor modality from shape.
     *
     * @return "image", "audio", "video", or "unknown"
     */
    public static String detectModality(Tensor tensor) {
        long[] shape = getShape(tensor);
        if (isImageShape(shape) || isBatchImageShape(shape)) {
            return "image";
        } else if (isAudioShape(shape)) {
            return "audio";
        } else if (isVideoShape(shape)) {
            return "video";
        }
        return "unknown";
    }

    /**
     * Ensure tensor is on CPU and in float32 format.
     */
    public static Tensor toCpuFloat(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return tensor.contiguous().cpu().to(ScalarType.Float);
    }

    /**
     * Clone tensor to ensure it's writable.
     */
    public static Tensor clone(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return tensor.clone();
    }
}
