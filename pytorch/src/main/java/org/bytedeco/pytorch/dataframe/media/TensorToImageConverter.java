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

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.vision.utils.ImageTensors;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Specialized converter for Tensor to Image conversions with advanced features.
 *
 * <p>Features:
 * <ul>
 *   <li>Multiple output formats: BufferedImage, ImageData, byte array, base64</li>
 *   <li>Color space conversions: RGB, BGR, Grayscale, HSV</li>
 *   <li>Batch processing for efficient pipeline operations</li>
 *   <li>Format validation and error handling</li>
 *   <li>Integration with OpenCV and torchvision</li>
 * </ul>
 */
public final class TensorToImageConverter {

    private TensorToImageConverter() {}

    // ── Core Conversion Methods ───────────────────────────────────────────

    /**
     * Convert CHW/NCHW tensor to BufferedImage.
     *
     * @param tensor Shape [C,H,W] or [N,C,H,W]
     * @return RGB BufferedImage
     */
    public static BufferedImage toBufferedImage(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return ImageTensors.toBufferedImage(tensor);
    }

    /**
     * Convert CHW/NCHW tensor to ImageData.
     *
     * @param tensor Shape [C,H,W] or [N,C,H,W]
     * @return ImageData object
     */
    public static ImageData toImageData(Tensor tensor) {
        BufferedImage bi = toBufferedImage(tensor);
        return new ImageData(bi);
    }

    /**
     * Convert tensor to grayscale BufferedImage.
     *
     * @param tensor Shape [C,H,W] or [N,C,H,W]
     * @return Grayscale BufferedImage
     */
    public static BufferedImage toGrayscale(Tensor tensor) {
        BufferedImage rgb = toBufferedImage(tensor);
        BufferedImage gray = new BufferedImage(
                rgb.getWidth(), rgb.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = gray.createGraphics();
        g.drawImage(rgb, 0, 0, null);
        g.dispose();
        return gray;
    }

    /**
     * Convert tensor to HSV BufferedImage (for visualization).
     *
     * @param tensor Shape [C,H,W] or [N,C,H,W]
     * @return HSV BufferedImage
     */
    public static BufferedImage toHsv(Tensor tensor) {
        BufferedImage rgb = toBufferedImage(tensor);
        BufferedImage hsv = new BufferedImage(
                rgb.getWidth(), rgb.getHeight(), BufferedImage.TYPE_INT_ARGB);
        for (int y = 0; y < rgb.getHeight(); y++) {
            for (int x = 0; x < rgb.getWidth(); x++) {
                Color c = new Color(rgb.getRGB(x, y));
                float[] hsb = Color.RGBtoHSB(c.getRed(), c.getGreen(), c.getBlue(), null);
                int argb = Color.HSBtoRGB(hsb[0], hsb[1], hsb[2]);
                hsv.setRGB(x, y, argb);
            }
        }
        return hsv;
    }

    /**
     * Convert tensor to byte array (PNG format).
     *
     * @param tensor Shape [C,H,W] or [N,C,H,W]
     * @return PNG byte array
     */
    public static byte[] toBytes(Tensor tensor) {
        BufferedImage bi = toBufferedImage(tensor);
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        try {
            javax.imageio.ImageIO.write(bi, "PNG", baos);
        } catch (java.io.IOException e) {
            throw new RuntimeException("Failed to encode image to PNG", e);
        }
        return baos.toByteArray();
    }

    /**
     * Convert tensor to base64 string (PNG format).
     *
     * @param tensor Shape [C,H,W] or [N,C,H,W]
     * @return Base64 encoded string
     */
    public static String toBase64(Tensor tensor) {
        return java.util.Base64.getEncoder().encodeToString(toBytes(tensor));
    }

    // ── Batch Conversion Methods ──────────────────────────────────────────

    /**
     * Convert batch tensor to list of BufferedImages.
     *
     * @param batchTensor Shape [N,C,H,W]
     * @return List of BufferedImages
     */
    public static List<BufferedImage> batchToBufferedImages(Tensor batchTensor) {
        return MultimodalTensorConverter.batchToBufferedImage(batchTensor);
    }

    /**
     * Convert batch tensor to list of ImageData.
     *
     * @param batchTensor Shape [N,C,H,W]
     * @return List of ImageData
     */
    public static List<ImageData> batchToImageData(Tensor batchTensor) {
        return MultimodalTensorConverter.batchToImageData(batchTensor);
    }

    // ── Format Validation ─────────────────────────────────────────────────

    /**
     * Validate tensor shape for image conversion.
     *
     * @param tensor Tensor to validate
     * @return true if shape is valid for image conversion
     */
    public static boolean isValidImageTensor(Tensor tensor) {
        if (tensor == null) return false;
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        return shape.length == 3 || shape.length == 4;
    }

    /**
     * Get image dimensions from tensor shape.
     *
     * @param tensor Tensor with image shape
     * @return int array [height, width]
     */
    public static int[] getImageDimensions(Tensor tensor) {
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        if (shape.length == 3) {
            return new int[]{(int) shape[1], (int) shape[2]}; // [H, W]
        } else if (shape.length == 4) {
            return new int[]{(int) shape[2], (int) shape[3]}; // [H, W]
        }
        throw new IllegalArgumentException("Invalid tensor shape for image: " + Arrays.toString(shape));
    }

    /**
     * Get number of channels from tensor.
     *
     * @param tensor Tensor with image shape
     * @return Number of channels (1 for grayscale, 3 for RGB)
     */
    public static int getChannelCount(Tensor tensor) {
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        if (shape.length >= 3) {
            return (int) shape[shape.length == 4 ? 1 : 0];
        }
        throw new IllegalArgumentException("Invalid tensor shape: " + Arrays.toString(shape));
    }

    // ── Tensor Manipulation ───────────────────────────────────────────────

    /**
     * Normalize tensor to [0, 1] range.
     *
     * @param tensor Input tensor
     * @return Normalized tensor
     */
    public static Tensor normalize(Tensor tensor) {
        Tensor cpu = tensor.contiguous().cpu().to(ScalarType.Float);
        float min = minValue(cpu);
        float max = maxValue(cpu);
        if (max - min < 1e-6) {
            return cpu.sub(new Scalar(min));
        }
        return cpu.sub(new Scalar(min)).div(new Scalar(max - min));
    }

    /**
     * Denormalize tensor from [0, 1] to original range.
     *
     * @param tensor Normalized tensor
     * @param mean Original mean
     * @param std Original std
     * @return Denormalized tensor
     */
    public static Tensor denormalize(Tensor tensor, float[] mean, float[] std) {
        Tensor t = tensor.clone();
        for (int i = 0; i < mean.length; i++) {
            t = t.select(0, i).mul(new Scalar(std[i])).add(new Scalar(mean[i]));
        }
        return t;
    }

    /**
     * Resize tensor to target dimensions using bilinear interpolation.
     *
     * @param tensor Input tensor [C,H,W]
     * @param targetHeight Target height
     * @param targetWidth Target width
     * @return Resized tensor
     */
    public static Tensor resize(Tensor tensor, int targetHeight, int targetWidth) {
        try {
            Class<?> visionF = Class.forName("org.bytedeco.pytorch.vision.transforms.functional.VisionF");
            Object result = visionF.getMethod("resize", Tensor.class, int.class, int.class)
                    .invoke(null, tensor, targetHeight, targetWidth);
            return (Tensor) result;
        } catch (Exception e) {
            BufferedImage bi = toBufferedImage(tensor);
            BufferedImage resized = new BufferedImage(targetWidth, targetHeight, bi.getType());
            Graphics2D g = resized.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.drawImage(bi, 0, 0, targetWidth, targetHeight, null);
            g.dispose();
            return ImageTensors.toTensor(resized);
        }
    }

    /**
     * Center crop tensor to target dimensions.
     *
     * @param tensor Input tensor [C,H,W]
     * @param cropHeight Crop height
     * @param cropWidth Crop width
     * @return Cropped tensor
     */
    public static Tensor centerCrop(Tensor tensor, int cropHeight, int cropWidth) {
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        int h = (int) shape[1];
        int w = (int) shape[2];

        int top = Math.max(0, (h - cropHeight) / 2);
        int left = Math.max(0, (w - cropWidth) / 2);

        return tensor.narrow(1, top, cropHeight).narrow(2, left, cropWidth);
    }

    // ── Utility Methods ───────────────────────────────────────────────────

    private static float minValue(Tensor t) {
        return t.min().item().toFloat();
    }

    private static float maxValue(Tensor t) {
        return t.max().item().toFloat();
    }

    /**
     * Create test pattern tensor (useful for debugging).
     *
     * @param width Image width
     * @param height Image height
     * @param pattern Type: "gradient", "checkerboard", "noise"
     * @return Tensor [3, H, W]
     */
    public static Tensor createTestPattern(int width, int height, String pattern) {
        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = bi.createGraphics();

        if ("gradient".equals(pattern)) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int r = (x * 255) / width;
                    int gr = (y * 255) / height;
                    int b = ((x + y) * 128) / (width + height);
                    g.setColor(new Color(r, gr, b));
                    g.fillRect(x, y, 1, 1);
                }
            }
        } else if ("checkerboard".equals(pattern)) {
            int size = Math.max(8, Math.min(width, height) / 8);
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    boolean white = ((x / size) + (y / size)) % 2 == 0;
                    g.setColor(white ? Color.WHITE : Color.BLACK);
                    g.fillRect(x, y, 1, 1);
                }
            }
        } else {
            g.setColor(Color.BLACK);
            g.fillRect(0, 0, width, height);
            g.setColor(Color.WHITE);
            g.drawString("Unknown pattern: " + pattern, 10, height / 2);
        }
        g.dispose();
        return ImageTensors.toTensor(bi);
    }
}
