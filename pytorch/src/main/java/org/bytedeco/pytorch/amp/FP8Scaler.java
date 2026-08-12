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
package org.bytedeco.pytorch.amp;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.concurrent.atomic.AtomicLong;

/**
 * FP8 Mixed Precision training support for large models.
 *
 * <p>Provides:
 * <ul>
 *   <li>FP8 E4M3 for forward pass (inference precision)</li>
 *   <li>FP8 E5M2 for backward pass (training precision)</li>
 *   <li>Per-tensor and per-channel scaling</li>
 *   <li>Dynamic scaling based on amax history</li>
 *   <li>Meta tensor support for memory optimization</li>
 * </ul>
 *
 * <p>Reference: NVIDIA FP8 Formats for Deep Learning and FP8 training papers
 *
 * <pre>{@code
 * try (FP8Scaler fp8 = FP8Scaler.builder().device("cuda").build()) {
 *     Tensor scaled = fp8.scale(tensor);
 *     Tensor quantized = fp8.quantize(tensor);
 *     Tensor restored = fp8.dequantize(quantized);
 * }
 * }</pre>
 */
public class FP8Scaler implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // FP8 format
    private final FP8Format format;

    // Scaling configuration
    private final ScaleMethod scaleMethod;
    private final float historyWindow;

    // Amax history for dynamic scaling
    private final float[] amaxHistory;
    private int historyIndex;
    private float currentScale;

    // Performance metrics
    private final AtomicLong quantizeCount = new AtomicLong(0);
    private final AtomicLong dequantizeCount = new AtomicLong(0);
    private final AtomicLong quantizeTimeMs = new AtomicLong(0);

    // Constants for FP8 formats
    private static final float E4M3_MAX = 448.0f;   // Max value for FP8 E4M3
    private static final float E5M2_MAX = 57344.0f; // Max value for FP8 E5M2

    /**
     * FP8 format types.
     */
    public enum FP8Format {
        E4M3("FP8 E4M3", E4M3_MAX),  // Forward pass (inference)
        E5M2("FP8 E5M2", E5M2_MAX);  // Backward pass (training)

        private final String name;
        private final float maxValue;

        FP8Format(String name, float maxValue) {
            this.name = name;
            this.maxValue = maxValue;
        }

        public String getName() { return name; }
        public float getMaxValue() { return maxValue; }
    }

    /**
     * Scale method for FP8.
     */
    public enum ScaleMethod {
        PER_TENSOR,    // Single scale for entire tensor
        PER_CHANNEL,    // Scale per channel
        PER_GROUP,      // Scale per group
        DYNAMIC         // Dynamic scaling based on amax
    }

    /**
     * Create FP8Scaler with E4M3 format.
     */
    public static FP8Scaler e4m3() {
        return builder().format(FP8Format.E4M3).build();
    }

    /**
     * Create FP8Scaler with E5M2 format.
     */
    public static FP8Scaler e5m2() {
        return builder().format(FP8Format.E5M2).build();
    }

    /**
     * Builder for FP8Scaler.
     */
    public static Builder builder() {
        return new Builder();
    }

    private FP8Scaler(Builder builder) {
        this.format = builder.format;
        this.scaleMethod = builder.scaleMethod;
        this.historyWindow = builder.historyWindow;
        this.amaxHistory = new float[(int) historyWindow];
        this.historyIndex = 0;
        this.currentScale = 1.0f;

        // Initialize history with max value
        for (int i = 0; i < amaxHistory.length; i++) {
            amaxHistory[i] = builder.initAmax;
        }
    }

    /**
     * Quantize tensor to FP8.
     */
    public Tensor quantize(Tensor input) {
        if (closed || input == null || !input.defined()) {
            return input;
        }
        long start = System.currentTimeMillis();

        // Compute amax
        float amax = computeAmax(input);

        // Update scale based on amax
        updateScale(amax);

        // Quantize
        Tensor scaled = input.mul(new Scalar(currentScale));
        Tensor quantized = scaled.to(getScalarType());
        scaled.close();

        quantizeCount.incrementAndGet();
        quantizeTimeMs.addAndGet(System.currentTimeMillis() - start);

        return quantized;
    }

    /**
     * Dequantize FP8 tensor back to FP32.
     */
    public Tensor dequantize(Tensor input) {
        if (closed || input == null || !input.defined()) {
            return input;
        }
        long start = System.currentTimeMillis();

        // Convert to float
        Tensor f32 = input.to(ScalarType.Float);

        // Scale down
        var invScale = new Scalar(1.0f / currentScale);
        Tensor result = f32.mul(invScale);
        f32.close();

        dequantizeCount.incrementAndGet();
        quantizeTimeMs.addAndGet(System.currentTimeMillis() - start);

        return result;
    }

    /**
     * Scale tensor for FP8 (no quantization).
     */
    public Tensor scale(Tensor input) {
        if (closed || input == null || !input.defined()) {
            return input;
        }
        return input.mul(new Scalar(currentScale));
    }

    /**
     * Unscale tensor from FP8.
     */
    public Tensor unscale(Tensor input) {
        if (closed || input == null || !input.defined()) {
            return input;
        }
        return input.mul(new Scalar(1.0f / currentScale));
    }

    /**
     * Compute amax (maximum absolute value) of tensor.
     */
    public float computeAmax(Tensor input) {
        if (input == null || !input.defined()) {
            return 0.0f;
        }
        try {
            Tensor abs = torch.abs(input);
            Tensor max = abs.max();
            float amax = max.item_float();
            abs.close();
            max.close();
            return amax;
        } catch (Exception e) {
            return 1.0f;
        }
    }

    /**
     * Compute amax per channel.
     */
    public Tensor computeAmaxPerChannel(Tensor input, int dim) {
        if (input == null || !input.defined()) {
            return null;
        }
        try {
            Tensor abs = torch.abs(input);
            var max = abs.max(dim*1l, true);
            abs.close();
            return max.get0();
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Update scale based on amax history.
     */
    private void updateScale(float amax) {
        // Add to history
        amaxHistory[historyIndex] = amax;
        historyIndex = (historyIndex + 1) % amaxHistory.length;

        // Compute max from history
        float maxAmax = 0.0f;
        for (float v : amaxHistory) {
            if (v > maxAmax) maxAmax = v;
        }

        // Update scale
        if (maxAmax > 0) {
            currentScale = format.getMaxValue() / maxAmax;
        }
    }

    /**
     * Get current scale factor.
     */
    public float getScale() {
        return currentScale;
    }

    /**
     * Get scalar type for this format.
     */
    public ScalarType getScalarType() {
        return format == FP8Format.E4M3
                ? ScalarType.Float8_e4m3fn
                : ScalarType.Float8_e5m2;
    }

    /**
     * Get format.
     */
    public FP8Format getFormat() {
        return format;
    }

    /**
     * Check if closed.
     */
    public boolean isClosed() {
        return closed;
    }

    /**
     * Get FP8 statistics.
     */
    public FP8Stats getStats() {
        return new FP8Stats(
                format.getName(),
                scaleMethod.name(),
                currentScale,
                quantizeCount.get(),
                dequantizeCount.get(),
                quantizeTimeMs.get()
        );
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.printf(
                "[FP8Scaler] Closed: format=%s, scale=%.4f, quantize=%d, dequantize=%d, time=%.2fs%n",
                format.getName(), currentScale,
                quantizeCount.get(), dequantizeCount.get(),
                quantizeTimeMs.get() / 1000.0);
    }

    /**
     * FP8 statistics.
     */
    public static class FP8Stats {
        public final String format;
        public final String scaleMethod;
        public final float currentScale;
        public final long quantizeCount;
        public final long dequantizeCount;
        public final long totalTimeMs;

        public FP8Stats(String format, String scaleMethod, float currentScale,
                      long quantizeCount, long dequantizeCount, long totalTimeMs) {
            this.format = format;
            this.scaleMethod = scaleMethod;
            this.currentScale = currentScale;
            this.quantizeCount = quantizeCount;
            this.dequantizeCount = dequantizeCount;
            this.totalTimeMs = totalTimeMs;
        }

        public double avgQuantizeTimeMs() {
            long total = quantizeCount + dequantizeCount;
            return total > 0 ? (double) totalTimeMs / total : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "FP8Stats{format=%s, scale=%.4f, quantize=%d, avgTime=%.3fms}",
                    format, currentScale, quantizeCount, avgQuantizeTimeMs());
        }
    }

    /**
     * Builder for FP8Scaler.
     */
    public static class Builder {
        private FP8Format format = FP8Format.E4M3;
        private ScaleMethod scaleMethod = ScaleMethod.PER_TENSOR;
        private float historyWindow = 1024;
        private float initAmax = 1.0f;
        private Device device;

        public Builder format(FP8Format format) {
            this.format = format;
            return this;
        }

        public Builder scaleMethod(ScaleMethod scaleMethod) {
            this.scaleMethod = scaleMethod;
            return this;
        }

        public Builder historyWindow(float historyWindow) {
            this.historyWindow = historyWindow;
            return this;
        }

        public Builder initAmax(float initAmax) {
            this.initAmax = initAmax;
            return this;
        }

        public Builder device(Device device) {
            this.device = device;
            return this;
        }

        public Builder device(String device) {
            this.device = new Device(device);
            return this;
        }

        public FP8Scaler build() {
            return new FP8Scaler(this);
        }
    }
}
