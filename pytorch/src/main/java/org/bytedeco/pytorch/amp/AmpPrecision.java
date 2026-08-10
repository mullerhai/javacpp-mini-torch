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

/**
 * AMP precision types supported by the mixed precision training system.
 *
 * <p>Reference: NVIDIA AMP, PyTorch autocast, FP8 training papers
 */
public enum AmpPrecision {

    /** Full precision (32-bit float). */
    FP32("float32", 32, 1.0f),

    /** Half precision (16-bit float). */
    FP16("float16", 16, 4.0f),

    /** Brain float (16-bit float, better numerical stability). */
    BF16("bfloat16", 16, 4.0f),

    /** Double precision (64-bit float). */
    FP64("float64", 64, 0.5f),

    /** FP8 E4M3 (4-bit exponent, 3-bit mantissa) - inference. */
    FP8_E4M3("float8_e4m3fn", 8, 16.0f),

    /** FP8 E5M2 (5-bit exponent, 2-bit mantissa) - training. */
    FP8_E5M2("float8_e5m2", 8, 16.0f),

    /** INT8 quantization. */
    INT8("int8", 8, 16.0f),

    /** INT4 quantization. */
    INT4("int4", 4, 32.0f);

    private final String name;
    private final int bits;
    private final float speedupFactor;

    AmpPrecision(String name, int bits, float speedupFactor) {
        this.name = name;
        this.bits = bits;
        this.speedupFactor = speedupFactor;
    }

    /**
     * Get the precision name.
     */
    public String getName() {
        return name;
    }

    /**
     * Get the number of bits.
     */
    public int getBits() {
        return bits;
    }

    /**
     * Get the speedup factor compared to FP32.
     */
    public float getSpeedupFactor() {
        return speedupFactor;
    }

    /**
     * Get memory savings compared to FP32.
     */
    public float getMemorySavings() {
        return 32.0f / bits;
    }

    /**
     * Check if this is a reduced precision format.
     */
    public boolean isReducedPrecision() {
        return bits < 32;
    }

    /**
     * Check if this is a floating point format.
     */
    public boolean isFloatingPoint() {
        return this == FP32 || this == FP16 || this == BF16 ||
               this == FP64 || this == FP8_E4M3 || this == FP8_E5M2;
    }

    /**
     * Check if this is a quantized format.
     */
    public boolean isQuantized() {
        return this == INT8 || this == INT4;
    }

    /**
     * Check if this is an 8-bit format.
     */
    public boolean is8Bit() {
        return this == FP8_E4M3 || this == FP8_E5M2 || this == INT8;
    }

    /**
     * Get a human-readable description.
     */
    public String getDescription() {
        switch (this) {
            case FP32: return "Full precision (32-bit float)";
            case FP16: return "Half precision (16-bit float)";
            case BF16: return "Brain float (16-bit float)";
            case FP64: return "Double precision (64-bit float)";
            case FP8_E4M3: return "FP8 E4M3 (4-bit exp, 3-bit mantissa)";
            case FP8_E5M2: return "FP8 E5M2 (5-bit exp, 2-bit mantissa)";
            case INT8: return "INT8 quantization (8-bit integer)";
            case INT4: return "INT4 quantization (4-bit integer)";
            default: return name;
        }
    }

    /**
     * Get default precision for training.
     */
    public static AmpPrecision defaultForTraining() {
        return BF16;  // BF16 is recommended for training
    }

    /**
     * Get default precision for inference.
     */
    public static AmpPrecision defaultForInference() {
        return FP16;  // FP16 is common for inference
    }

    @Override
    public String toString() {
        return name;
    }
}
