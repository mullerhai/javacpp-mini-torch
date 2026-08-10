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
package org.bytedeco.pytorch.data.safetensors.format;

import org.bytedeco.pytorch.global.torch.ScalarType;

import java.util.Locale;

/**
 * Extended dtype support for safetensors and quantized formats.
 *
 * <p>Includes:
 * <ul>
 *   <li>Standard dtypes: F32, F16, BF16, FP8, I8, I16, I32, I64, U8, BOOL</li>
 *   <li>Quantized dtypes: NF4, FP4, Q80</li>
 *   <li>Mixed precision formats</li>
 * </ul>
 *
 * <p>Reference: safetensors format, BitsAndBytes, GPTQ, AWQ research
 */
public enum SafeDType {

    // ============= Standard floating point =============
    /** 64-bit floating point. */
    F64("F64", 8, ScalarType.Double, true),

    /** 32-bit floating point. */
    F32("F32", 4, ScalarType.Float, true),

    /** 16-bit floating point (IEEE 754 half precision). */
    F16("F16", 2, ScalarType.Half, true),

    /** Brain float 16 (谷歌 16-bit float, better dynamic range). */
    BF16("BF16", 2, ScalarType.BFloat16, true),

    /** FP8 E4M3 (4-bit exp, 3-bit mantissa) - forward pass. */
    F8_E4M3("F8_E4M3", 1, ScalarType.Float8_e4m3fn, true),

    /** FP8 E5M2 (5-bit exp, 2-bit mantissa) - backward pass. */
    F8_E5M2("F8_E5M2", 1, ScalarType.Float8_e5m2, true),

    // ============= Integer types =============
    /** 64-bit signed integer. */
    I64("I64", 8, ScalarType.Long, true),

    /** 32-bit signed integer. */
    I32("I32", 4, ScalarType.Int, true),

    /** 16-bit signed integer. */
    I16("I16", 2, ScalarType.Short, true),

    /** 8-bit signed integer. */
    I8("I8", 1, ScalarType.Char, true),

    /** 8-bit unsigned integer. */
    U8("U8", 1, ScalarType.Byte, true),

    /** Boolean (1 byte per value). */
    BOOL("BOOL", 1, ScalarType.Bool, true),

    // ============= Quantized types =============
    /** 4-bit normalized float (BitsAndBytes NF4). */
    NF4("NF4", 0.5, "4-bit normalized float (BitsAndBytes)", false),

    /** 4-bit floating point. */
    FP4("FP4", 0.5, "4-bit floating point", false),

    /** 8-bit quantization. */
    Q80("Q80", 1.0, "8-bit quantization (Q8_0)", false),

    /** 4-bit quantization. */
    Q40("Q40", 0.5, "4-bit quantization (Q4_0)", false),

    /** 5-bit quantization. */
    Q50("Q50", 0.625, "5-bit quantization", false),

    /** 2-bit quantization. */
    Q20("Q20", 0.25, "2-bit quantization", false),

    /** 1-bit quantization (binary). */
    Q10("Q10", 0.125, "1-bit quantization (binary)", false),

    // ============= Mixed precision =============
    /** Mixed FP16/FP32. */
    MIXED_F16_F32("MIXED_F16_F32", -1, "Mixed FP16/FP32", true),

    /** Mixed BF16/FP32. */
    MIXED_BF16_F32("MIXED_BF16_F32", -1, "Mixed BF16/FP32", true);

    private final String name;
    private final double bytesPerElement;  // 0.5 for 4-bit, -1 for variable
    private final String description;
    private final ScalarType torchType;
    private final boolean standardFormat;

    SafeDType(String name, double bytesPerElement, String description, boolean standardFormat) {
        this.name = name;
        this.bytesPerElement = bytesPerElement;
        this.description = description;
        this.torchType = null;
        this.standardFormat = standardFormat;
    }

    SafeDType(String name, double bytesPerElement, ScalarType torchType, boolean nativeLayout) {
        this.name = name;
        this.bytesPerElement = bytesPerElement;
        this.description = name;
        this.torchType = torchType;
        this.standardFormat = nativeLayout;
    }

    /**
     * Get the dtype name.
     */
    public String getName() {
        return name;
    }

    /**
     * Get bytes per element (0.5 for 4-bit, 1 for 8-bit, etc.).
     */
    public double getBytesPerElement() {
        return bytesPerElement;
    }

    /**
     * Get description.
     */
    public String getDescription() {
        return description;
    }

    /**
     * Get torch ScalarType (null for quantized types).
     */
    public ScalarType toTorch() {
        return torchType;
    }

    /**
     * Check if this is a standard format (vs quantized).
     */
    public boolean isStandard() {
        return standardFormat;
    }

    /**
     * Check if this is a quantized format.
     */
    public boolean isQuantized() {
        return this == NF4 || this == FP4 || this == Q80 || this == Q40 ||
               this == Q50 || this == Q20 || this == Q10;
    }

    /**
     * Check if this is a 4-bit format.
     */
    public boolean is4Bit() {
        return this == NF4 || this == FP4 || this == Q40 || this == Q20 || this == Q10;
    }

    /**
     * Check if this is an 8-bit format.
     */
    public boolean is8Bit() {
        return this == I8 || this == U8 || this == Q80 || this == Q50;
    }

    /**
     * Check if this is a floating point format.
     */
    public boolean isFloatingPoint() {
        return this == F64 || this == F32 || this == F16 || this == BF16 ||
               this == F8_E4M3 || this == F8_E5M2;
    }

    /**
     * Check if this is a reduced precision format.
     */
    public boolean isReducedPrecision() {
        return isQuantized() || (isFloatingPoint() && !this.name.contains("32") && !this.name.contains("64"));
    }

    /**
     * Check if storage layout is native (can use from_blob).
     */
    public boolean isNativeLayout() {
        return this != BOOL;
    }

    /**
     * Parse dtype from string (case-insensitive).
     */
    public static SafeDType fromString(String s) {
        if (s == null) return null;
        String upper = s.toUpperCase(Locale.ROOT).replace("-", "_").replace(" ", "_");
        try {
            return SafeDType.valueOf(upper);
        } catch (IllegalArgumentException e) {
            // Try common aliases
            switch (upper) {
                case "FLOAT64": case "DOUBLE": return F64;
                case "FLOAT32": case "FLOAT": case "F32": return F32;
                case "FLOAT16": case "HALF": case "FP16": return F16;
                case "BFLOAT16": return BF16;
                case "F8E4M3": case "FLOAT8_E4M3FN": return F8_E4M3;
                case "F8E5M2": case "FLOAT8_E5M2": return F8_E5M2;
                case "INT64": case "LONG": return I64;
                case "INT32": case "INT": return I32;
                case "INT16": case "SHORT": return I16;
                case "INT8": case "CHAR": return I8;
                case "UINT8": case "BYTE": case "UBYTE": return U8;
                case "BOOLEAN": case "BOOL": return BOOL;
                case "NF4": return NF4;
                case "FP4": case "F4": return FP4;
                case "Q8_0": case "Q80": return Q80;
                case "Q4_0": case "Q40": return Q40;
                case "Q5_0": case "Q50": return Q50;
                case "Q2_0": case "Q20": return Q20;
                case "Q1_0": case "Q10": return Q10;
                default: return null;
            }
        }
    }

    /**
     * Map torch ScalarType to SafeDType.
     */
    public static SafeDType fromTorch(ScalarType st) {
        if (st == null) return F32;
        ScalarType s = st.intern();
        switch (s) {
            case Double: return F64;
            case Float: return F32;
            case Half: return F16;
            case BFloat16: return BF16;
            case Float8_e4m3fn: return F8_E4M3;
            case Float8_e5m2: return F8_E5M2;
            case Long: return I64;
            case Int: return I32;
            case Short: return I16;
            case Char: return I8;
            case Byte: return U8;
            case Bool: return BOOL;
            default: return F32;
        }
    }

    /**
     * Get storage size for a tensor with given shape.
     */
    public long storageSize(long numElements) {
        if (bytesPerElement < 0) return -1;  // Variable size
        return (long) (numElements * bytesPerElement);
    }

    /**
     * Get bits per element.
     */
    public int bitsPerElement() {
        return (int) (bytesPerElement * 8);
    }

    @Override
    public String toString() {
        return name;
    }
}
