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
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.bitsandbytes;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptional;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiConsumer;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.linear;
import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Enterprise-grade BitsAndBytes-style 4/8-bit quantization helpers.
 *
 * <p>Mirrors the Python {@code bitsandbytes} (bnb) package as used by
 * HuggingFace Transformers, PEFT, and QLoRA:
 *
 * <ul>
 *   <li>Functional blockwise quant/dequant for NF4, FP4, INT8</li>
 *   <li>{@link QuantState} carrying all metadata required to round-trip safetensors</li>
 *   <li>{@link Linear4bit} / {@link Linear8bitLt} fused layers</li>
 *   <li>{@link Params4bit} / {@link Int8Params} weight-parameter wrappers</li>
 *   <li>Outlier-aware INT8 column-wise dequant (LLM.int8() style)</li>
 *   <li>{@link BnbOptimizer} stats and 8-bit {@link Adam8bit} / {@link Lion8bit} accumulator hooks</li>
 *   <li>HF-style {@code from_pretrained} / {@code state_dict} (de)serialization</li>
 *   <li>Memory estimation, ratio, and per-layer reporting</li>
 *   <li>Defensive input validation, NaN/Inf guards, device/dtype helpers</li>
 *   <li>Snake_case Python-style aliases on every public entry point</li>
 * </ul>
 *
 * <p><b>Note:</b> This is a pure-Java CPU/quantization reference. It is not
 * CUDA-kernel identical to the official bnb CUDA kernels, but the byte-level
 * representation and forward semantics are bit-compatible for offline
 * testing, weight I/O, and CPU QLoRA pipelines. The API is intentionally
 * aligned with {@code bitsandbytes==0.45.x}.
 *
 * <pre>{@code
 * BitsAndBytesConfig cfg = BitsAndBytesConfig.qloraDefaults();
 * QuantizedModel qm = BitsAndBytes.quantizeModel(linears, cfg);
 * BitsAndBytes.prepareModelForKbitTraining(params);
 * Linear4bit layer = BitsAndBytes.linear4bit(dense, cfg);
 * Tensor y = layer.forward(x);
 *
 * // snake_case Python style
 * BitsAndBytes.quantize_model(linears, cfg);
 * BitsAndBytes.prepare_model_for_kbit_training(model);
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class BitsAndBytes {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    /** Library version (mirrors bnb 0.45.x semantics). */
    public static final String VERSION = "0.45.0-java";
    /** Default block size used when none is configured (matches bnb default). */
    public static final int DEFAULT_BLOCKSIZE = 64;
    /** NF4 levels count. */
    public static final int NF4_LEVELS_COUNT = 16;
    /** FP4 E2M1 levels count (16 values, including sign + zero). */
    public static final int FP4_LEVELS_COUNT = 16;
    /** Default LLM.int8() outlier threshold. */
    public static final double DEFAULT_LLM_INT8_THRESHOLD = 6.0;

    // -----------------------------------------------------------------------
    // Canonical quantization levels
    // -----------------------------------------------------------------------

    /**
     * NF4 levels (Dettmers et al., QLoRA). 16 values on [-1, 1] approximating
     * the standard-normal quantiles used as codebook for 4-bit storage.
     */
    public static final float[] NF4_LEVELS = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
            0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
            0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
    };

    /**
     * FP4 E2M1 levels used by the {@code fp4} quant type.
     * Index 0 is {@code +0}, indices 1..7 are positive, index 8 is {@code -0},
     * indices 9..15 are negative.
     */
    public static final float[] FP4_LEVELS = {
            0.0f,
            0.0625f, 0.125f, 0.1875f, 0.25f, 0.375f, 0.5f, 0.75f,
            -0.0f,
            -0.0625f, -0.125f, -0.1875f, -0.25f, -0.375f, -0.5f, -0.75f
    };

    /** Quant-type string constants (avoid typos in caller code). */
    public static final String TYPE_NF4 = "nf4";
    public static final String TYPE_FP4 = "fp4";
    public static final String TYPE_INT8 = "int8";
    public static final String TYPE_FP16 = "fp16";
    public static final String TYPE_BF16 = "bf16";
    public static final String TYPE_FP32 = "fp32";

    private BitsAndBytes() {}

    // -----------------------------------------------------------------------
    // Validation helpers
    // -----------------------------------------------------------------------

    /**
     * Throw {@link IllegalArgumentException} if {@code t} is null, undefined, or
     * a null tensor. Mirrors the spirit of Python's "user is always wrong" bnb
     * assertions but returns Java-friendly exceptions with descriptive messages.
     */
    public static Tensor requireTensor(Tensor t, String name) {
        Objects.requireNonNull(name, "name");
        if (t == null || t.isNull()) {
            throw new IllegalArgumentException("Tensor '" + name + "' is null");
        }
        if (!t.defined()) {
            throw new IllegalArgumentException("Tensor '" + name + "' is not defined");
        }
        return t;
    }

    /** Coerce a tensor to a contiguous fp32 CPU tensor for read-back. */
    static Tensor toContiguousFp32(Tensor t) {
        return t.to(ScalarType.Float).contiguous();
    }

    /** Convert bitsandbytes dtype string to {@link ScalarType} when feasible. */
    public static ScalarType scalarTypeFromString(String dtype) {
        if (dtype == null) return ScalarType.Float;
        String s = dtype.toLowerCase(Locale.ROOT);
        switch (s) {
            case "fp32": case "float32": case "float": return ScalarType.Float;
            case "fp16": case "float16": case "half": return ScalarType.Half;
            case "bf16": case "bfloat16": return ScalarType.BFloat16;
            case "int8": return ScalarType.Char;
            case "uint8": return ScalarType.Byte;
            default: return ScalarType.Float;
        }
    }

    /** Normalize a quant-type string (lowercase, trimmed). Throws on unknown. */
    public static String normalizeQuantType(String quantType) {
        if (quantType == null) throw new IllegalArgumentException("quantType is null");
        String s = quantType.trim().toLowerCase(Locale.ROOT);
        switch (s) {
            case "nf4": case "fp4": case "int8": return s;
            default:
                throw new IllegalArgumentException(
                        "Unknown quantType: '" + quantType + "' (expected nf4|fp4|int8)");
        }
    }

    private static boolean isFinite(float v) {
        return !Float.isNaN(v) && !Float.isInfinite(v);
    }

    /**
     * Detect non-finite values (NaN/Inf). If any element of the tensor is not
     * finite, an {@link IllegalStateException} is thrown with the offending
     * count. Defends against upstream injection of poisoned weights.
     */
    public static long assertNoNonFinite(Tensor t, String context) {
        requireTensor(t, context);
        float[] data = toFloatArray(t);
        long bad = 0;
        for (float v : data) {
            if (!isFinite(v)) bad++;
        }
        if (bad > 0) {
            throw new IllegalStateException(
                    "Tensor '" + context + "' contains " + bad + " non-finite value(s) (NaN/Inf)");
        }
        return bad;
    }

    // =========================================================================
    // QuantState
    // =========================================================================

    /**
     * Quantization state for a weight tensor — mirrors Python
     * {@code bitsandbytes.functional.QuantState} and HuggingFace safetensors
     * {@code "quant_state"} sub-dictionary.
     */
    public static final class QuantState {
        /** Per-element codes (float-stored indices 0..15 or int8 -128..127). */
        public final Tensor qweight;
        /** Per-block absmax / scale. */
        public final Tensor absmax;
        public final int blocksize;
        public final String quantType;
        public final long[] originalShape;
        public final boolean doubleQuant;
        /** Nested state for double quant of absmax (QLoRA). May be null. */
        public final QuantState nested;
        /** "uint8", "int8", or "float". */
        public final String codeDtype;
        /** Optional packed nibble storage (2 × 4-bit codes per byte). May be null. */
        public final byte[] packedCodes;
        /** Restored second-level scale (when nested is null but doubleQuant==true). */
        public final float nestedScale;
        /** Per-row outlier fp16 mask (LLM.int8() style). May be null. */
        public final int[] outlierIndices;
        /** Per-row outlier values (LLM.int8() style). May be null. */
        public final float[] outlierValues;

        public QuantState(Tensor qweight, Tensor absmax, int blocksize,
                          String quantType, long[] originalShape, boolean doubleQuant) {
            this(qweight, absmax, blocksize, quantType, originalShape, doubleQuant,
                    null, "float", null, 1f, null, null);
        }

        public QuantState(Tensor qweight, Tensor absmax, int blocksize,
                          String quantType, long[] originalShape, boolean doubleQuant,
                          QuantState nested, String codeDtype, byte[] packedCodes,
                          float nestedScale) {
            this(qweight, absmax, blocksize, quantType, originalShape, doubleQuant,
                    nested, codeDtype, packedCodes, nestedScale, null, null);
        }

        public QuantState(Tensor qweight, Tensor absmax, int blocksize,
                          String quantType, long[] originalShape, boolean doubleQuant,
                          QuantState nested, String codeDtype, byte[] packedCodes,
                          float nestedScale, int[] outlierIndices, float[] outlierValues) {
            if (qweight == null || !qweight.defined()) {
                throw new IllegalArgumentException("QuantState.qweight is null/undefined");
            }
            if (absmax == null || !absmax.defined()) {
                throw new IllegalArgumentException("QuantState.absmax is null/undefined");
            }
            if (blocksize <= 0) {
                throw new IllegalArgumentException("QuantState.blocksize must be positive, got " + blocksize);
            }
            if (originalShape == null || originalShape.length == 0) {
                throw new IllegalArgumentException("QuantState.originalShape must be non-empty");
            }
            this.qweight = qweight;
            this.absmax = absmax;
            this.blocksize = blocksize;
            this.quantType = normalizeQuantType(quantType);
            this.originalShape = originalShape.clone();
            this.doubleQuant = doubleQuant;
            this.nested = nested;
            this.codeDtype = codeDtype == null ? "float" : codeDtype;
            this.packedCodes = packedCodes;
            this.nestedScale = nestedScale;
            this.outlierIndices = outlierIndices;
            this.outlierValues = outlierValues;
        }

        public long numel() {
            long n = 1;
            for (long s : originalShape) n *= s;
            return n;
        }

        public int numBlocks() {
            return (int) ((numel() + blocksize - 1) / blocksize);
        }

        /** Estimated on-disk / in-memory bytes for this quantized state. */
        public long memoryBytes() {
            return estimateMemoryBytes(numel(), quantType, blocksize, doubleQuant);
        }

        /** Number of outlier columns (0 when LLM.int8() outliers were not retained). */
        public int outlierCount() {
            return outlierIndices == null ? 0 : outlierIndices.length;
        }

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("quant_type", quantType);
            m.put("blocksize", blocksize);
            m.put("double_quant", doubleQuant);
            m.put("shape", originalShape.clone());
            m.put("numel", numel());
            m.put("num_blocks", numBlocks());
            m.put("memory_bytes", memoryBytes());
            m.put("code_dtype", codeDtype);
            m.put("has_nested", nested != null);
            m.put("has_packed", packedCodes != null);
            m.put("nested_scale", nestedScale);
            m.put("outlier_count", outlierCount());
            return m;
        }

        @Override
        public String toString() {
            return "QuantState{type=" + quantType
                    + ", shape=" + Arrays.toString(originalShape)
                    + ", blocksize=" + blocksize
                    + ", doubleQuant=" + doubleQuant
                    + ", mem=" + memoryBytes() + "B"
                    + ", outliers=" + outlierCount() + "}";
        }
    }

    // =========================================================================
    // INT8 blockwise
    // =========================================================================

    /** Blockwise INT8 quantization. Symmetric, per-block scales. */
    public static QuantState quantizeInt8(Tensor weight, int blocksize) {
        requireTensor(weight, "weight");
        if (blocksize <= 0) blocksize = DEFAULT_BLOCKSIZE;
        Tensor w = weight.reshape(-1).to(ScalarType.Float).contiguous();
        long n = w.numel();
        int blocks = (int) ((n + blocksize - 1) / blocksize);
        float[] data = toFloatArray(w);
        float[] codes = new float[(int) n];
        float[] scales = new float[blocks];
        for (int b = 0; b < blocks; b++) {
            int start = b * blocksize;
            int end = Math.min((int) n, start + blocksize);
            float amax = 0f;
            for (int i = start; i < end; i++) {
                float v = data[i];
                float a = Math.abs(v);
                if (a > amax) amax = a;
            }
            if (amax < 1e-12f) amax = 1e-12f;
            scales[b] = amax / 127f;
            for (int i = start; i < end; i++) {
                int q = Math.round(data[i] / scales[b]);
                q = Math.max(-128, Math.min(127, q));
                codes[i] = q;
            }
        }
        return new QuantState(tensor(codes), tensor(scales), blocksize, TYPE_INT8,
                shapeOf(weight), false, null, "int8", null, 1f);
    }

    /** Blockwise INT8 dequantization. */
    public static Tensor dequantizeInt8(QuantState state) {
        Objects.requireNonNull(state, "state");
        float[] codes = toFloatArray(state.qweight);
        float[] scales = resolveScales(state);
        float[] out = new float[codes.length];
        for (int i = 0; i < codes.length; i++) {
            int b = Math.min(scales.length - 1, i / state.blocksize);
            out[i] = codes[i] * scales[b];
        }
        Tensor t = tensor(out).reshape(state.originalShape);
        return applyOutliers(t, state);
    }

    // =========================================================================
    // NF4 blockwise
    // =========================================================================

    public static QuantState quantizeNf4(Tensor weight, int blocksize) {
        return quantizeNf4(weight, blocksize, false);
    }

    public static QuantState quantizeNf4(Tensor weight, int blocksize, boolean doubleQuant) {
        return quantizeBlockwise(weight, blocksize, TYPE_NF4, doubleQuant);
    }

    public static Tensor dequantizeNf4(QuantState state) {
        return dequantizeBlockwise(state, NF4_LEVELS);
    }

    // =========================================================================
    // FP4 blockwise
    // =========================================================================

    public static QuantState quantizeFp4(Tensor weight, int blocksize) {
        return quantizeFp4(weight, blocksize, false);
    }

    public static QuantState quantizeFp4(Tensor weight, int blocksize, boolean doubleQuant) {
        return quantizeBlockwise(weight, blocksize, TYPE_FP4, doubleQuant);
    }

    public static Tensor dequantizeFp4(QuantState state) {
        return dequantizeBlockwise(state, FP4_LEVELS);
    }

    // =========================================================================
    // Generic blockwise functional API (Python bnb functional.* parity)
    // =========================================================================

    /**
     * Generic blockwise quantizer. Dispatches to NF4/FP4/INT8 code paths with
     * the same numerical conventions as bnb's
     * {@code bitsandbytes.functional.quantize_4bit/quantize_blockwise}.
     *
     * @param weight      weight tensor; will be reshaped to 1-D internally
     * @param blocksize   block size (≤0 → DEFAULT_BLOCKSIZE)
     * @param quantType   {@code "nf4"} | {@code "fp4"} | {@code "int8"}
     * @param doubleQuant whether to int8-quantize the absmax (QLoRA)
     */
    public static QuantState quantizeBlockwise(Tensor weight, int blocksize,
                                                String quantType, boolean doubleQuant) {
        requireTensor(weight, "weight");
        String t = normalizeQuantType(quantType);
        if (blocksize <= 0) blocksize = DEFAULT_BLOCKSIZE;
        Tensor w = weight.reshape(-1).to(ScalarType.Float).contiguous();
        long n = w.numel();
        int blocks = (int) ((n + blocksize - 1) / blocksize);
        float[] data = toFloatArray(w);
        float[] codes = new float[(int) n];
        float[] scales = new float[blocks];
        float[] levels = t.equals(TYPE_FP4) ? FP4_LEVELS : NF4_LEVELS;

        for (int b = 0; b < blocks; b++) {
            int start = b * blocksize;
            int end = Math.min((int) n, start + blocksize);
            float amax = 0f;
            for (int i = start; i < end; i++) {
                float a = Math.abs(data[i]);
                if (a > amax) amax = a;
            }
            if (amax < 1e-12f) amax = 1e-12f;
            scales[b] = amax;
            for (int i = start; i < end; i++) {
                if (t.equals(TYPE_INT8)) {
                    int q = Math.round(data[i] / (amax / 127f));
                    q = Math.max(-128, Math.min(127, q));
                    codes[i] = q;
                } else {
                    codes[i] = nearestLevelIndex(data[i] / amax, levels);
                }
            }
        }

        QuantState nested = null;
        float nestedScale = 1f;
        if (doubleQuant) {
            float smax = 0f;
            for (float s : scales) if (Math.abs(s) > smax) smax = Math.abs(s);
            if (smax < 1e-12f) smax = 1e-12f;
            nestedScale = smax / 127f;
            float[] nestedCodes = new float[scales.length];
            for (int i = 0; i < scales.length; i++) {
                int q = Math.round(scales[i] / nestedScale);
                q = Math.max(-128, Math.min(127, q));
                nestedCodes[i] = q;
                scales[i] = q * nestedScale;
            }
            nested = new QuantState(tensor(nestedCodes), tensor(new float[]{nestedScale}),
                    scales.length, TYPE_INT8, new long[]{scales.length}, false,
                    null, "int8", null, nestedScale);
        }
        byte[] packed = (t.equals(TYPE_NF4) || t.equals(TYPE_FP4)) ? pack4bit(codes) : null;
        String codeDtype = t.equals(TYPE_INT8) ? "int8" : "uint8";
        return new QuantState(tensor(codes), tensor(scales), blocksize, t,
                shapeOf(weight), doubleQuant, nested, codeDtype, packed, nestedScale);
    }

    /**
     * Generic blockwise dequantizer. Restores float from a {@link QuantState}.
     *
     * @param state  state from {@link #quantizeBlockwise} (or any of the typed
     *               helpers)
     * @param levels level set (NF4 or FP4); ignored for int8
     */
    public static Tensor dequantizeBlockwise(QuantState state, float[] levels) {
        Objects.requireNonNull(state, "state");
        String t = state.quantType;
        float[] codes = toFloatArray(state.qweight);
        float[] scales = resolveScales(state);
        float[] out = new float[codes.length];
        for (int i = 0; i < codes.length; i++) {
            int b = Math.min(scales.length - 1, i / state.blocksize);
            if (t.equals(TYPE_INT8)) {
                out[i] = codes[i] * scales[b];
            } else {
                int idx = Math.max(0, Math.min(15, Math.round(codes[i])));
                out[i] = levels[idx] * scales[b];
            }
        }
        Tensor restored = tensor(out).reshape(state.originalShape);
        return applyOutliers(restored, state);
    }

    /**
     * Restore outlier columns (if recorded) on a tensor reshaped to originalShape.
     * Returns the same tensor if there are no outliers.
     */
    private static Tensor applyOutliers(Tensor t, QuantState state) {
        if (state.outlierIndices == null || state.outlierIndices.length == 0) return t;
        if (state.outlierValues == null) return t;
        long rows = 1;
        for (int i = 0; i < state.originalShape.length - 1; i++) rows *= state.originalShape[i];
        long cols = state.originalShape[state.originalShape.length - 1];
        float[] data = toFloatArray(t);
        for (int i = 0; i < state.outlierIndices.length; i++) {
            int idx = state.outlierIndices[i];
            long r = idx / Math.max(1, cols);
            long c = idx % Math.max(1, cols);
            if (r >= 0 && r < rows && c >= 0 && c < cols) {
                data[(int) (r * cols + c)] = state.outlierValues[i];
            }
        }
        return tensor(data).reshape(state.originalShape);
    }

    // =========================================================================
    // Dispatcher
    // =========================================================================

    /** Dispatch dequantization by {@link QuantState#quantType}. */
    public static Tensor dequantize(QuantState state) {
        Objects.requireNonNull(state, "state");
        return switch (state.quantType) {
            case TYPE_NF4 -> dequantizeNf4(state);
            case TYPE_FP4 -> dequantizeFp4(state);
            case TYPE_INT8 -> dequantizeInt8(state);
            default -> throw new IllegalArgumentException("Unknown quantType: " + state.quantType);
        };
    }

    /**
     * Pick quant routine from {@link BitsAndBytesConfig}. Mirrors HF's
     * {@code model.quantization_config} dispatch logic.
     */
    public static QuantState quantize(Tensor weight, BitsAndBytesConfig cfg) {
        return quantize(weight, cfg, cfg == null ? DEFAULT_BLOCKSIZE : cfg.getBlocksize());
    }

    public static QuantState quantize(Tensor weight, BitsAndBytesConfig cfg, int blocksize) {
        if (cfg != null && cfg.isLoadIn8Bit()) {
            return quantizeInt8(weight, blocksize);
        }
        String t = cfg == null ? TYPE_NF4 : cfg.getBnb4BitQuantType();
        boolean dq = cfg != null && cfg.isBnb4BitUseDoubleQuant();
        return quantizeBlockwise(weight, blocksize,
                (t == null ? TYPE_NF4 : t.toLowerCase(Locale.ROOT)), dq);
    }

    /** Quantize-then-dequantize; standard reconstruction-error test path. */
    public static Tensor quantizeDequantize(Tensor weight, BitsAndBytesConfig cfg) {
        return dequantize(quantize(weight, cfg));
    }

    /** Snake_case alias for {@link #quantize(Tensor, BitsAndBytesConfig)}. */
    public static QuantState quantize_blockwise(Tensor weight, BitsAndBytesConfig cfg) {
        return quantize(weight, cfg);
    }

    /** Snake_case alias for {@link #dequantize(QuantState)}. */
    public static Tensor dequantize_blockwise(QuantState state) {
        return dequantize(state);
    }

    // =========================================================================
    // Pack / unpack 4-bit
    // =========================================================================

    /** Pack float codes in [0,15] into nibbles (2 codes per byte). */
    public static byte[] pack4bit(float[] codes) {
        Objects.requireNonNull(codes, "codes");
        int n = codes.length;
        byte[] out = new byte[(n + 1) / 2];
        for (int i = 0; i < n; i += 2) {
            int lo = Math.max(0, Math.min(15, Math.round(codes[i])));
            int hi = (i + 1 < n) ? Math.max(0, Math.min(15, Math.round(codes[i + 1]))) : 0;
            out[i / 2] = (byte) ((hi << 4) | lo);
        }
        return out;
    }

    /** Unpack nibble-packed codes back to float indices. */
    public static float[] unpack4bit(byte[] packed, int numel) {
        Objects.requireNonNull(packed, "packed");
        if (numel < 0) throw new IllegalArgumentException("numel must be non-negative");
        float[] out = new float[numel];
        for (int i = 0; i < numel; i++) {
            int b = packed[i / 2] & 0xFF;
            out[i] = (i % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
        }
        return out;
    }

    /** Pack uint8 codes into nibbles (no clamping needed; values must be 0..15). */
    public static byte[] pack4bit(byte[] codes) {
        Objects.requireNonNull(codes, "codes");
        int n = codes.length;
        byte[] out = new byte[(n + 1) / 2];
        for (int i = 0; i < n; i += 2) {
            int lo = codes[i] & 0x0F;
            int hi = (i + 1 < n) ? (codes[i + 1] & 0x0F) : 0;
            out[i / 2] = (byte) ((hi << 4) | lo);
        }
        return out;
    }

    // =========================================================================
    // Memory estimation
    // =========================================================================

    public static long estimateMemoryBytes(long numel, String quantType) {
        return estimateMemoryBytes(numel, quantType, DEFAULT_BLOCKSIZE, false);
    }

    /**
     * Estimate memory (bytes) needed to store a tensor of {@code numel} elements
     * under {@code quantType}/{@code blocksize}/{@code doubleQuant}.
     */
    public static long estimateMemoryBytes(long numel, String quantType, int blocksize, boolean doubleQuant) {
        if (numel < 0) throw new IllegalArgumentException("numel must be non-negative");
        if (blocksize <= 0) blocksize = DEFAULT_BLOCKSIZE;
        long blocks = (numel + blocksize - 1) / blocksize;
        return switch (quantType == null ? TYPE_FP32 : quantType.toLowerCase(Locale.ROOT)) {
            case TYPE_NF4, TYPE_FP4 -> {
                long codeBytes = (numel + 1) / 2; // packed nibbles
                long scaleBytes = doubleQuant
                        ? blocks + 4   // int8 nested codes + nested scale
                        : blocks * 4L; // fp32 absmax
                yield codeBytes + scaleBytes;
            }
            case TYPE_INT8 -> numel + blocks * 4L;
            case TYPE_FP16, "float16", "half" -> numel * 2L;
            case TYPE_BF16, "bfloat16" -> numel * 2L;
            default -> numel * 4L;
        };
    }

    public static double compressionRatio(long numel, String quantType, boolean doubleQuant) {
        long fp32 = numel * 4L;
        long q = estimateMemoryBytes(numel, quantType, DEFAULT_BLOCKSIZE, doubleQuant);
        return q == 0 ? 0.0 : (double) fp32 / (double) q;
    }

    /** Aggregate memory estimate for a map of named linears. */
    public static long estimateModelMemoryBytes(Map<String, LinearImpl> linears,
                                                 BitsAndBytesConfig cfg) {
        BitsAndBytesConfig c = cfg == null ? BitsAndBytesConfig.qloraDefaults() : cfg;
        long total = 0;
        for (Map.Entry<String, LinearImpl> e : linears.entrySet()) {
            if (c.shouldSkipModule(e.getKey())) continue;
            Tensor w = e.getValue() == null ? null : e.getValue().weight();
            if (w == null || !w.defined()) continue;
            total += estimateMemoryBytes(w.numel(), c.isLoadIn8Bit() ? TYPE_INT8 : c.getBnb4BitQuantType(),
                    c.getBlocksize(), c.isBnb4BitUseDoubleQuant());
        }
        return total;
    }

    // =========================================================================
    // Linear8bitLt (LLM.int8() style)
    // =========================================================================

    /**
     * 8-bit linear layer (bnb {@code Linear8bitLt} API surface). Forward
     * dequantizes weight then runs standard matmul.
     *
     * <p>When the underlying {@link QuantState} contains outlier rows
     * ({@link QuantState#outlierIndices}), those rows are routed through fp16
     * to mitigate LLM.int8()'s systematic degradation — this is the same
     * strategy bnb uses with the {@code threshold} parameter.
     */
    public static final class Linear8bitLt implements AutoCloseable {
        private final QuantState weightState;
        private final Tensor bias;
        private final long inFeatures;
        private final long outFeatures;
        private final boolean hasFp16Weights;
        private final double threshold;
        private Tensor cachedWeight; // dequant cache
        private boolean closed;

        public Linear8bitLt(QuantState weightState, Tensor bias, long inFeatures, long outFeatures) {
            this(weightState, bias, inFeatures, outFeatures, false, DEFAULT_LLM_INT8_THRESHOLD);
        }

        public Linear8bitLt(QuantState weightState, Tensor bias, long inFeatures, long outFeatures,
                            boolean hasFp16Weights, double threshold) {
            this.weightState = Objects.requireNonNull(weightState, "weightState");
            this.bias = bias;
            this.inFeatures = inFeatures;
            this.outFeatures = outFeatures;
            this.hasFp16Weights = hasFp16Weights;
            this.threshold = Double.isFinite(threshold) ? threshold : DEFAULT_LLM_INT8_THRESHOLD;
            if (this.threshold < 0) {
                throw new IllegalArgumentException("LLM.int8 threshold must be non-negative, got " + threshold);
            }
        }

        /** Forward pass: {@code y = x Wᵀ + b}. */
        public Tensor forward(Tensor input) {
            if (closed) throw new IllegalStateException("Layer is closed");
            Objects.requireNonNull(input, "input");
            if (input.size(-1) != inFeatures) {
                throw new IllegalArgumentException(
                        "Input last dim " + input.size(-1) + " != inFeatures " + inFeatures);
            }
            Tensor w = weight();
            if (bias == null) return linear(input, w);
            return linear(input, w, new TensorOptional(bias));
        }

        /** Dequantized weight (cached). */
        public Tensor weight() {
            ensureOpen();
            if (cachedWeight == null || !cachedWeight.defined()) {
                cachedWeight = dequantizeInt8(weightState);
            }
            return cachedWeight;
        }

        public QuantState weightState() { return weightState; }
        public Tensor bias() { return bias; }
        public long inFeatures() { return inFeatures; }
        public long outFeatures() { return outFeatures; }
        public boolean hasFp16Weights() { return hasFp16Weights; }
        public double threshold() { return threshold; }

        /** Replace underlying dense weight in-place (requires_grad=false). */
        public void materializeInto(LinearImpl dense) {
            ensureOpen();
            Objects.requireNonNull(dense, "dense");
            Tensor w = weight();
            try (NoGradGuard guard = new NoGradGuard()) {
                if (!dense.weight().defined()) {
                    throw new IllegalStateException("dense.weight is undefined");
                }
                dense.weight().requires_grad_(false);
                dense.weight().copy_(w);
                if (bias != null && dense.bias() != null && dense.bias().defined()) {
                    dense.bias().requires_grad_(false);
                    dense.bias().copy_(bias);
                }
            }
        }

        private void ensureOpen() {
            if (closed) throw new IllegalStateException("Layer is closed");
        }

        @Override
        public void close() {
            if (closed) return;
            if (cachedWeight != null) {
                try { cachedWeight.close(); } catch (Exception ignored) {}
                cachedWeight = null;
            }
            closed = true;
        }
    }

    public static Linear8bitLt linear8bit(LinearImpl dense) {
        return linear8bit(dense, null);
    }

    public static Linear8bitLt linear8bit(LinearImpl dense, BitsAndBytesConfig cfg) {
        Objects.requireNonNull(dense, "dense");
        Tensor w = dense.weight();
        Tensor b = safeBias(dense);
        int bs = cfg == null ? DEFAULT_BLOCKSIZE : cfg.getBlocksize();
        QuantState qs = quantizeInt8(w, bs);
        boolean hasFp16 = cfg != null && cfg.isLlmInt8HasFp16Weight();
        double thr = cfg == null ? DEFAULT_LLM_INT8_THRESHOLD : cfg.getLlmInt8Threshold();
        return new Linear8bitLt(qs, b, w.size(1), w.size(0), hasFp16, thr);
    }

    public static Linear8bitLt linear8bit(long inFeatures, long outFeatures) {
        return linear8bit(new LinearImpl(inFeatures, outFeatures));
    }

    /** Python-style snake_case alias. */
    public static Linear8bitLt Linear8bitLt(LinearImpl dense, BitsAndBytesConfig cfg) {
        return linear8bit(dense, cfg);
    }

    // =========================================================================
    // Linear4bit (QLoRA / HF Linear4bit)
    // =========================================================================

    /**
     * 4-bit linear layer — mirrors bnb's {@code Linear4bit} and HF's
     * {@code bnb.nn.Linear4bit}.
     */
    public static final class Linear4bit implements AutoCloseable {
        private final QuantState weightState;
        private final Tensor bias;
        private final long inFeatures;
        private final long outFeatures;
        private final String computeDtype;
        private final boolean quantStorage;
        private Tensor cachedWeight;
        private boolean closed;

        public Linear4bit(QuantState weightState, Tensor bias,
                          long inFeatures, long outFeatures, String computeDtype) {
            this.weightState = Objects.requireNonNull(weightState, "weightState");
            this.bias = bias;
            this.inFeatures = inFeatures;
            this.outFeatures = outFeatures;
            this.computeDtype = computeDtype == null ? TYPE_FP32 : computeDtype;
            this.quantStorage = weightState.packedCodes != null;
            if (!this.computeDtype.equalsIgnoreCase(TYPE_FP32)
                    && !this.computeDtype.equalsIgnoreCase(TYPE_FP16)
                    && !this.computeDtype.equalsIgnoreCase(TYPE_BF16)) {
                throw new IllegalArgumentException(
                        "computeDtype must be one of {float32, fp16, bfloat16}, got '" + computeDtype + "'");
            }
        }

        public Tensor forward(Tensor input) {
            ensureOpen();
            Objects.requireNonNull(input, "input");
            if (input.size(-1) != inFeatures) {
                throw new IllegalArgumentException(
                        "Input last dim " + input.size(-1) + " != inFeatures " + inFeatures);
            }
            Tensor w = weight();
            if (bias == null) return linear(input, w);
            return linear(input, w, new TensorOptional(bias));
        }

        /** Dequantized weight (cached). */
        public Tensor weight() {
            ensureOpen();
            if (cachedWeight == null || !cachedWeight.defined()) {
                cachedWeight = dequantize(weightState);
            }
            return cachedWeight;
        }

        public QuantState weightState() { return weightState; }
        public Tensor bias() { return bias; }
        public String computeDtype() { return computeDtype; }
        public long inFeatures() { return inFeatures; }
        public long outFeatures() { return outFeatures; }
        public boolean quantStorage() { return quantStorage; }

        public void materializeInto(LinearImpl dense) {
            ensureOpen();
            Objects.requireNonNull(dense, "dense");
            Tensor w = weight();
            try (NoGradGuard guard = new NoGradGuard()) {
                if (!dense.weight().defined()) {
                    throw new IllegalStateException("dense.weight is undefined");
                }
                dense.weight().requires_grad_(false);
                dense.weight().copy_(w);
                if (bias != null && dense.bias() != null && dense.bias().defined()) {
                    dense.bias().requires_grad_(false);
                    dense.bias().copy_(bias);
                }
            }
        }

        public Map<String, Object> stats() {
            Map<String, Object> m = new LinkedHashMap<>(weightState.toMap());
            m.put("in_features", inFeatures);
            m.put("out_features", outFeatures);
            m.put("compute_dtype", computeDtype);
            m.put("quant_storage", quantStorage);
            return m;
        }

        private void ensureOpen() {
            if (closed) throw new IllegalStateException("Layer is closed");
        }

        @Override
        public void close() {
            if (closed) return;
            if (cachedWeight != null) {
                try { cachedWeight.close(); } catch (Exception ignored) {}
                cachedWeight = null;
            }
            closed = true;
        }
    }

    public static Linear4bit linear4bit(LinearImpl dense, BitsAndBytesConfig cfg) {
        Objects.requireNonNull(dense, "dense");
        BitsAndBytesConfig c = cfg == null
                ? BitsAndBytesConfig.builder().loadIn4Bit(true).build()
                : cfg;
        Tensor w = dense.weight();
        if (w == null || !w.defined()) {
            throw new IllegalArgumentException("dense.weight is undefined");
        }
        Tensor b = safeBias(dense);
        QuantState qs = quantize(w, c, c.getBlocksize());
        return new Linear4bit(qs, b, w.size(1), w.size(0), c.getBnb4BitComputeDtype());
    }

    public static Linear4bit linear4bit(long inFeatures, long outFeatures, BitsAndBytesConfig cfg) {
        return linear4bit(new LinearImpl(inFeatures, outFeatures), cfg);
    }

    public static Linear4bit linear4bit(long inFeatures, long outFeatures) {
        return linear4bit(inFeatures, outFeatures,
                BitsAndBytesConfig.builder().loadIn4Bit(true).build());
    }

    /** Python-style snake_case alias. */
    public static Linear4bit Linear4bit(LinearImpl dense, BitsAndBytesConfig cfg) {
        return linear4bit(dense, cfg);
    }

    // =========================================================================
    // Params4bit (HuggingFace bnb.nn.Params4bit)
    // =========================================================================

    /**
     * A leaf-tensor wrapper that bnb/HF use to hold 4-bit weights separately
     * from the {@code int8}-quantized absmax. Useful for PEFT LoRA bindings
     * which expect to mutate {@code weight.data} in-place during adapter
     * training.
     */
    public static final class Params4bit implements AutoCloseable {
        /** Optional nibble-packed storage (2 codes / byte). May be null. */
        public final byte[] data;
        /** Quantization state. */
        public final QuantState quantState;
        /** Public shape (logical numel). */
        public final long[] shape;
        /** Optional underlying dense tensor (for HF {@code .weight.data} lookups). */
        public final Tensor backingTensor;
        private boolean closed;

        public Params4bit(byte[] data, QuantState quantState, long[] shape) {
            this(data, quantState, shape, null);
        }

        public Params4bit(byte[] data, QuantState quantState, long[] shape, Tensor backingTensor) {
            this.data = data;
            this.quantState = Objects.requireNonNull(quantState, "quantState");
            this.shape = Objects.requireNonNull(shape, "shape").clone();
            this.backingTensor = backingTensor;
        }

        public long numel() {
            long n = 1;
            for (long s : shape) n *= s;
            return n;
        }

        public Tensor dequantize() {
            return BitsAndBytes.dequantize(quantState);
        }

        /** Materialize as a regular float tensor (allocate + close on caller). */
        public Tensor asFloatTensor() {
            Tensor t = dequantize();
            return t;
        }

        @Override
        public void close() {
            if (closed) return;
            closed = true;
        }
    }

    /**
     * Build a {@link Params4bit} from an existing dense tensor by quantizing
     * it according to {@link BitsAndBytesConfig}. Mirrors HF's
     * {@code torch.nn.utils.parametrize.register_parametrization} output.
     */
    public static Params4bit params4bit(Tensor weight, BitsAndBytesConfig cfg) {
        requireTensor(weight, "weight");
        BitsAndBytesConfig c = cfg == null ? BitsAndBytesConfig.qloraDefaults() : cfg;
        QuantState qs = quantize(weight, c, c.getBlocksize());
        return new Params4bit(qs.packedCodes, qs, qs.originalShape, null);
    }

    // =========================================================================
    // Int8Params (LLM.int8() column-wise)
    // =========================================================================

    /**
     * Column-wise INT8 state — bnb stores per-row int8 codes plus an outlier
     * fp16 mask for the small fraction of activations/columns that exceed
     * {@code threshold} standard deviations.
     */
    public static final class Int8Params implements AutoCloseable {
        public final Tensor qweight;
        public final Tensor scales;
        public final int[] outlierIndices;
        public final float[] outlierValues;
        public final long[] shape;
        private boolean closed;

        public Int8Params(Tensor qweight, Tensor scales, int[] outlierIndices,
                          float[] outlierValues, long[] shape) {
            this.qweight = Objects.requireNonNull(qweight, "qweight");
            this.scales = Objects.requireNonNull(scales, "scales");
            this.outlierIndices = outlierIndices == null ? new int[0] : outlierIndices;
            this.outlierValues = outlierValues == null ? new float[0] : outlierValues;
            if (this.outlierIndices.length != this.outlierValues.length) {
                throw new IllegalArgumentException("outlier indices/values length mismatch");
            }
            this.shape = Objects.requireNonNull(shape, "shape").clone();
        }

        public long numel() {
            long n = 1;
            for (long s : shape) n *= s;
            return n;
        }

        public int outlierCount() { return outlierIndices.length; }

        public QuantState toQuantState(int blocksize) {
            return new QuantState(qweight, scales, blocksize, TYPE_INT8, shape, false,
                    null, "int8", null, 1f, outlierIndices, outlierValues);
        }

        public Tensor dequantize() {
            float[] codes = toFloatArray(qweight);
            float[] scales = toFloatArray(this.scales);
            float[] out = new float[codes.length];
            for (int i = 0; i < codes.length; i++) {
                int b = Math.min(scales.length - 1, i / Math.max(1, blocksize()));
                out[i] = codes[i] * scales[b];
            }
            Tensor t = tensor(out).reshape(shape);
            if (outlierIndices.length == 0) return t;
            float[] data = toFloatArray(t);
            long cols = shape[shape.length - 1];
            for (int i = 0; i < outlierIndices.length; i++) {
                int idx = outlierIndices[i];
                long r = idx / Math.max(1, cols);
                long c = idx % Math.max(1, cols);
                if (r >= 0 && r < shape[0] && c >= 0 && c < cols) {
                    data[(int) (r * cols + c)] = outlierValues[i];
                }
            }
            return tensor(data).reshape(shape);
        }

        private int blocksize() {
            // Approximate block size (codes.length / scales.length)
            if (scales.numel() <= 0) return DEFAULT_BLOCKSIZE;
            long n = qweight.numel();
            return (int) Math.max(1, n / scales.numel());
        }

        @Override
        public void close() {
            if (closed) return;
            closed = true;
        }
    }

    /**
     * Find column outliers by std-dev threshold. {@code threshold} is the
     * multiplier on the per-column absolute max; rows whose column max
     * exceeds {@code mean + threshold * std} are recorded for fp16 routing.
     */
    public static int[] findOutlierColumns(float[] data, long cols, double threshold) {
        Objects.requireNonNull(data, "data");
        if (cols <= 0) throw new IllegalArgumentException("cols must be positive");
        long rows = (data.length + cols - 1) / cols;
        double[] colMax = new double[(int) cols];
        for (long r = 0; r < rows; r++) {
            for (long c = 0; c < cols; c++) {
                long idx = r * cols + c;
                if (idx >= data.length) break;
                double v = Math.abs(data[(int) idx]);
                if (v > colMax[(int) c]) colMax[(int) c] = v;
            }
        }
        double mean = 0, var = 0;
        for (double v : colMax) mean += v;
        mean /= Math.max(1, colMax.length);
        for (double v : colMax) var += (v - mean) * (v - mean);
        var = colMax.length > 0 ? var / colMax.length : 0.0;
        double std = Math.sqrt(var);
        double cutoff = mean + threshold * std;
        List<Integer> idx = new ArrayList<>();
        for (int c = 0; c < colMax.length; c++) {
            if (colMax[c] > cutoff) idx.add(c);
        }
        int[] out = new int[idx.size()];
        for (int i = 0; i < idx.size(); i++) out[i] = idx.get(i);
        return out;
    }

    /**
     * Build {@link Int8Params} by column-wise int8 quantization with outlier
     * detection. Each column of {@code weight} (last dim) is quantized with
     * its own scale.
     */
    public static Int8Params int8ParamsColumnwise(Tensor weight, double threshold) {
        requireTensor(weight, "weight");
        if (weight.dim() < 2) {
            throw new IllegalArgumentException("int8ParamsColumnwise expects a tensor with dim>=2, got " + weight.dim());
        }
        Tensor w = weight.to(ScalarType.Float).contiguous();
        long rows = 1;
        for (int i = 0; i < w.dim() - 1; i++) rows *= w.size(i);
        long cols = w.size(w.dim() - 1);
        float[] data = toFloatArray(w);
        int[] outliers = findOutlierColumns(data, cols, threshold);
        boolean[] isOutlier = new boolean[(int) cols];
        for (int c : outliers) if (c < isOutlier.length) isOutlier[c] = true;

        float[] codes = new float[data.length];
        float[] scales = new float[(int) rows];
        float[] outlierVals = new float[outliers.length];
        int outIdx = 0;
        for (long r = 0; r < rows; r++) {
            float amax = 0f;
            for (long c = 0; c < cols; c++) {
                long idx = r * cols + c;
                if (idx >= data.length) break;
                if (isOutlier[(int) c]) continue;
                float a = Math.abs(data[(int) idx]);
                if (a > amax) amax = a;
            }
            if (amax < 1e-12f) amax = 1e-12f;
            scales[(int) r] = amax / 127f;
            for (long c = 0; c < cols; c++) {
                long idx = r * cols + c;
                if (idx >= data.length) break;
                if (isOutlier[(int) c]) {
                    outlierVals[outIdx++] = data[(int) idx];
                    codes[(int) idx] = 0; // placeholder; outlier array replaces
                } else {
                    int q = Math.round(data[(int) idx] / scales[(int) r]);
                    q = Math.max(-128, Math.min(127, q));
                    codes[(int) idx] = q;
                }
            }
        }
        long[] shape = new long[(int) weight.dim()];
        for (int i = 0; i < shape.length; i++) shape[i] = weight.size(i);
        // Record outlier indices (flat) for reproducibility.
        int[] flatOutliers = new int[outliers.length];
        for (int c : outliers) {
            // Map to first row's column (flat indices of column-major outliers)
            flatOutliers[c] = c;
        }
        // Use the per-column index list we already computed.
        int[] idxList = new int[outliers.length];
        for (int i = 0; i < outliers.length; i++) idxList[i] = outliers[i];
        return new Int8Params(tensor(codes), tensor(scales), idxList,
                copyOf(outlierVals, outliers.length), shape);
    }

    private static float[] copyOf(float[] src, int len) {
        float[] out = new float[len];
        System.arraycopy(src, 0, out, 0, Math.min(src.length, len));
        return out;
    }

    // =========================================================================
    // HF-style state_dict (de)serialization
    // =========================================================================

    /**
     * Serialize a quantized layer's state into a HF safetensors-friendly map.
     * Mirrors the structure written by bnb's
     * {@code save_pretrained_fragmented} for one linear layer.
     */
    public static Map<String, Object> toStateDict(Linear4bit layer) {
        Objects.requireNonNull(layer, "layer");
        Map<String, Object> m = new LinkedHashMap<>();
        QuantState qs = layer.weightState();
        m.put("weight", qs.packedCodes != null
                ? qs.packedCodes
                : toFloatArray(qs.qweight));
        Map<String, Object> qsMap = new LinkedHashMap<>();
        qsMap.put("quant_type", qs.quantType);
        qsMap.put("blocksize", qs.blocksize);
        qsMap.put("double_quant", qs.doubleQuant);
        qsMap.put("shape", qs.originalShape.clone());
        qsMap.put("code_dtype", qs.codeDtype);
        qsMap.put("nested_scale", qs.nestedScale);
        qsMap.put("absmax", toFloatArray(qs.absmax));
        if (qs.nested != null) {
            Map<String, Object> nested = new LinkedHashMap<>();
            nested.put("absmax", toFloatArray(qs.nested.absmax));
            nested.put("qweight", toFloatArray(qs.nested.qweight));
            nested.put("quant_type", qs.nested.quantType);
            nested.put("blocksize", qs.nested.blocksize);
            qsMap.put("nested", nested);
        }
        m.put("quant_state", qsMap);
        if (layer.bias() != null) {
            m.put("bias", toFloatArray(layer.bias()));
        }
        m.put("compute_dtype", layer.computeDtype());
        m.put("in_features", layer.inFeatures());
        m.put("out_features", layer.outFeatures());
        return m;
    }

    /**
     * Reconstruct a {@link Linear4bit} from a state-dict map written by
     * {@link #toStateDict(Linear4bit)} or by Python bnb/HF.
     */
    @SuppressWarnings("unchecked")
    public static Linear4bit fromStateDict(Map<String, Object> sd) {
        Objects.requireNonNull(sd, "sd");
        Object w = sd.get("weight");
        Object b = sd.get("bias");
        Object qso = sd.get("quant_state");
        if (qso == null || !(qso instanceof Map)) {
            throw new IllegalArgumentException("state_dict missing 'quant_state' map");
        }
        Map<String, Object> qs = (Map<String, Object>) qso;
        String quantType = (String) qs.get("quant_type");
        int blocksize = ((Number) qs.get("blocksize")).intValue();
        boolean doubleQuant = Boolean.TRUE.equals(qs.get("double_quant"));
        long[] shape = toLongArray(qs.get("shape"));
        String codeDtype = (String) qs.get("code_dtype");
        float nestedScale = qs.get("nested_scale") instanceof Number
                ? ((Number) qs.get("nested_scale")).floatValue() : 1f;

        Tensor absmax = tensor(toFloatArray((Object[]) qs.get("absmax")));
        Tensor qweight;
        byte[] packed = null;
        if (w instanceof byte[]) {
            packed = (byte[]) w;
            float[] codes = unpack4bit(packed, (int) (shape[0] * shape[1]));
            qweight = tensor(codes);
        } else {
            qweight = tensor(toFloatArray((Object[]) w));
        }
        QuantState nested = null;
        Object nestedObj = qs.get("nested");
        if (nestedObj instanceof Map) {
            Map<String, Object> nm = (Map<String, Object>) nestedObj;
            Tensor nAbs = tensor(toFloatArray((Object[]) nm.get("absmax")));
            Tensor nQw = tensor(toFloatArray((Object[]) nm.get("qweight")));
            int nBs = ((Number) nm.get("blocksize")).intValue();
            String nQt = (String) nm.get("quant_type");
            nested = new QuantState(nQw, nAbs, nBs, nQt,
                    new long[]{nAbs.numel()}, false);
        }
        QuantState full = new QuantState(qweight, absmax, blocksize, quantType, shape,
                doubleQuant, nested, codeDtype, packed, nestedScale);
        Tensor biasT = b == null ? null : tensor(toFloatArray((Object[]) b));
        long inF = sd.get("in_features") instanceof Number
                ? ((Number) sd.get("in_features")).longValue() : shape[1];
        long outF = sd.get("out_features") instanceof Number
                ? ((Number) sd.get("out_features")).longValue() : shape[0];
        String comp = (String) sd.getOrDefault("compute_dtype", TYPE_FP32);
        return new Linear4bit(full, biasT, inF, outF, comp);
    }

    /** Save a state-dict map to a Java-properties-style text file. */
    public static void saveStateDict(Map<String, Object> sd, Path file) throws java.io.IOException {
        Objects.requireNonNull(sd, "sd");
        Objects.requireNonNull(file, "file");
        StringBuilder sb = new StringBuilder();
        writeMap(sb, sd, 0);
        Files.writeString(file, sb.toString());
    }

    private static void writeMap(StringBuilder sb, Map<String, Object> m, int indent) {
        for (Map.Entry<String, Object> e : m.entrySet()) {
            pad(sb, indent);
            sb.append(e.getKey()).append(" = ");
            Object v = e.getValue();
            if (v instanceof Map) {
                sb.append('\n');
                writeMap(sb, (Map<String, Object>) v, indent + 1);
            } else if (v instanceof float[]) {
                sb.append("float[").append(((float[]) v).length).append("]\n");
            } else if (v instanceof byte[]) {
                sb.append("byte[").append(((byte[]) v).length).append("]\n");
            } else {
                sb.append(String.valueOf(v)).append('\n');
            }
        }
    }

    private static void pad(StringBuilder sb, int indent) {
        for (int i = 0; i < indent; i++) sb.append("  ");
    }

    private static float[] toFloatArray(Object o) {
        if (o == null) return new float[0];
        if (o instanceof float[]) return (float[]) o;
        if (o instanceof Object[]) {
            Object[] arr = (Object[]) o;
            float[] out = new float[arr.length];
            for (int i = 0; i < arr.length; i++) {
                out[i] = ((Number) arr[i]).floatValue();
            }
            return out;
        }
        if (o instanceof Number) return new float[]{((Number) o).floatValue()};
        throw new IllegalArgumentException("Cannot convert " + o.getClass() + " to float[]");
    }

    private static long[] toLongArray(Object o) {
        if (o == null) return new long[0];
        if (o instanceof long[]) return (long[]) o;
        if (o instanceof int[]) {
            int[] a = (int[]) o;
            long[] out = new long[a.length];
            for (int i = 0; i < a.length; i++) out[i] = a[i];
            return out;
        }
        if (o instanceof Object[]) {
            Object[] arr = (Object[]) o;
            long[] out = new long[arr.length];
            for (int i = 0; i < arr.length; i++) out[i] = ((Number) arr[i]).longValue();
            return out;
        }
        throw new IllegalArgumentException("Cannot convert " + o.getClass() + " to long[]");
    }

    // =========================================================================
    // 8-bit optimizer stubs (Adam8bit / Lion8bit / PagedAdam8bit)
    // =========================================================================

    /**
     * Common accumulator for {@link Adam8bit} / {@link Lion8bit}. bnb keeps the
     * optimizer state (m, v) in 8-bit to reduce GPU/CPU memory pressure.
     * The Java reference port is a faithful byte-level representation usable
     * for I/O and offline inspection — the actual optimizer step still relies
     * on libtorch autograd via the wrapped model parameters.
     */
    public static final class BnbOptimizerState implements AutoCloseable {
        /** Block size used to quantize optimizer state tensors. */
        public final int blocksize;
        public final Tensor qState1; // int8 packed first moment
        public final Tensor absmax1;
        public final Tensor qState2; // int8 packed second moment (Adam only)
        public final Tensor absmax2;
        private boolean closed;

        public BnbOptimizerState(int blocksize, Tensor qState1, Tensor absmax1,
                                 Tensor qState2, Tensor absmax2) {
            this.blocksize = blocksize > 0 ? blocksize : DEFAULT_BLOCKSIZE;
            this.qState1 = qState1;
            this.absmax1 = absmax1;
            this.qState2 = qState2;
            this.absmax2 = absmax2;
        }

        public boolean hasSecondMoment() { return qState2 != null; }

        public long memoryBytes() {
            long n1 = qState1 == null ? 0 : qState1.numel();
            long n2 = qState2 == null ? 0 : qState2.numel();
            return n1 + n2 + ((absmax1 == null ? 0 : absmax1.numel())
                    + (absmax2 == null ? 0 : absmax2.numel())) * 4L;
        }

        @Override
        public void close() {
            if (closed) return;
            closed = true;
        }
    }

    /**
     * Stub for bnb's {@code Adam8bit} / {@code PagedAdam8bit} optimizer. Use
     * {@link #quantizeState} to compress first/second moments to int8.
     */
    public static final class Adam8bit {
        private final double beta1, beta2, eps;
        private final double lr;
        private final int weightDecay;
        private final int blocksize;
        private BnbOptimizerState state;

        public Adam8bit() { this(0.9, 0.999, 1e-8, 1e-3, 0, DEFAULT_BLOCKSIZE); }
        public Adam8bit(double beta1, double beta2, double eps,
                        double lr, int weightDecay, int blocksize) {
            if (!(beta1 >= 0 && beta1 < 1)) throw new IllegalArgumentException("beta1 out of range");
            if (!(beta2 >= 0 && beta2 < 1)) throw new IllegalArgumentException("beta2 out of range");
            if (eps <= 0) throw new IllegalArgumentException("eps must be positive");
            if (lr < 0) throw new IllegalArgumentException("lr must be non-negative");
            if (blocksize <= 0) throw new IllegalArgumentException("blocksize must be positive");
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.lr = lr;
            this.weightDecay = weightDecay;
            this.blocksize = blocksize;
        }

        public BnbOptimizerState quantizeState(Tensor grad) {
            requireTensor(grad, "grad");
            QuantState qs1 = quantizeInt8(grad, blocksize);
            QuantState qs2 = quantizeInt8(grad.mul(grad), blocksize);
            this.state = new BnbOptimizerState(blocksize,
                    qs1.qweight, qs1.absmax,
                    qs2.qweight, qs2.absmax);
            return state;
        }

        public BnbOptimizerState state() { return state; }

        public Map<String, Object> config() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("type", "Adam8bit");
            m.put("beta1", beta1);
            m.put("beta2", beta2);
            m.put("eps", eps);
            m.put("lr", lr);
            m.put("weight_decay", weightDecay);
            m.put("blocksize", blocksize);
            return m;
        }
    }

    /**
     * Stub for bnb's {@code Lion8bit} / {@code PagedLion8bit} optimizer —
     * single-state (sign-momentum) variant.
     */
    public static final class Lion8bit {
        private final double beta1, beta2, lr, weightDecay;
        private final int blocksize;
        private BnbOptimizerState state;

        public Lion8bit() { this(0.9, 0.99, 1e-4, 0.0, DEFAULT_BLOCKSIZE); }
        public Lion8bit(double beta1, double beta2, double lr,
                        double weightDecay, int blocksize) {
            if (!(beta1 >= 0 && beta1 < 1)) throw new IllegalArgumentException("beta1 out of range");
            if (!(beta2 >= 0 && beta2 < 1)) throw new IllegalArgumentException("beta2 out of range");
            if (blocksize <= 0) throw new IllegalArgumentException("blocksize must be positive");
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.lr = lr;
            this.weightDecay = weightDecay;
            this.blocksize = blocksize;
        }

        public BnbOptimizerState quantizeState(Tensor grad) {
            requireTensor(grad, "grad");
            QuantState qs = quantizeInt8(grad, blocksize);
            this.state = new BnbOptimizerState(blocksize,
                    qs.qweight, qs.absmax, null, null);
            return state;
        }

        public BnbOptimizerState state() { return state; }

        public Map<String, Object> config() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("type", "Lion8bit");
            m.put("beta1", beta1);
            m.put("beta2", beta2);
            m.put("lr", lr);
            m.put("weight_decay", weightDecay);
            m.put("blocksize", blocksize);
            return m;
        }
    }

    /** Snake_case alias matching Python bnb. */
    public static Adam8bit Adam8bit() { return new Adam8bit(); }
    /** Snake_case alias matching Python bnb. */
    public static Lion8bit Lion8bit() { return new Lion8bit(); }

    // =========================================================================
    // Model-level helpers
    // =========================================================================

    /**
     * Result of quantizing a collection of named linears (HF-style
     * {@code replace_with_bnb_linear} bookkeeping without Module-tree rewrite).
     */
    public static final class QuantizedModel implements AutoCloseable {
        private final Map<String, Object> layers;
        private final Map<String, QuantState> states;
        private final BitsAndBytesConfig config;
        private final long totalParams;
        private final long quantMemoryBytes;
        private final long fp32MemoryBytes;
        private final long skippedParams;

        public QuantizedModel(Map<String, Object> layers, Map<String, QuantState> states,
                              BitsAndBytesConfig config, long totalParams, long quantMemoryBytes,
                              long fp32MemoryBytes, long skippedParams) {
            this.layers = Collections.unmodifiableMap(new LinkedHashMap<>(layers));
            this.states = Collections.unmodifiableMap(new LinkedHashMap<>(states));
            this.config = config;
            this.totalParams = totalParams;
            this.quantMemoryBytes = quantMemoryBytes;
            this.fp32MemoryBytes = fp32MemoryBytes;
            this.skippedParams = skippedParams;
        }

        public Map<String, Object> layers() { return layers; }
        public Map<String, QuantState> states() { return states; }
        public BitsAndBytesConfig config() { return config; }
        public long totalParams() { return totalParams; }
        public long quantMemoryBytes() { return quantMemoryBytes; }
        public long fp32MemoryBytes() { return fp32MemoryBytes; }
        public long skippedParams() { return skippedParams; }
        public int size() { return layers.size(); }

        public Object get(String name) { return layers.get(name); }
        public QuantState state(String name) { return states.get(name); }

        public Linear4bit as4bit(String name) {
            Object o = layers.get(name);
            return o instanceof Linear4bit l ? l : null;
        }

        public Linear8bitLt as8bit(String name) {
            Object o = layers.get(name);
            return o instanceof Linear8bitLt l ? l : null;
        }

        public int materializeInto(Map<String, LinearImpl> dense) {
            Objects.requireNonNull(dense, "dense");
            int n = 0;
            for (Map.Entry<String, Object> e : layers.entrySet()) {
                LinearImpl d = dense.get(e.getKey());
                if (d == null) continue;
                Object layer = e.getValue();
                if (layer instanceof Linear4bit l4) {
                    l4.materializeInto(d);
                    n++;
                } else if (layer instanceof Linear8bitLt l8) {
                    l8.materializeInto(d);
                    n++;
                }
            }
            return n;
        }

        /** Iterate layer names in insertion order. */
        public List<String> layerNames() {
            return new ArrayList<>(layers.keySet());
        }

        /** Iterate layer names matching a substring filter. */
        public List<String> layerNamesContaining(String filter) {
            List<String> out = new ArrayList<>();
            if (filter == null) return out;
            String f = filter.toLowerCase(Locale.ROOT);
            for (String n : layers.keySet()) {
                if (n != null && n.toLowerCase(Locale.ROOT).contains(f)) out.add(n);
            }
            return out;
        }

        /** For each layer, run a visitor. Used by callers wanting per-layer stats. */
        public void forEach(BiConsumer<String, Object> visitor) {
            Objects.requireNonNull(visitor, "visitor");
            for (Map.Entry<String, Object> e : layers.entrySet()) visitor.accept(e.getKey(), e.getValue());
        }

        public Map<String, Object> stats() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("num_layers", layers.size());
            m.put("total_params", totalParams);
            m.put("skipped_params", skippedParams);
            m.put("quant_memory_bytes", quantMemoryBytes);
            m.put("fp32_memory_bytes", fp32MemoryBytes);
            m.put("compression_ratio", fp32MemoryBytes == 0
                    ? 0.0
                    : (double) fp32MemoryBytes / (double) Math.max(1, quantMemoryBytes));
            m.put("config", config == null ? null : config.toString());
            m.put("load_in_4bit", config != null && config.isLoadIn4Bit());
            m.put("load_in_8bit", config != null && config.isLoadIn8Bit());
            return m;
        }

        @Override
        public void close() {
            for (Object o : layers.values()) {
                if (o instanceof AutoCloseable c) {
                    try { c.close(); } catch (Exception ignored) {}
                }
            }
        }
    }

    /**
     * Quantize a map of named {@link LinearImpl}s according to {@code cfg}.
     * Skips modules listed in {@link BitsAndBytesConfig#shouldSkipModule(String)}.
     */
    public static QuantizedModel quantizeModel(Map<String, LinearImpl> linears, BitsAndBytesConfig cfg) {
        Objects.requireNonNull(linears, "linears");
        BitsAndBytesConfig c = cfg == null ? BitsAndBytesConfig.qloraDefaults() : cfg;
        Map<String, Object> layers = new LinkedHashMap<>();
        Map<String, QuantState> states = new LinkedHashMap<>();
        long total = 0;
        long skipped = 0;
        long mem = 0;
        long fp32 = 0;
        for (Map.Entry<String, LinearImpl> e : linears.entrySet()) {
            String name = e.getKey();
            LinearImpl lin = e.getValue();
            if (lin == null || lin.weight() == null || !lin.weight().defined()) continue;
            if (c.shouldSkipModule(name)) {
                skipped += lin.weight().numel();
                continue;
            }
            long numel = lin.weight().numel();
            total += numel;
            fp32 += numel * 4L;
            if (c.isLoadIn8Bit()) {
                Linear8bitLt l8 = linear8bit(lin, c);
                layers.put(name, l8);
                states.put(name, l8.weightState());
                mem += l8.weightState().memoryBytes();
            } else {
                Linear4bit l4 = linear4bit(lin, c);
                layers.put(name, l4);
                states.put(name, l4.weightState());
                mem += l4.weightState().memoryBytes();
            }
        }
        return new QuantizedModel(layers, states, c, total, mem, fp32, skipped);
    }

    /** Snake_case alias matching Python bnb. */
    public static QuantizedModel quantize_model(Map<String, LinearImpl> linears, BitsAndBytesConfig cfg) {
        return quantizeModel(linears, cfg);
    }

    /**
     * Quantize then materialize dequantized (frozen) weights back into the
     * same linears — the practical path for Java Module graphs that cannot
     * freely swap submodule types. Returns the {@link QuantizedModel}.
     */
    public static QuantizedModel replaceLinearWithBnb(Map<String, LinearImpl> linears, BitsAndBytesConfig cfg) {
        QuantizedModel qm = quantizeModel(linears, cfg);
        qm.materializeInto(linears);
        return qm;
    }

    /** Snake_case alias for {@link #replaceLinearWithBnb}. */
    public static QuantizedModel replace_linear(Map<String, LinearImpl> linears, BitsAndBytesConfig cfg) {
        return replaceLinearWithBnb(linears, cfg);
    }

    /**
     * Freeze all parameters in {@code params} (set {@code requires_grad=False}).
     * Mirrors HF {@code prepare_model_for_kbit_training} base-weight freeze step.
     *
     * @return number of tensors frozen
     */
    public static int prepareModelForKbitTraining(TensorVector params) {
        if (params == null) return 0;
        int n = 0;
        for (long i = 0, m = params.size(); i < m; i++) {
            Tensor p = params.get((int) i);
            if (p != null && !p.isNull() && p.defined()) {
                p.requires_grad_(false);
                n++;
            }
        }
        return n;
    }

    /** Snake_case alias matching Python transformers / peft. */
    public static int prepare_model_for_kbit_training(TensorVector params) {
        return prepareModelForKbitTraining(params);
    }

    /** Freeze every parameter of a Module (base model for QLoRA). */
    public static int prepareModelForKbitTraining(Module model) {
        if (model == null) return 0;
        try {
            return prepareModelForKbitTraining(model.parameters());
        } catch (Exception e) {
            return 0;
        }
    }

    /** Snake_case alias for {@link #prepareModelForKbitTraining(Module)}. */
    public static int prepare_model_for_kbit_training(Module model) {
        return prepareModelForKbitTraining(model);
    }

    /**
     * Freeze base linears then leave them ready for LoRA injection
     * (quant-dequant materialize + freeze).
     */
    public static QuantizedModel prepareForQLoRA(Map<String, LinearImpl> linears, BitsAndBytesConfig cfg) {
        BitsAndBytesConfig c = cfg == null ? BitsAndBytesConfig.qloraDefaults() : cfg;
        QuantizedModel qm = replaceLinearWithBnb(linears, c);
        for (LinearImpl lin : linears.values()) {
            if (lin == null) continue;
            try {
                if (lin.weight() != null && lin.weight().defined()) {
                    lin.weight().requires_grad_(false);
                }
                Tensor b = safeBias(lin);
                if (b != null) b.requires_grad_(false);
            } catch (Exception ignored) {}
        }
        return qm;
    }

    /** Mean absolute reconstruction error after quantize→dequantize. */
    public static double reconstructionMae(Tensor weight, BitsAndBytesConfig cfg) {
        Tensor restored = quantizeDequantize(weight, cfg);
        try {
            float[] a = toFloatArray(weight.reshape(-1).to(ScalarType.Float));
            float[] b = toFloatArray(restored.reshape(-1).to(ScalarType.Float));
            double sum = 0;
            int n = Math.min(a.length, b.length);
            for (int i = 0; i < n; i++) sum += Math.abs(a[i] - b[i]);
            return n == 0 ? 0.0 : sum / n;
        } finally {
            try { restored.close(); } catch (Exception ignored) {}
        }
    }

    /** Cosine similarity between original and quant→dequant weight (1 = perfect). */
    public static double reconstructionCosine(Tensor weight, BitsAndBytesConfig cfg) {
        Tensor restored = quantizeDequantize(weight, cfg);
        try {
            float[] a = toFloatArray(weight.reshape(-1).to(ScalarType.Float));
            float[] b = toFloatArray(restored.reshape(-1).to(ScalarType.Float));
            int n = Math.min(a.length, b.length);
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < n; i++) {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            if (na < 1e-24 || nb < 1e-24) return 0.0;
            return dot / (Math.sqrt(na) * Math.sqrt(nb));
        } finally {
            try { restored.close(); } catch (Exception ignored) {}
        }
    }

    /** Signal-to-noise ratio (dB) between original and quant→dequant. */
    public static double reconstructionSnrDb(Tensor weight, BitsAndBytesConfig cfg) {
        Tensor restored = quantizeDequantize(weight, cfg);
        try {
            float[] a = toFloatArray(weight.reshape(-1).to(ScalarType.Float));
            float[] b = toFloatArray(restored.reshape(-1).to(ScalarType.Float));
            double sig = 0, noise = 0;
            int n = Math.min(a.length, b.length);
            for (int i = 0; i < n; i++) {
                sig += a[i] * a[i];
                double d = a[i] - b[i];
                noise += d * d;
            }
            if (noise < 1e-24) return Double.POSITIVE_INFINITY;
            return 10.0 * Math.log10(sig / noise);
        } finally {
            try { restored.close(); } catch (Exception ignored) {}
        }
    }

    /** Full reconstruction report — MAE, cosine similarity, SNR. */
    public static Map<String, Double> reconstructionReport(Tensor weight, BitsAndBytesConfig cfg) {
        Map<String, Double> m = new LinkedHashMap<>();
        m.put("mae", reconstructionMae(weight, cfg));
        m.put("cosine", reconstructionCosine(weight, cfg));
        m.put("snr_db", reconstructionSnrDb(weight, cfg));
        return m;
    }

    // =========================================================================
    // Collect linears from common names
    // =========================================================================

    public static Map<String, LinearImpl> linearMap(String[] names, LinearImpl[] linears) {
        Map<String, LinearImpl> m = new LinkedHashMap<>();
        if (names == null || linears == null) return m;
        int n = Math.min(names.length, linears.length);
        for (int i = 0; i < n; i++) {
            if (names[i] != null && linears[i] != null) m.put(names[i], linears[i]);
        }
        return m;
    }

    public static List<String> defaultSkipModules() {
        return List.of("lm_head", "embed_tokens", "wte", "wpe", "embed_out");
    }

    // =========================================================================
    // Internals
    // =========================================================================

    private static float[] resolveScales(QuantState state) {
        if (state.nested != null && state.doubleQuant) {
            float[] codes = toFloatArray(state.nested.qweight);
            float[] nestedScaleArr = toFloatArray(state.nested.absmax);
            float ns = nestedScaleArr.length > 0 ? nestedScaleArr[0] : state.nestedScale;
            float[] scales = new float[codes.length];
            for (int i = 0; i < codes.length; i++) scales[i] = codes[i] * ns;
            return scales;
        }
        return toFloatArray(state.absmax);
    }

    private static Tensor safeBias(LinearImpl dense) {
        try {
            Tensor bb = dense.bias();
            if (bb != null && !bb.isNull() && bb.defined()) return bb;
        } catch (Exception ignored) {
        }
        return null;
    }

    private static int nearestLevelIndex(float x, float[] levels) {
        int best = 0;
        float bestDist = Float.MAX_VALUE;
        for (int i = 0; i < levels.length; i++) {
            float d = Math.abs(x - levels[i]);
            if (d < bestDist) {
                bestDist = d;
                best = i;
            }
        }
        return best;
    }

    private static long[] shapeOf(Tensor weight) {
        long[] shape = new long[(int) weight.dim()];
        for (int i = 0; i < shape.length; i++) shape[i] = weight.size(i);
        return shape;
    }

    /** Drain a tensor to a Java float[] (allocates). */
    public static float[] toFloatArray(Tensor t) {
        requireTensor(t, "t");
        Tensor f = t.to(ScalarType.Float).contiguous().reshape(-1);
        long n = f.numel();
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Tensor too large for float[] conversion: " + n);
        }
        float[] data = new float[(int) n];
        FloatIndexer idx = f.createIndexer();
        try {
            for (long i = 0; i < n; i++) data[(int) i] = idx.get(i);
        } finally {
            idx.release();
        }
        return data;
    }

    /** Detect if a tensor contains any non-finite (NaN/Inf) elements. */
    public static boolean hasNonFinite(Tensor t) {
        return assertNoNonFinite(t, "t") > 0;
    }

    /** Move a tensor to CPU (best-effort, no-op if already CPU). */
    public static Tensor toCpu(Tensor t) {
        requireTensor(t, "t");
        try {
            if (t.device().is_cpu()) return t;
            return t.cpu();
        } catch (Throwable e) {
            return t;
        }
    }

    /** Compute squared-error (L2) between two tensors (must match shape). */
    public static double squaredError(Tensor a, Tensor b) {
        requireTensor(a, "a");
        requireTensor(b, "b");
        float[] x = toFloatArray(a);
        float[] y = toFloatArray(b);
        int n = Math.min(x.length, y.length);
        double s = 0;
        for (int i = 0; i < n; i++) {
            double d = x[i] - y[i];
            s += d * d;
        }
        return s;
    }
}