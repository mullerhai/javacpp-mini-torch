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
package org.bytedeco.pytorch.llm.modules;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.concurrent.atomic.AtomicReference;

/**
 * Long Context RoPE (Rotary Position Embedding) implementations.
 *
 * <p>Supports:
 * <ul>
 *   <li>YaRN RoPE (Yet another RoPE extensioN)</li>
 *   <li>LongRoPE</li>
 *   <li>NTK-aware scaling</li>
 *   <li>Dynamic scaling factors</li>
 * </ul>
 *
 * <p>Reference: YaRN, LongRoPE, NTK-aware scaling
 *
 * <pre>{@code
 * LongContextRoPE rope = LongContextRoPE.builder()
 *     .originalContextLength(4096)
 *     .extendedContextLength(128000)
 *     .method(LongContextRoPE.Method.YARN)
 *     .betaFast(32)
 *     .betaSlow(1)
 *     .build();
 *
 * Tensor extended = rope.applyRotary(queries, keys, positionIds);
 * }</pre>
 */
public class LongContextRoPE {

    public static final String VERSION = "2.0";

    // Configuration
    private final int originalContextLength;
    private final int extendedContextLength;
    private final Method method;
    private final float base;
    private final int dimensions;
    private final float betaFast;
    private final float betaSlow;
    private final boolean useDynamicScaling;
    private final boolean useNtkScaling;

    // Cached values
    private final float[] freqs;  // Precomputed frequency array

    /**
     * RoPE extension methods.
     */
    public enum Method {
        /** YaRN: Yet another RoPE extensioN */
        YARN,
        /** LongRoPE: Extended context window */
        LONG_ROPE,
        /** NTK-aware scaling */
        NTK,
        /** Linear interpolation (naive) */
        LINEAR,
        /** Dynamic scaling */
        DYNAMIC,
        /** No extension (standard RoPE) */
        STANDARD
    }

    public static Builder builder() {
        return new Builder();
    }

    private LongContextRoPE(Builder builder) {
        this.originalContextLength = builder.originalContextLength;
        this.extendedContextLength = builder.extendedContextLength;
        this.method = builder.method;
        this.base = builder.base;
        this.dimensions = builder.dimensions;
        this.betaFast = builder.betaFast;
        this.betaSlow = builder.betaSlow;
        this.useDynamicScaling = builder.useDynamicScaling;
        this.useNtkScaling = builder.useNtkScaling;

        // Precompute frequencies
        this.freqs = precomputeFreqs();
    }

    /**
     * Precompute frequency array for all positions up to extended length.
     */
    private float[] precomputeFreqs() {
        float[] freqs = new float[extendedContextLength * dimensions];

        for (int pos = 0; pos < extendedContextLength; pos++) {
            for (int i = 0; i < dimensions; i++) {
                float freq = (float) Math.pow(base, -2.0 * i / dimensions);
                freqs[pos * dimensions + i] = freq;
            }
        }

        return freqs;
    }

    /**
     * Apply rotary position embedding to queries and keys.
     *
     * @param queries   Query tensor [..., seq_len, head_dim]
     * @param keys     Key tensor [..., seq_len, head_dim]
     * @param positionIds Position IDs tensor [seq_len]
     * @return Rotated queries and keys as array [0]=queries, [1]=keys
     */
    public Tensor[] applyRotary(Tensor queries, Tensor keys, Tensor positionIds) {
        switch (method) {
            case YARN:
                return applyYarn(queries, keys, positionIds);
            case NTK:
                return applyNtkScaling(queries, keys, positionIds);
            case DYNAMIC:
                return applyDynamicScaling(queries, keys, positionIds);
            case LONG_ROPE:
                return applyLongRoPE(queries, keys, positionIds);
            case LINEAR:
                return applyLinearInterpolation(queries, keys, positionIds);
            case STANDARD:
            default:
                return applyStandard(queries, keys, positionIds);
        }
    }

    /**
     * Apply standard RoPE (no extension).
     */
    private Tensor[] applyStandard(Tensor queries, Tensor keys, Tensor positionIds) {
        // Standard RoPE without any extension
        Tensor cos = computeCos(queries.dim() - 2, positionIds);
        Tensor sin = computeSin(queries.dim() - 2, positionIds);

        Tensor qRot = rotateHalf(queries);
        Tensor kRot = rotateHalf(keys);

        Tensor qOut = queries.mul(cos).add(qRot.mul(sin));
        Tensor kOut = keys.mul(cos).add(kRot.mul(sin));

        return new Tensor[]{qOut, kOut};
    }

    /**
     * Apply YaRN RoPE.
     */
    private Tensor[] applyYarn(Tensor queries, Tensor keys, Tensor positionIds) {
        int seqLen = (int) positionIds.size(0);

        // 1. Compute scaling factor
        float scaleFactor = computeYarnScale(seqLen);

        // 2. Scale positions
        float[] scaledPositions = new float[seqLen];
        for (int i = 0; i < seqLen; i++) {
            float pos = positionIds.get(i);
            float scaled = pos * scaleFactor;
            // Apply affine transformation
            float t = Math.min(1.0f, pos / betaFast) * Math.max(0.0f, Math.min(1.0f, (pos - originalContextLength) / (extendedContextLength - originalContextLength)));
            scaledPositions[i] = scaled * (1 + t * (betaSlow / betaFast - 1));
        }

        // 3. Create scaled position tensor
        Tensor scaledPos = torch.tensor(scaledPositions);

        // 4. Apply RoPE with scaled positions
        Tensor cos = computeCos(queries.dim() - 2, scaledPos);
        Tensor sin = computeSin(queries.dim() - 2, scaledPos);

        Tensor qRot = rotateHalf(queries);
        Tensor kRot = rotateHalf(keys);

        Tensor qOut = queries.mul(cos).add(qRot.mul(sin));
        Tensor kOut = keys.mul(cos).add(kRot.mul(sin));

        return new Tensor[]{qOut, kOut};
    }

    /**
     * Apply NTK-aware scaling.
     */
    private Tensor[] applyNtkScaling(Tensor queries, Tensor keys, Tensor positionIds) {
        int seqLen = (int) positionIds.size(0);

        // 1. Compute NTK scaling factor
        float scale = (float) Math.pow(originalContextLength / (float) Math.max(seqLen, originalContextLength), 4.0 / dimensions);

        // 2. Scale base frequency
        float scaledBase = base * scale;

        // 3. Compute frequencies with scaled base
        Tensor cos = computeCosNtk(queries.dim() - 2, positionIds, scaledBase);
        Tensor sin = computeSinNtk(queries.dim() - 2, positionIds, scaledBase);

        Tensor qRot = rotateHalf(queries);
        Tensor kRot = rotateHalf(keys);

        Tensor qOut = queries.mul(cos).add(qRot.mul(sin));
        Tensor kOut = keys.mul(cos).add(kRot.mul(sin));

        return new Tensor[]{qOut, kOut};
    }

    /**
     * Apply dynamic scaling.
     */
    private Tensor[] applyDynamicScaling(Tensor queries, Tensor keys, Tensor positionIds) {
        int seqLen = (int) positionIds.size(0);

        // Compute dynamic scale based on sequence length
        float scale;
        if (seqLen <= originalContextLength) {
            scale = 1.0f;
        } else {
            // Exponential decay
            float alpha = (float) Math.log(seqLen / originalContextLength) / Math.log(extendedContextLength / originalContextLength);
            scale = (float) Math.pow(extendedContextLength / originalContextLength, -alpha);
        }

        // Apply with dynamic scaling
        float scaledBase = base * scale;
        Tensor cos = computeCosNtk(queries.dim() - 2, positionIds, scaledBase);
        Tensor sin = computeSinNtk(queries.dim() - 2, positionIds, scaledBase);

        Tensor qRot = rotateHalf(queries);
        Tensor kRot = rotateHalf(keys);

        Tensor qOut = queries.mul(cos).add(qRot.mul(sin));
        Tensor kOut = keys.mul(cos).add(kRot.mul(sin));

        return new Tensor[]{qOut, kOut};
    }

    /**
     * Apply LongRoPE method.
     */
    private Tensor[] applyLongRoPE(Tensor queries, Tensor keys, Tensor positionIds) {
        int seqLen = (int) positionIds.size(0);

        // 1. Interpolate positions for the original context
        // 2. Extend positions for the new context

        float ratio = (float) extendedContextLength / originalContextLength;

        float[] interpolatedPositions = new float[seqLen];
        for (int i = 0; i < seqLen; i++) {
            float pos = positionIds.get(i);
            if (pos < originalContextLength) {
                // Linear interpolation
                interpolatedPositions[i] = pos * ratio;
            } else {
                // Non-linear extension
                float extra = pos - originalContextLength;
                float scale = (float) Math.pow(ratio, 1 - extra / (extendedContextLength - originalContextLength));
                interpolatedPositions[i] = originalContextLength * ratio + extra * scale;
            }
        }

        Tensor interpPos = torch.tensor(interpolatedPositions);
        return applyStandard(queries, keys, interpPos);
    }

    /**
     * Apply linear interpolation.
     */
    private Tensor[] applyLinearInterpolation(Tensor queries, Tensor keys, Tensor positionIds) {
        float ratio = (float) originalContextLength / extendedContextLength;

        float[] scaledPositions = new float[(int) positionIds.size(0)];
        for (int i = 0; i < scaledPositions.length; i++) {
            scaledPositions[i] = positionIds.get(i) * ratio;
        }

        Tensor scaledPos = torch.tensor(scaledPositions);
        return applyStandard(queries, keys, scaledPos);
    }

    /**
     * Compute YaRN scaling factor.
     */
    private float computeYarnScale(int seqLen) {
        if (seqLen <= originalContextLength) {
            return 1.0f;
        }

        // Scale factor for extended context
        float scale = (float) Math.pow(originalContextLength / (float) Math.max(seqLen, originalContextLength), 2.0 / dimensions);
        return scale;
    }

    /**
     * Compute cosine values.
     */
    private Tensor computeCos(int dim, Tensor positions) {
        int seqLen = (int) positions.size(0);
        float[] cosValues = new float[seqLen * dimensions];

        for (int pos = 0; pos < seqLen; pos++) {
            for (int i = 0; i < dimensions; i += 2) {
                float angle = positions.get(pos) * freqs[pos * dimensions + i];
                cosValues[pos * dimensions + i] = (float) Math.cos(angle);
                if (i + 1 < dimensions) {
                    cosValues[pos * dimensions + i + 1] = (float) Math.cos(angle);
                }
            }
        }

        Tensor cos = torch.tensor(cosValues).reshape(1, seqLen, dimensions);
        return cos;
    }

    /**
     * Compute sine values.
     */
    private Tensor computeSin(int dim, Tensor positions) {
        int seqLen = (int) positions.size(0);
        float[] sinValues = new float[seqLen * dimensions];

        for (int pos = 0; pos < seqLen; pos++) {
            for (int i = 0; i < dimensions; i += 2) {
                float angle = positions.get(pos) * freqs[pos * dimensions + i];
                sinValues[pos * dimensions + i] = (float) Math.sin(angle);
                if (i + 1 < dimensions) {
                    sinValues[pos * dimensions + i + 1] = (float) Math.sin(angle);
                }
            }
        }

        Tensor sin = torch.tensor(sinValues).reshape(1, seqLen, dimensions);
        return sin;
    }

    /**
     * Compute cosine with custom base.
     */
    private Tensor computeCosNtk(int dim, Tensor positions, float customBase) {
        int seqLen = (int) positions.size(0);
        float[] cosValues = new float[seqLen * dimensions];

        for (int pos = 0; pos < seqLen; pos++) {
            for (int i = 0; i < dimensions; i += 2) {
                float freq = (float) Math.pow(customBase, -2.0 * i / dimensions);
                float angle = positions.get(pos) * freq;
                cosValues[pos * dimensions + i] = (float) Math.cos(angle);
                if (i + 1 < dimensions) {
                    cosValues[pos * dimensions + i + 1] = (float) Math.cos(angle);
                }
            }
        }

        Tensor cos = torch.tensor(cosValues).reshape(1, seqLen, dimensions);
        return cos;
    }

    /**
     * Compute sine with custom base.
     */
    private Tensor computeSinNtk(int dim, Tensor positions, float customBase) {
        int seqLen = (int) positions.size(0);
        float[] sinValues = new float[seqLen * dimensions];

        for (int pos = 0; pos < seqLen; pos++) {
            for (int i = 0; i < dimensions; i += 2) {
                float freq = (float) Math.pow(customBase, -2.0 * i / dimensions);
                float angle = positions.get(pos) * freq;
                sinValues[pos * dimensions + i] = (float) Math.sin(angle);
                if (i + 1 < dimensions) {
                    sinValues[pos * dimensions + i + 1] = (float) Math.sin(angle);
                }
            }
        }

        Tensor sin = torch.tensor(sinValues).reshape(1, seqLen, dimensions);
        return sin;
    }

    /**
     * Rotate half of the tensor.
     */
    private Tensor rotateHalf(Tensor x) {
        // x[..., : dim/2] -> -x[..., dim/2:]
        // x[..., dim/2:] -> x[..., :dim/2]
        int headDim = dimensions;
        Tensor x1 = x.slice(x.dim() - 1, 0, headDim / 2);
        Tensor x2 = x.slice(x.dim() - 1, headDim / 2, headDim);
        return torch.cat(new Tensor[]{
                x2.neg(),
                x1
        }, x.dim() - 1);
    }

    /**
     * Get method being used.
     */
    public Method getMethod() {
        return method;
    }

    /**
     * Get original context length.
     */
    public int getOriginalContextLength() {
        return originalContextLength;
    }

    /**
     * Get extended context length.
     */
    public int getExtendedContextLength() {
        return extendedContextLength;
    }

    @Override
    public String toString() {
        return String.format(
                "LongContextRoPE{method=%s, original=%d, extended=%d, dimensions=%d}",
                method, originalContextLength, extendedContextLength, dimensions);
    }

    /**
     * Builder.
     */
    public static class Builder {
        private int originalContextLength = 4096;
        private int extendedContextLength = 128000;
        private Method method = Method.YARN;
        private float base = 10000.0f;
        private int dimensions = 128;
        private float betaFast = 32;
        private float betaSlow = 1;
        private boolean useDynamicScaling = false;
        private boolean useNtkScaling = false;

        public Builder originalContextLength(int v) { this.originalContextLength = v; return this; }
        public Builder extendedContextLength(int v) { this.extendedContextLength = v; return this; }
        public Builder method(Method v) { this.method = v; return this; }
        public Builder base(float v) { this.base = v; return this; }
        public Builder dimensions(int v) { this.dimensions = v; return this; }
        public Builder betaFast(float v) { this.betaFast = v; return this; }
        public Builder betaSlow(float v) { this.betaSlow = v; return this; }
        public Builder useDynamicScaling(boolean v) { this.useDynamicScaling = v; return this; }
        public Builder useNtkScaling(boolean v) { this.useNtkScaling = v; return this; }

        /**
         * Configure for LLaMA models.
         */
        public Builder llama() {
            this.base = 10000.0f;
            this.dimensions = 128;
            return this;
        }

        /**
         * Configure for Qwen models.
         */
        public Builder qwen() {
            this.base = 1000000.0f;
            this.dimensions = 128;
            return this;
        }

        /**
         * Configure for Mistral models.
         */
        public Builder mistral() {
            this.base = 10000.0f;
            this.dimensions = 128;
            this.betaFast = 512;
            return this;
        }

        public LongContextRoPE build() {
            return new LongContextRoPE(this);
        }
    }
}
