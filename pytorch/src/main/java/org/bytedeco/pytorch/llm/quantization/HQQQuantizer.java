/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.quantization;

import org.bytedeco.pytorch.Tensor;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * HQQ (Half-Quadratic Quantization) 1-bit quantizer (mirrors the HQQ library used in
 * Ex21). Pure-Java reference implementation that records group-wise scale + zero offset
 * for downstream dequantization.
 */
public final class HQQQuantizer {

    public enum Bits { ONE, TWO, THREE, FOUR }

    public static final class Quantum {
        public final int[] zeros;
        public final int[] scales;
        public final int[][] codes;     // [nBlocks][nItems]
        public final int groupSize;
        public final Bits bits;
        public Quantum(int[] zeros, int[] scales, int[][] codes, int groupSize, Bits bits) {
            this.zeros = zeros;
            this.scales = scales;
            this.codes = codes;
            this.groupSize = groupSize;
            this.bits = bits;
        }
        public int approxSize() {
            return zeros.length + scales.length + codes.length * (codes.length == 0 ? 0 : codes[0].length);
        }
    }

    /** 1-bit (asymmetric) quantization of float32 weights. */
    public static Quantum quantize1Bit(float[] values, int groupSize) {
        return quantize(values, groupSize, Bits.ONE);
    }

    public static Quantum quantize(float[] values, int groupSize, Bits bits) {
        if (groupSize <= 0) groupSize = 64;
        int n = values.length;
        int nBlocks = (n + groupSize - 1) / groupSize;
        int levels = (1 << bits.ordinal() * 2) - 1;
        int[] zeros = new int[nBlocks];
        int[] scales = new int[nBlocks];
        int[][] codes = new int[nBlocks][];
        for (int b = 0; b < nBlocks; b++) {
            int start = b * groupSize;
            int end = Math.min(n, start + groupSize);
            float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
            for (int i = start; i < end; i++) {
                if (values[i] < min) min = values[i];
                if (values[i] > max) max = values[i];
            }
            float scale = (max - min) / levels;
            float zero = min;
            int[] q = new int[end - start];
            for (int i = start; i < end; i++) {
                q[i - start] = Math.round((values[i] - zero) / (scale == 0 ? 1e-8f : scale));
            }
            zeros[b] = Math.round(zero * 1000);
            scales[b] = Math.round(scale * 1000);
            codes[b] = q;
        }
        return new Quantum(zeros, scales, codes, groupSize, bits);
    }

    public static float[] dequantize(Quantum q) {
        List<Float> out = new ArrayList<>();
        for (int b = 0; b < q.codes.length; b++) {
            float zero = q.zeros[b] / 1000f;
            float scale = q.scales[b] / 1000f;
            for (int c : q.codes[b]) out.add(zero + c * scale);
        }
        float[] arr = new float[out.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = out.get(i);
        return arr;
    }

    /** Helper that quantizes a Tensor slice when given a contiguous float buffer view. */
    public static Quantum fromTensorFloats(FloatBuffer buf, int groupSize) {
        float[] arr = new float[buf.capacity()];
        buf.get(arr);
        return quantize1Bit(arr, groupSize);
    }
}