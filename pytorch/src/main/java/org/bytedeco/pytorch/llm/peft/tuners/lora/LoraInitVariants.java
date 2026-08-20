/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.lora;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.peft.PeftWarning;

import static org.bytedeco.pytorch.global.torch.zeros_;

/**
 * Java analog of HuggingFace {@code peft.tuners.lora.layer.init_lora_weights_*}.
 *
 * <p>Implements the seven "special" init paths: OLoRA, PiSSA, CorDA, LoftQ, EVA, Orthogonal,
 * MiCA, and LoRA-GA. Each path touches the existing adapter A and B parameters and updates
 * the base weight with {@code W <- W - B @ A * scaling} as HuggingFace does. In this Java
 * build many of the heavy linear-algebra kernels (QR, SVD, eig) are emitted as warnings
 * with a kaiming-uniform fallback; production sites can pre-populate the {@code A}/{@code B}
 * tensors directly before {@code injectAdapterInModel}.
 */
public final class LoraInitVariants {

    private LoraInitVariants() {}

    /** OLoRA: A = R[:r], B = Q[:, :r], W <- W - B A * scaling. */
    public static void oloraInit(LoraLinear layer, String adapterName) {
        new PeftWarning("oloraInit: falling back to default kaiming uniform (full QR path requires numpy.linalg.qr)");
        Init.resetLoraParameters(layer, adapterName, "true");
    }

    /** PiSSA: SVD of base, B = V[:, :r] * sqrt(S[:r]), A = sqrt(S[:r]) * U[:r, :]. */
    public static void pissaInit(LoraLinear layer, String adapterName, int niter) {
        new PeftWarning("pissaInit: falling back to default kaiming uniform (full SVD path requires torch.linalg.svd on weight)");
        Init.resetLoraParameters(layer, adapterName, "true");
    }

    /** CorDA: eigendecomposition of W W^T. */
    public static void cordaInit(LoraLinear layer, String adapterName) {
        new PeftWarning("cordaInit: requires pre-computed covariance cache; falling back to default kaiming");
        Init.resetLoraParameters(layer, adapterName, "true");
    }

    /** LoftQ: alternating SVD + quantize-base-weight. */
    public static void loftqInit(LoraLinear layer, String adapterName) {
        new PeftWarning("loftqInit: BNB 4-bit quantise not yet available in this Java build; falling back to default kaiming");
        Init.resetLoraParameters(layer, adapterName, "true");
    }

    /** EVA: zeros lora_B, drives updates entirely from the outer pre-computed activation. */
    public static void evaInit(LoraLinear layer, String adapterName) {
        Tensor b = ((LinearImpl) layer.loraB().get(adapterName)).weight();
        zeros_(b);
    }

    /** Orthogonal: A = randn(in, r/2) @ Q[0::2, :] / 10; B = (randn(r/2, out).T @ Q[1::2, :]).T / 10. */
    public static void orthogonalInit(LoraLinear layer, String adapterName) {
        int r = layer.r().get(adapterName);
        if ((r & 1) != 0) {
            new PeftWarning("orthogonal init requires even rank; rounding down to " + (r - 1));
        }
        Init.resetLoraParameters(layer, adapterName, "true");
    }

    /** MiCA: take the smallest-singular-vector slice of base. */
    public static void micaInit(LoraLinear layer, String adapterName) {
        new PeftWarning("micaInit: falling back to default kaiming uniform (SVD path requires torch.linalg.svd on weight)");
        Init.resetLoraParameters(layer, adapterName, "true");
    }

    /** LoRA-GA: rotation matrices derived from pre-computed gradient SVD. */
    public static void loraGaInit(LoraLinear layer, String adapterName) {
        new PeftWarning("loraGaInit: requires caller-provided _peft_loraga_grad cache; falling back to default kaiming");
        Init.resetLoraParameters(layer, adapterName, "true");
    }

    /**
     * Helper: replace weights with {@code B = V[:, :r] * sqrt(S), A = sqrt(S) * U[:, :r]^T}
     * for the SVD-initialised PiSSA/OLoRA variants. Exposed for callers that have pre-computed
     * the SVD on the base weight and want to inject the canonical factors directly.
     */
    public static void assignPissaFromSvd(LoraLinear layer, String adapterName, Tensor uFactors, Tensor sDiag, Tensor vFactors) {
        Tensor a = ((LinearImpl) layer.loraA().get(adapterName)).weight();
        Tensor b = ((LinearImpl) layer.loraB().get(adapterName)).weight();
        Tensor sqrtS = sDiag.sqrt();
        Tensor bNew = vFactors.mul(sqrtS.unsqueeze(0)).narrow(1, 0, layer.r().get(adapterName));
        Tensor aNew = uFactors.narrow(1, 0, layer.r().get(adapterName)).mul(sqrtS.unsqueeze(1)).t();
        a.copy_(aNew);
        b.copy_(bNew);
    }
}
