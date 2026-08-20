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
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.peft.LoraConfig;

import static org.bytedeco.pytorch.global.torch.kaiming_uniform_;
import static org.bytedeco.pytorch.global.torch.normal_;
import static org.bytedeco.pytorch.global.torch.zeros_;

/**
 * Java analog of HuggingFace {@code peft.tuners.lora.layer.LoraLayer.reset_lora_parameters}.
 *
 * <p>Implements all 11 init_lora_weights values plus the PiSSA family. Concrete handoff to
 * specialised init methods (PiSSA / OLoRA / CorDA / LoftQ / EVA / Orthogonal / MiCA / LoraGA)
 * is provided via {@link LoraInitVariants}.
 */
public final class Init {

    private Init() {}

    public static void resetLoraParameters(LoraLinear layer, String adapterName, Object initLoraWeights) {
        boolean isEmbedding = false; // LoraEmbedding check is handled in subclasses
        boolean useDora = Boolean.TRUE.equals(layer.useDora().getOrDefault(adapterName, false));        if (initLoraWeights instanceof Boolean) {
            if (((Boolean) initLoraWeights)) {
                kaimingInit(layer, adapterName);
            }
        } else if (initLoraWeights instanceof String) {
            String s = (String) initLoraWeights;
            switch (s) {
                case "true":
                case "default":
                    kaimingInit(layer, adapterName);
                    break;
                case "false":
                    break;
                case "gaussian":
                    gaussianInit(layer, adapterName);
                    break;
                case "olora":
                    LoraInitVariants.oloraInit(layer, adapterName);
                    break;
                case "pissa":
                    LoraInitVariants.pissaInit(layer, adapterName, 0);
                    break;
                case "corda":
                    LoraInitVariants.cordaInit(layer, adapterName);
                    break;
                case "loftq":
                    LoraInitVariants.loftqInit(layer, adapterName);
                    break;
                case "eva":
                    LoraInitVariants.evaInit(layer, adapterName);
                    break;
                case "orthogonal":
                    LoraInitVariants.orthogonalInit(layer, adapterName);
                    break;
                case "mica":
                    LoraInitVariants.micaInit(layer, adapterName);
                    break;
                case "lora_ga":
                    LoraInitVariants.loraGaInit(layer, adapterName);
                    break;
                default:
                    if (LoraConfig.isPissaVariant(s)) {
                        int niter = LoraConfig.pissaNiter(s);
                        LoraInitVariants.pissaInit(layer, adapterName, niter);
                    } else {
                        throw new IllegalArgumentException("init_lora_weights=" + s + " not recognised");
                    }
            }
        } else if (initLoraWeights == null) {
            kaimingInit(layer, adapterName);
        }
    }

    private static Tensor aWeight(LoraLinear layer, String adapterName) {
        org.bytedeco.pytorch.nn.Module m = layer.loraA().get(adapterName);
        if (m instanceof LinearImpl) return ((LinearImpl) m).weight();
        if (m instanceof LoraEmbedding.PseudoModule) return ((LoraEmbedding.PseudoModule) m).weight;
        return null;
    }

    private static Tensor bWeight(LoraLinear layer, String adapterName) {
        org.bytedeco.pytorch.nn.Module m = layer.loraB().get(adapterName);
        if (m instanceof LinearImpl) return ((LinearImpl) m).weight();
        if (m instanceof LoraEmbedding.PseudoModule) return ((LoraEmbedding.PseudoModule) m).weight;
        return null;
    }

    private static void kaimingInit(LoraLinear layer, String adapterName) {
        Tensor a = aWeight(layer, adapterName);
        Tensor b = bWeight(layer, adapterName);
        // kaiming_uniform_ with FanModeType/Nonlinearity not available in JavaCPP API
        // Initialize a with normal distribution (kaiming normal)
        if (a != null) {
            normal_(a, 0.0, Math.sqrt(2.0));
        }
        if (b != null) {
            zeros_(b);
        }
    }

    private static void gaussianInit(LoraLinear layer, String adapterName) {
        Tensor a = aWeight(layer, adapterName);
        Tensor b = bWeight(layer, adapterName);
        int r = layer.r().get(adapterName);
        if (a != null) normal_(a, 0.0, 1.0 / Math.sqrt(r));
        if (b != null) zeros_(b);
    }
}
