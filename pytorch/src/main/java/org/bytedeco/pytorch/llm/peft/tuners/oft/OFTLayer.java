/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.oft;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.oft.layer.OFTLayer}.
 *
 * <p>Per-adapter holds: the OFT rotation module (Cayley transform of a block-wise
 * skew-symmetric matrix), the per-block scaling, and the optional multiplicative dropout.
 */
public abstract class OFTLayer extends BaseTunerLayer {

    /** Per-adapter rotation module producing a block-diagonal orthogonal matrix. */
    protected final Map<String, Module> oftR = new LinkedHashMap<>();
    /** Per-adapter block-wise scaling factor (typically 1.0). */
    protected final Map<String, Tensor> oftS = new LinkedHashMap<>();
    /** Per-adapter logit scaling of the input (default 1.0). */
    protected final Map<String, Tensor> oftB = new LinkedHashMap<>();
    /** Per-adapter has-consistent-trainable flag. */
    protected final Map<String, Boolean> trainable = new LinkedHashMap<>();

    public Map<String, Module> oftR() { return oftR; }
    public Map<String, Tensor> oftS() { return oftS; }
    public Map<String, Tensor> oftB() { return oftB; }
    public Map<String, Boolean> trainable() { return trainable; }

    public abstract void updateLayer(String adapterName, String initWeights, boolean layerCalledFromInit);

    @Override public List<String> adapterLayerNames() {
        return new java.util.ArrayList<>(java.util.Arrays.asList("oft_R", "oft_s", "oft_b"));
    }

    @Override public List<String> otherParamNames() {
        return new java.util.ArrayList<>(java.util.Arrays.asList("trainable"));
    }

    /** Backref interface for runtime config fields. */
    public interface OFTConfigBackref {
        int r();
        int oftBlockSize();
        boolean coft();
        double eps();
        boolean blockShare();
        boolean useCayleyNeumann();
        int numCayleyNeumannTerms();
        double moduleDropout();
    }
}