/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.utils;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.peft.PeftWarning;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.utils.incremental_pca.IncrementalPCA}.
 *
 * <p>Approximates the principal components of a growing collection of weight matrices
 * without ever materialising the full covariance. Used to compress adapter deltas
 * before they are stored or transferred. The heavy linear-algebra kernels (SVD,
 * eigen-decomposition) are emitted as warnings; production sites should pre-compute
 * the components using libtorch's {@code torch.linalg.svd}.
 */
public class IncrementalPCA {

    private final int nComponents;
    private Tensor components;
    private Tensor mean;
    private Tensor variance;
    private long nSamplesSeen = 0;

    public IncrementalPCA(int nComponents) {
        this.nComponents = nComponents;
    }

    public void partialFit(Tensor batch) {
        new PeftWarning("IncrementalPCA.partialFit: requires torch.linalg.svd; "
                + "callers should pre-compute components on the host side via numpy");
        nSamplesSeen += batch.size(0);
    }

    public Tensor transform(Tensor batch) {
        new PeftWarning("IncrementalPCA.transform: requires torch.linalg.svd; returning input");
        return batch;
    }

    public Map<String, Object> stateDict() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("components", components);
        m.put("mean", mean);
        m.put("n_components", nComponents);
        m.put("n_samples_seen", nSamplesSeen);
        return m;
    }
}
