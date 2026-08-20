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
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.peft.tuners.BaseTunerLayer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java analog of HuggingFace {@code peft.tuners.oft.layer.Linear} configured for
 * Block-Diagonal Orthogonal Fine-Tuning (BOFT). Each OFT block is learned
 * independently with its own Cayley transform. The number of blocks is
 * {@code (out_features / block_size)^2} and the trainable parameters scale with
 * {@code block_size^2}.
 */
public class BoftLinear extends OftLinear {

    public BoftLinear(LinearImpl baseLayer, String name, int blockSize) {
        super(baseLayer, name, blockSize, true);
    }
}
