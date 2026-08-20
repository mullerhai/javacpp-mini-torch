/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option) any later version (collectively, the "License");
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
package org.bytedeco.pytorch.llm.transformers.configuration;

import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import java.util.Map;

/**
 * HuggingFace <code>mamba2Config</code>.
 * Reference: transformers/models/mamba2/configuration_mamba2.py
 */
public final class Mamba2Config extends Config {

    public static final String MODEL_TYPE = "mamba2";

    private final int numHeads;
    private final int headDim;
    private final int stateSize;
    private final int expand;
    private final int convKernel;
    private final int nGroups;
    private final boolean useBias;
    private final boolean useConvBias;
    private final boolean residualInFp32;
    private final String timeStepRank;
    private final double timeStepMin;
    private final double timeStepMax;
    private final double timeStepFloor;
    private final String timeStepLimit;
    private final boolean rescalePrenormResidual;
    private final boolean rmsNorm;
    private final int chunkSize;

    public Mamba2Config(PretrainedConfig base) {
        super(base);
        this.numHeads = toInt(base.extra().get("num_heads"), 128);
        this.headDim = toInt(base.extra().get("head_dim"), 64);
        this.stateSize = toInt(base.extra().get("state_size"), 128);
        this.expand = toInt(base.extra().get("expand"), 2);
        this.convKernel = toInt(base.extra().get("conv_kernel"), 4);
        this.nGroups = toInt(base.extra().get("n_groups"), 8);
        this.useBias = base.extra().get("use_bias") == Boolean.TRUE;
        this.useConvBias = base.extra().get("use_conv_bias") == Boolean.TRUE;
        this.residualInFp32 = base.extra().get("residual_in_fp32") == Boolean.TRUE;
        this.timeStepRank = String.valueOf(base.extra().get("time_step_rank"));
        this.timeStepMin = toDouble(base.extra().get("time_step_min"), 0.001);
        this.timeStepMax = toDouble(base.extra().get("time_step_max"), 0.1);
        this.timeStepFloor = toDouble(base.extra().get("time_step_floor"), 0.0001);
        this.timeStepLimit = String.valueOf(base.extra().get("time_step_limit"));
        this.rescalePrenormResidual = base.extra().get("rescale_prenorm_residual") == Boolean.TRUE;
        this.rmsNorm = base.extra().get("rms_norm") == Boolean.TRUE;
        this.chunkSize = toInt(base.extra().get("chunk_size"), 256);
    }

    public int numHeads() { return toInt(base().extra().get("num_heads"), 128); }
    public int headDim() { return toInt(base().extra().get("head_dim"), 64); }
    public int stateSize() { return toInt(base().extra().get("state_size"), 128); }
    public int expand() { return toInt(base().extra().get("expand"), 2); }
    public int convKernel() { return toInt(base().extra().get("conv_kernel"), 4); }
    public int nGroups() { return toInt(base().extra().get("n_groups"), 8); }
    public boolean useBias() { return base().extra().get("use_bias") == Boolean.TRUE; }
    public boolean useConvBias() { return base().extra().get("use_conv_bias") == Boolean.TRUE; }
    public boolean residualInFp32() { return base().extra().get("residual_in_fp32") == Boolean.TRUE; }
    public String timeStepRank() { Object v = base().extra().get("time_step_rank"); return v == null ? "auto" : String.valueOf(v); }
    public double timeStepMin() { return toDouble(base().extra().get("time_step_min"), 0.001); }
    public double timeStepMax() { return toDouble(base().extra().get("time_step_max"), 0.1); }
    public double timeStepFloor() { return toDouble(base().extra().get("time_step_floor"), 0.0001); }
    public String timeStepLimit() { Object v = base().extra().get("time_step_limit"); return v == null ? "(0.0, inf)" : String.valueOf(v); }
    public boolean rescalePrenormResidual() { return base().extra().get("rescale_prenorm_residual") == Boolean.TRUE; }
    public boolean rmsNorm() { return base().extra().get("rms_norm") == Boolean.TRUE; }
    public int chunkSize() { return toInt(base().extra().get("chunk_size"), 256); }

    @Override public String modelType() { return MODEL_TYPE; }
    @Override public Class<? extends Config> getClass_() { return Mamba2Config.class; }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }
}