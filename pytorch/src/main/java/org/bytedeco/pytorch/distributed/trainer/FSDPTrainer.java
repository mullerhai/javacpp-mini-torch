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
package org.bytedeco.pytorch.distributed.trainer;
import org.bytedeco.pytorch.distributed.ModuleForward;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.distributed.enums.ShardingStrategy;
import org.bytedeco.pytorch.distributed.Work;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.DeviceType;
import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.cross_entropy;
import static org.bytedeco.pytorch.global.torch.empty;
import static org.bytedeco.pytorch.global.torch.zeros;
import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * Fully Sharded Data Parallel trainer (Java port of the Scala helper).
 *
 * <p>Flattens parameters, shards them across ranks, all-gathers full weights
 * for forward, then reduce-scatters gradients before the optimizer step.
 *
 * <pre>{@code
 * try (FSDPTrainer trainer = FSDPTrainer.create(model, pg)) {
 *     Tensor loss = trainer.step(input, target, optimizer);
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class FSDPTrainer implements BaseDistributedTrainer, AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";

    private final Module module;
    private final ProcessGroupWrapper processGroup;
    private final ShardingStrategy shardingStrategy;
    private final boolean reshardAfterForward;
    private final ModuleForward moduleForward;
    private final Device device;

    private final List<Tensor> shardedParams = new ArrayList<>();
    private final List<Tensor> shardedGrads = new ArrayList<>();
    private long totalParamNumel;
    private long shardSize;
    private long numForwardCalls;
    private long numBackwardCalls;
    private boolean useFullPrecision;
    private boolean closed;

    public FSDPTrainer(Module module, ProcessGroupWrapper processGroup) {
        this(module, processGroup, ShardingStrategy.FULL_SHARD, true, true);
    }

    public FSDPTrainer(
            Module module,
            ProcessGroupWrapper processGroup,
            ShardingStrategy shardingStrategy,
            boolean reshardAfterForward,
            boolean useFullPrecision) {
        this.module = Objects.requireNonNull(module, "module");
        this.processGroup = Objects.requireNonNull(processGroup, "processGroup");
        this.shardingStrategy = shardingStrategy;
        this.reshardAfterForward = reshardAfterForward;
        this.useFullPrecision = useFullPrecision;
        this.moduleForward = ModuleForward.of(module);
        this.device = processGroup.getDevice();

        if (device.type() == DeviceType.CUDA) {
            module.to(device, true);
        }
        collectParamMetadata();
        shardParameters();
        broadcastFullParameters();
        System.out.printf(
                "[FSDPTrainer] Initialized strategy=%s, shardSize=%d, rank=%d%n",
                shardingStrategy, shardSize, processGroup.getRank());
    }

    public static FSDPTrainer create(Module module, ProcessGroupWrapper pg) {
        return builder().module(module).processGroup(pg).build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private void collectParamMetadata() {
        totalParamNumel = 0;
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor t = params.get(i);
            if (t != null && !t.isNull()) {
                totalParamNumel += t.numel();
            }
        }
        int world = Math.max(1, processGroup.getWorldSize());
        shardSize = (totalParamNumel + world - 1) / world;
    }

    private void shardParameters() {
        int rank = processGroup.getRank();
        Tensor flat = null;
        Tensor shard = null;
        try {
            flat = flattenParameters();
            long start = rank * shardSize;
            long end = Math.min(start + shardSize, totalParamNumel);
            shard = flat.slice(0, new LongOptional(start), new LongOptional(end), 1);
            Tensor sharded = shard.clone().detach();
            sharded.requires_grad_(true);

            shardedParams.forEach(t -> { try { t.close(); } catch (Throwable ignored) {} });
            shardedGrads.forEach(t -> { try { t.close(); } catch (Throwable ignored) {} });
            shardedParams.clear();
            shardedParams.add(sharded);
            shardedGrads.clear();
            shardedGrads.add(zeros_like(sharded));
        } finally {
            if (shard != null) { try { shard.close(); } catch (Throwable ignored) {} }
            if (flat != null) { try { flat.close(); } catch (Throwable ignored) {} }
        }
    }

    private Tensor flattenParameters() {
        TensorVector flatList = new TensorVector();
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor t = params.get(i);
            if (t != null && !t.isNull()) {
                flatList.push_back(t.flatten());
            }
        }
        if (flatList.size() == 0) {
            return zeros(1);
        }
        return cat(flatList);
    }

    private void broadcastFullParameters() {
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor t = params.get(i);
            if (t != null && !t.isNull()) {
                processGroup.broadcast(t, 0);
            }
        }
    }

    private Tensor flattenGradients() {
        TensorVector gradList = new TensorVector();
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor t = params.get(i);
            if (t == null || t.isNull()) {
                continue;
            }
            try {
                Tensor g = t.grad();
                if (g != null && !g.isNull() && g.defined()) {
                    gradList.push_back(g.flatten());
                }
            } catch (Exception ignored) {
                // skip
            }
        }
        if (gradList.size() == 0) {
            return zeros(1);
        }
        return cat(gradList);
    }

    /**
     * All-gather local shard into full parameter buffer.
     * Returns a tensor that caller MUST close to prevent memory leaks.
     */
    private Tensor allGatherParameters() {
        int world = processGroup.getWorldSize();
        long fullSize = shardSize * world;
        Tensor full = empty(fullSize).to(device, ScalarType.Float);

        Tensor paddedInput = null;
        try {
            Tensor padded = shardedParams.get(0);
            if (padded.numel() < shardSize) {
                Tensor pad = zeros(shardSize - padded.numel()).to(device, ScalarType.Float);
                TensorVector v = new TensorVector();
                v.push_back(padded);
                v.push_back(pad);
                paddedInput = cat(v);
                pad.close();  // pad is now part of paddedInput
            } else if (padded.device().type() != device.type()) {
                paddedInput = padded.to(device, ScalarType.Float);
            } else {
                paddedInput = padded;
            }

            Work w = processGroup.allgatherBase(full, paddedInput);
            if (w != null && !w.isNull()) w._wait();

            if (full.numel() > totalParamNumel) {
                Tensor ret = full.slice(0, new LongOptional(0), new LongOptional(totalParamNumel), 1L);
                return ret;
            }
            return full;
        } finally {
            if (paddedInput != null && paddedInput != shardedParams.get(0)) {
                try { paddedInput.close(); } catch (Throwable ignored) {}
            }
        }
    }

    public Tensor forward(Tensor input) {
        Tensor inputAdj = input;
        if (input.device().type() != device.type()) {
            inputAdj = input.to(device, ScalarType.Float);
        }
        Tensor fullParams = allGatherParameters();
        try {
            writeToModule(fullParams);
            Tensor output = moduleForward.apply(module, inputAdj);
            numForwardCalls++;
            return output;
        } finally {
            if (fullParams != null) {
                try { fullParams.close(); } catch (Throwable ignored) {}
            }
            if (inputAdj != input) {
                try { inputAdj.close(); } catch (Throwable ignored) {}
            }
        }
    }

    public Tensor forward(Tensor input, Tensor target) {
        Tensor output = forward(input);
        return cross_entropy(output, target);
    }

    public Tensor step(Tensor input, Tensor target, Optimizer optimizer) {
        zeroGrad();
        Tensor output = forward(input);
        Tensor loss = cross_entropy(output, target);
        loss.backward();
        numBackwardCalls++;
        reduceScatterGradients();
        optimizer.step();
        return loss;
    }

    public Tensor trainingStep(Tensor loss, Optimizer optimizer) {
        optimizer.zero_grad();
        loss.backward();
        reduceScatterGradients();
        optimizer.step();
        return loss;
    }

    private void reduceScatterGradients() {
        Tensor gradFlat = null;
        Tensor padded = null;
        Tensor out = null;
        try {
            gradFlat = flattenGradients();
            if (gradFlat == null || gradFlat.isNull()) {
                return;
            }
            int world = processGroup.getWorldSize();
            long fullSize = shardSize * world;
            // Pad to fullSize so reduce_scatter_base can split evenly.
            if (gradFlat.numel() < fullSize) {
                Tensor pad = zeros(fullSize - gradFlat.numel()).to(device, ScalarType.Float);
                TensorVector v = new TensorVector();
                v.push_back(gradFlat);
                v.push_back(pad);
                padded = cat(v);
                pad.close();  // pad is now part of padded
            } else {
                padded = gradFlat;
            }

            out = empty(shardSize).to(device, ScalarType.Float);
            Work w = processGroup.reduceScatterBase(out, padded);
            if (w != null && !w.isNull()) w._wait();

            if (shardedGrads.isEmpty()) {
                shardedGrads.add(zeros_like(shardedParams.get(0)));
            }
            long local = shardedParams.get(0).numel();
            Tensor shard = out.numel() > local
                    ? out.slice(0, new LongOptional(0), new LongOptional(local), 1L)
                    : out;
            shardedGrads.get(0).data().copy_(shard.div(new Scalar((float) world)));
            if (shard != out) {
                try { shard.close(); } catch (Throwable ignored) {}
            }
        } finally {
            if (out != null) { try { out.close(); } catch (Throwable ignored) {} }
            if (padded != null && padded != gradFlat) {
                try { padded.close(); } catch (Throwable ignored) {}
            }
            if (gradFlat != null) { try { gradFlat.close(); } catch (Throwable ignored) {} }
        }
    }

    private void writeToModule(Tensor flatParams) {
        try (NoGradGuard guard = new NoGradGuard()) {
            long offset = 0;
            TensorVector params = module.parameters();
            for (long i = 0, n = params.size(); i < n; i++) {
                Tensor t = params.get(i);
                if (t == null || t.isNull()) {
                    continue;
                }
                long num = t.numel();
                if (offset + num <= flatParams.numel()) {
                    Tensor src = flatParams.narrow(0, offset, num);
                    t.copy_(src.view(t.sizes()));
                    try { src.close(); } catch (Throwable ignored) {}
                }
                offset += num;
            }
        }
    }

    public void zeroGrad() {
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor t = params.get(i);
            if (t == null || t.isNull()) {
                continue;
            }
            try {
                Tensor g = t.grad();
                if (g != null && !g.isNull() && g.defined()) {
                    g.zero_();
                }
            } catch (Exception ignored) {
                // skip
            }
        }
    }

    public Module getModule() { return module; }
    public List<Tensor> getShardedParameters() { return List.copyOf(shardedParams); }
    public List<Tensor> getShardedGradients() { return List.copyOf(shardedGrads); }

    public List<Tensor> parameters() {
        List<Tensor> list = new ArrayList<>();
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor t = params.get(i);
            if (t != null && !t.isNull()) {
                list.add(t);
            }
        }
        return list;
    }

    public void train() { module.train(true); }
    public void eval() { module.eval(); }
    public boolean isTraining() { return module.is_training(); }
    public void setFullPrecision(boolean useFullPrec) { this.useFullPrecision = useFullPrec; }
    public boolean isFullPrecision() { return useFullPrecision; }

    public ProcessGroupWrapper getProcessGroup() { return processGroup; }
    public ShardingStrategy getShardingStrategy() { return shardingStrategy; }
    public int getRank() { return processGroup.getRank(); }
    public int getWorldSize() { return processGroup.getWorldSize(); }
    public boolean isMainProcess() { return processGroup.isMainProcess(); }
    public Device getDevice() { return device; }
    public long getShardSize() { return shardSize; }
    public long getTotalParamSize() { return totalParamNumel; }
    public long getNumForwardCalls() { return numForwardCalls; }
    public long getNumBackwardCalls() { return numBackwardCalls; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        shardedParams.forEach(t -> { try { t.close(); } catch (Throwable ignored) {} });
        shardedParams.clear();

        shardedGrads.forEach(t -> { try { t.close(); } catch (Throwable ignored) {} });
        shardedGrads.clear();

        if (module != null) {
            try { module.close(); } catch (Throwable ignored) {}
        }
    }

    @Override
    public String toString() {
        return "FSDPTrainer{rank=" + processGroup.getRank()
                + ", worldSize=" + processGroup.getWorldSize()
                + ", strategy=" + shardingStrategy
                + ", shardSize=" + shardSize + '}';
    }

    public static final class Builder {
        private Module module;
        private ProcessGroupWrapper processGroup;
        private ShardingStrategy shardingStrategy = ShardingStrategy.FULL_SHARD;
        private boolean reshardAfterForward = true;
        private boolean useFullPrecision = true;

        public Builder module(Module m) { this.module = m; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.processGroup = pg; return this; }
        public Builder shardingStrategy(ShardingStrategy s) { this.shardingStrategy = s; return this; }
        public Builder reshardAfterForward(boolean b) { this.reshardAfterForward = b; return this; }
        public Builder useFullPrecision(boolean b) { this.useFullPrecision = b; return this; }

        public FSDPTrainer build() {
            Objects.requireNonNull(module, "module is required");
            Objects.requireNonNull(processGroup, "processGroup is required");
            return new FSDPTrainer(module, processGroup, shardingStrategy, reshardAfterForward, useFullPrecision);
        }
    }
}
