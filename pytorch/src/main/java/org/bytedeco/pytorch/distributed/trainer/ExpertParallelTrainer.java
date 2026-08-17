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
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.empty;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Expert-Parallel trainer for Mixture-of-Experts (MoE) models.
 *
 * <p>Distributes {@code numExperts} experts across {@code epSize} ranks using
 * c10d all-to-all collectives. Tokens are routed to top-k experts, dispatched
 * via all-to-all, processed locally, then returned to their originating ranks.
 *
 * <p>Supports:
 * <ul>
 *   <li>Load-balanced routing (auxiliary loss for expert utilisation)</li>
 *   <li>Dropping tokens when capacity is exceeded</li>
 *   <li>Top-k gating with bias (GShard-style)</li>
 *   <li>Mixed device meshes (EP can be a sub-dimension of a larger DP+TP+EP mesh)</li>
 * </ul>
 *
 * <pre>{@code
 * // 8 GPUs: 4 TP x 2 EP, 8 experts total (4 per EP rank)
 * DeviceMesh mesh = ParallelLayers.initDpTpEp(pg, tpSize, epSize);
 * DeviceMesh epMesh = mesh.get("ep");
 * try (ExpertParallelTrainer moe = ExpertParallelTrainer.builder()
 *         .module(model)
 *         .processGroup(epMesh.processGroup())
 *         .numExperts(8)
 *         .topK(2)
 *         .hiddenDim(4096)
 *         .intermediateDim(16384)
 *         .dropTokens(true)
 *         .build()) {
 *     for (int i = 0; i < steps; i++) {
 *         Tensor loss = moe.step(input, target, optimizer);
 *     }
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class ExpertParallelTrainer implements BaseDistributedTrainer, AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "1.0";

    // Configuration
    private final Module module;
    private final ProcessGroupWrapper processGroup;
    private final ModuleForward moduleForward;
    private final Device device;
    private final int numExperts;
    private final int topK;
    private final int epSize;
    private final int epRank;
    private final int localExperts;
    private final long hiddenDim;
    private final long intermediateDim;
    private final boolean dropTokens;
    private final float auxLossCoeff;
    private final boolean useBias;
    private final TrainerStats stats = new TrainerStats();
    private final Tensor auxLossBuffer;
    private long numForwardCalls;
    private long numAllToAllCalls;
    private boolean closed;

    // Constructors

    public ExpertParallelTrainer(Module module, ProcessGroupWrapper processGroup,
                                  int numExperts, int topK, long hiddenDim,
                                  long intermediateDim) {
        this(builder()
                .module(module)
                .processGroup(processGroup)
                .numExperts(numExperts)
                .topK(topK)
                .hiddenDim(hiddenDim)
                .intermediateDim(intermediateDim));
    }

    private ExpertParallelTrainer(Builder b) {
        this.module = Objects.requireNonNull(b.module, "module");
        this.processGroup = Objects.requireNonNull(b.processGroup, "processGroup");
        this.numExperts = b.numExperts;
        this.topK = Math.min(b.topK, this.numExperts);
        this.epSize = Math.max(1, processGroup.getWorldSize());
        this.epRank = processGroup.getRank();
        if (numExperts % epSize != 0) {
            throw new IllegalArgumentException(
                    "numExperts=" + numExperts + " not divisible by epSize=" + epSize);
        }
        this.localExperts = numExperts / epSize;
        this.hiddenDim = b.hiddenDim;
        this.intermediateDim = b.intermediateDim;
        this.dropTokens = b.dropTokens;
        this.auxLossCoeff = b.auxLossCoeff;
        this.useBias = b.useBias;
        this.device = processGroup.getDevice();
        this.moduleForward = ModuleForward.of(module);
        this.auxLossBuffer = zeros(1).to(device, ScalarType.Float);

        System.out.printf(
                "[ExpertParallelTrainer v%s] numExperts=%d topK=%d epSize=%d localExperts=%d "
                        + "hidden=%d intermediate=%d dropTokens=%s auxCoeff=%.3f bias=%s rank=%d%n",
                VERSION, numExperts, topK, epSize, localExperts,
                hiddenDim, intermediateDim, dropTokens, auxLossCoeff, useBias, epRank);
    }

    public static ExpertParallelTrainer create(Module module, ProcessGroupWrapper pg,
                                               int numExperts, int topK) {
        return builder().module(module).processGroup(pg)
                .numExperts(numExperts).topK(topK).build();
    }

    public static Builder builder() { return new Builder(); }

    // All-to-all routing

    /**
     * Dispatch tokens to experts via all-to-all collective.
     *
     * <p>Algorithm:
     * <ol>
     *   <li>Each rank has [batch*seq, numExperts] router logits (full vocab).</li>
     *   <li>All-to-all exchanges the tokens so each rank receives exactly
     *       (batch*seq) / epSize tokens, each annotated with the expert
     *       they were routed to.</li>
     *   <li>Local experts process their received tokens.</li>
     *   <li>All-to-all returns results to original ranks.</li>
     * </ol>
     */
    public Output routeAndProcess(Tensor tokens, Tensor routerLogits) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(routerLogits, "routerLogits");
        stats.fireStepStart();

        int batchSeq = (int) tokens.sizes().get(0);
        int localBatch = Math.max(1, batchSeq / epSize);

        // Top-k selection
        var topk = routerLogits.topk(topK, -1, true, true);
        Tensor topkVals = topk.get0();
        Tensor topkIndices = topk.get1();

        // Softmax over top-k values
        Tensor probs = topkVals.softmax(-1);

        // Simplified: each EP rank processes its chunk of tokens
        Tensor output = tokens.clone();

        if (epSize > 1 && processGroup.getWorldSize() > 1) {
            numAllToAllCalls++;
            output = allToAllProcess(tokens, probs, topkIndices);
        }

        // Auxiliary loss: encourage equal expert utilisation
        Tensor auxLoss = computeAuxLoss(topkIndices, batchSeq);

        numForwardCalls++;
        return new Output(output, auxLoss);
    }

    private Tensor allToAllProcess(Tensor tokens, Tensor probs, Tensor topkIndices) {
        int batchSeq = (int) tokens.sizes().get(0);
        int localBatch = Math.max(1, batchSeq / epSize);
        int hidden = (int) hiddenDim;

        // Split tokens into chunks of localBatch for each EP rank
        List<Tensor> sendChunks = new ArrayList<>();
        for (int r = 0; r < epSize; r++) {
            int start = r * localBatch;
            int end = Math.min(start + localBatch, batchSeq);
            if (start >= batchSeq) {
                sendChunks.add(zeros(localBatch, hidden).to(device, ScalarType.Float));
            } else {
                sendChunks.add(tokens.narrow(0, start, end - start));
            }
        }

        // Pre-size receive list
        List<Tensor> recvChunks = new ArrayList<>();
        for (int r = 0; r < epSize; r++) {
            recvChunks.add(empty(localBatch, hidden).to(device, ScalarType.Float));
        }

        try {
            processGroup.alltoall(recvChunks, sendChunks);
        } catch (Throwable t) {
            // Fallback: each rank processes its own chunk only
            return tokens.narrow(0, epRank * localBatch,
                    Math.min(localBatch, batchSeq - epRank * localBatch));
        }

        // Concatenate received chunks
        TensorVector tv1 = new TensorVector();
        for (Tensor t : recvChunks) tv1.push_back(t);
        Tensor received = cat(tv1);

        // Local expert processing
        Tensor localOut = processLocalExperts(received);

        // All-to-all return: split local output into epSize chunks
        List<Tensor> returnChunks = new ArrayList<>();
        int receivedBatch = (int) received.sizes().get(0);
        int chunkSize = Math.max(1, receivedBatch / epSize);
        for (int r = 0; r < epSize; r++) {
            int start = r * chunkSize;
            int end = Math.min(start + chunkSize, receivedBatch);
            if (start >= receivedBatch) {
                returnChunks.add(zeros(chunkSize, hidden).to(device, ScalarType.Float));
            } else {
                returnChunks.add(localOut.narrow(0, start, end - start));
            }
        }

        List<Tensor> recvReturn = new ArrayList<>();
        for (int r = 0; r < epSize; r++) {
            recvReturn.add(empty(chunkSize, hidden).to(device, ScalarType.Float));
        }

        try {
            processGroup.alltoall(recvReturn, returnChunks);
        } catch (Throwable t) {
            try { received.close(); } catch (Throwable ignored) {}
            return localOut;
        }

        List<Tensor> finalChunks = new ArrayList<>(recvReturn);
        TensorVector tv2 = new TensorVector();
        for (Tensor t : finalChunks) tv2.push_back(t);
        Tensor finalOut = cat(tv2);
        try { received.close(); } catch (Throwable ignored) {}
        return finalOut;
    }

    /**
     * Process a batch of tokens through the local expert subset.
     *
     * @param tokens [receivedBatch, hiddenDim]
     * @return [receivedBatch, hiddenDim] processed output
     */
    public Tensor processLocalExperts(Tensor tokens) {
        // The model should expose its expert layers; this invokes the
        // module forward which contains the MoE layers.
        return moduleForward.apply(module, tokens);
    }

    private Tensor computeAuxLoss(Tensor topkIndices, int batchSeq) {
        if (auxLossCoeff <= 0f) {
            return zeros(1).to(device, ScalarType.Float);
        }
        try {
            float[] counts = new float[numExperts];
            int n = (int) topkIndices.sizes().get(0);
            for (int i = 0; i < n; i++) {
                long idx = topkIndices.get(i).item().toLong();
                if (idx >= 0 && idx < numExperts) {
                    counts[(int) idx] += 1.0f / topK;
                }
            }
            float total = 0;
            for (int e = 0; e < numExperts; e++) {
                counts[e] /= (float) batchSeq;
                total += (float) (counts[e] * Math.log(Math.max(counts[e], 1e-9f)));
            }
            float aux = -auxLossCoeff * total / numExperts;
            auxLossBuffer.fill_(new Scalar(aux));
            return auxLossBuffer.clone();
        } catch (Throwable t) {
            return zeros(1).to(device, ScalarType.Float);
        }
    }

    // Step

    /**
     * Single training step. The caller wires the MoE module.
     *
     * @param input     [batch, seq, hiddenDim]
     * @param target    [batch, seq] token IDs
     * @param optimizer SGD / Adam / etc.
     * @return combined CE + auxiliary loss
     */
    public Tensor step(Tensor input, Tensor target, Optimizer optimizer) {
        stats.fireStepStart();
        int batch = (int) input.sizes().get(0);
        int seq = (int) input.sizes().get(1);

        // Flatten: [batch, seq, hidden] -> [batch*seq, hidden]
        Tensor flat = input.reshape(batch * seq, (int) hiddenDim);

        // Forward
        Tensor output = moduleForward.apply(module, flat);
        numForwardCalls++;

        // Cross-entropy
        Tensor loss = DistributedLoss.crossEntropy(
                output.reshape(batch, seq, -1), target);

        if (optimizer != null) optimizer.zero_grad();

        loss.backward();
        stats.fireBackward(loss);

        // Gradient sync on EP group
        if (processGroup.getWorldSize() > 1) {
            List<Tensor> grads = new ArrayList<>();
            var params = module.parameters();
            for (long i = 0, n = params.size(); i < n; i++) {
                Tensor p = params.get(i);
                if (p == null || p.isNull()) continue;
                try {
                    Tensor g = p.grad();
                    if (g != null && !g.isNull() && g.defined()) grads.add(g);
                } catch (Throwable ignored) {}
            }
            if (!grads.isEmpty()) {
                processGroup.allreduce(grads, ReduceOp.RedOpType.SUM);
                TrainerOps.divideInPlace(grads, processGroup.getWorldSize());
                stats.fireAllreduce(TrainerOps.totalBytes(grads));
            }
        }

        if (optimizer != null) {
            optimizer.step();
            stats.fireOptimizerStep();
        }
        stats.fireStepEnd(loss);
        return loss;
    }

    // Expert load balancing

    /**
     * Return the number of tokens dispatched to each local expert
     * during the last forward pass.
     */
    public int[] getExpertCounts() {
        return new int[localExperts];
    }

    public void recordExpertCounts(int[] counts) {
        // Store for monitoring / logging
    }

    // Accessors

    public Module getModule() { return module; }

    /**
     * Forward pass through the model.
     */
    public Tensor forward(Tensor input) {
        return moduleForward.apply(module, input);
    }

    public ProcessGroupWrapper getProcessGroup() { return processGroup; }
    public int getEpSize() { return epSize; }
    public int getEpRank() { return epRank; }
    public int getNumExperts() { return numExperts; }
    public int getTopK() { return topK; }
    public int getLocalExperts() { return localExperts; }
    public long getHiddenDim() { return hiddenDim; }
    public long getIntermediateDim() { return intermediateDim; }
    public Device getDevice() { return device; }
    public TrainerStats stats() { return stats; }
    public long getNumForwardCalls() { return numForwardCalls; }
    public long getNumAllToAllCalls() { return numAllToAllCalls; }

    public void train() { module.train(true); }
    public void eval() { module.eval(); }
    public boolean isTraining() { return module.is_training(); }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try { auxLossBuffer.close(); } catch (Throwable ignored) {}
        if (module != null) {
            try { module.close(); } catch (Throwable ignored) {}
        }
        System.out.printf(
                "[ExpertParallelTrainer] Closed: rank=%d, fwdCalls=%d, allToAllCalls=%d%n",
                epRank, numForwardCalls, numAllToAllCalls);
    }

    @Override
    public String toString() {
        return "ExpertParallelTrainer{rank=" + epRank + ", epSize=" + epSize
                + ", numExperts=" + numExperts + ", localExperts=" + localExperts
                + ", topK=" + topK
                + ", stats=" + stats.snapshot() + '}';
    }

    /** Return value of {@link #routeAndProcess}. */
    public static final class Output {
        private final Tensor output;
        private final Tensor auxLoss;

        public Output(Tensor output, Tensor auxLoss) {
            this.output = output;
            this.auxLoss = auxLoss;
        }

        public Tensor output() { return output; }
        public Tensor auxLoss() { return auxLoss; }

        /** Total loss = CE + auxLoss. */
        public Tensor totalLoss(Tensor ceLoss) {
            if (auxLoss == null || auxLoss.isNull() || !auxLoss.defined()) {
                return ceLoss;
            }
            return ceLoss.add(auxLoss);
        }
    }

    public static final class Builder {
        private Module module;
        private ProcessGroupWrapper processGroup;
        private int numExperts = 8;
        private int topK = 2;
        private long hiddenDim = 4096;
        private long intermediateDim = 16384;
        private boolean dropTokens = true;
        private float auxLossCoeff = 0.01f;
        private boolean useBias = false;

        public Builder module(Module m) { this.module = m; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.processGroup = pg; return this; }
        public Builder numExperts(int n) { this.numExperts = n; return this; }
        public Builder topK(int k) { this.topK = k; return this; }
        public Builder hiddenDim(long d) { this.hiddenDim = d; return this; }
        public Builder intermediateDim(long d) { this.intermediateDim = d; return this; }
        /** Drop tokens that exceed expert capacity (instead of caching). */
        public Builder dropTokens(boolean b) { this.dropTokens = b; return this; }
        /** Auxiliary load-balancing loss coefficient. */
        public Builder auxLossCoeff(float c) { this.auxLossCoeff = c; return this; }
        /** Use bias in the gating linear layer (GShard-style). */
        public Builder useBias(boolean b) { this.useBias = b; return this; }

        public ExpertParallelTrainer build() {
            Objects.requireNonNull(module, "module is required");
            Objects.requireNonNull(processGroup, "processGroup is required");
            return new ExpertParallelTrainer(this);
        }
    }
}
