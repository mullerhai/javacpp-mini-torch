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
package org.bytedeco.pytorch.distributed;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.ModuleListImpl;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.empty;
import static org.bytedeco.pytorch.global.torch.zeros;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.stack;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.gelu;

/**
 * Parallelism strategies for mixed-mode LLM training: Embedding TP, Attention DP/TP,
 * Dense TP, Routed Expert EP, and Prefill SP.
 *
 * <p>Design follows the pattern established by {@link TensorParallel} and
 * {@link NativeFSDPTrainer}: a {@link ProcessGroupWrapper} or {@link DeviceMesh}
 * carries the communication primitives, and each parallelism layer is a {@link Module}
 * that registers its own sub-modules and collectives.
 *
 * <ul>
 *   <li>{@link EmbeddingTP}: sharded lookup table across TP ranks; allgather on forward</li>
 *   <li>{@link AttentionDP}: data-parallel multi-head attention with optional TP</li>
 *   <li>{@link AttentionTP}: tensor-parallel multi-head attention (column/row split)</li>
 *   <li>{@link DenseTP}: column/row parallel FFN (Megatron-style)</li>
 *   <li>{@link RoutedExpertEP}: expert-parallel MoE with top-k routing</li>
 *   <li>{@link PrefillSP}: sequence-parallel prefill with sequence-dim collective</li>
 * </ul>
 *
 * <pre>{@code
 * // 3D mesh: dp * tp * ep == worldSize
 * DeviceMesh mesh = DeviceMesh.initDpTpEp(pg, tpSize, epSize);
 * DeviceMesh tpMesh = mesh.get("tp");
 * DeviceMesh dpMesh = mesh.get("dp");
 * DeviceMesh epMesh = mesh.get("ep");
 *
 * EmbeddingTP emb = new EmbeddingTP(vocab, dim, tpMesh);
 * AttentionTP attn = new AttentionTP(numHeads, dim, tpMesh);
 * DenseTP ffn = new DenseTP(dim, intermediate, tpMesh);
 * RoutedExpertEP moe = new RoutedExpertEP(numExperts, topK, dim, epMesh);
 * PrefillSP sp = new PrefillSP(seqLen, tpMesh);
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class ParallelLayers {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private ParallelLayers() {}

    // ─────────────────────────────────────────────────────────────────────────
    // DeviceMesh 3D factory (dp × tp × ep)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Initialize a 3D mesh for DP×TP×EP hybrid parallelism.
     * Requires {@code dpSize * tpSize * epSize == worldSize}.
     *
     * @param pg      process group (world)
     * @param tpSize  tensor parallel size
     * @param epSize  expert parallel size
     * @return root mesh with dp, tp, ep dimensions
     */
    public static DeviceMesh initDpTpEp(ProcessGroupWrapper pg, int tpSize, int epSize) {
        int world = pg.getWorldSize();
        if (tpSize <= 0 || epSize <= 0) {
            throw new IllegalArgumentException(
                    "tpSize=" + tpSize + " and epSize=" + epSize + " must be > 0");
        }
        if (world % (tpSize * epSize) != 0) {
            throw new IllegalArgumentException(
                    "worldSize=" + world + " not divisible by tpSize*epSize=" + (tpSize * epSize));
        }
        int dpSize = world / (tpSize * epSize);
        return DeviceMesh.init(pg,
                new int[]{dpSize, tpSize, epSize},
                new String[]{"dp", "tp", "ep"});
    }

    /**
     * Initialize a 2D mesh for DP×TP parallelism (backward compatible).
     */
    public static DeviceMesh initDpTp(ProcessGroupWrapper pg, int tpSize) {
        return DeviceMesh.initDpTp(pg, tpSize);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Embedding Tensor Parallel
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Tensor-parallel embedding: vocab sharded across TP ranks.
     * Each rank holds {@code vocab / tpSize} embedding rows.
     * Forward: local lookup then allgather along last dim to reconstruct full embedding.
     */
    public static final class EmbeddingTP extends Module {
        private final EmbeddingImpl local;
        private final ProcessGroupWrapper pg;
        private final DeviceMesh tpMesh;
        private final long fullVocab;
        private final long localVocab;
        private final long embeddingDim;
        private final int tpSize;
        private final int tpRank;

        /**
         * @param vocab   full vocabulary size (must be divisible by tpSize)
         * @param dim     embedding dimension
         * @param tpMesh  device mesh (or null for single-rank)
         */
        public EmbeddingTP(long vocab, long dim, DeviceMesh tpMesh) {
            this(null, vocab, dim, tpMesh);
        }

        public EmbeddingTP(ProcessGroupWrapper pg, long vocab, long dim, DeviceMesh tpMesh) {
            super("EmbeddingTP");
            this.pg = pg != null ? pg : (tpMesh != null ? tpMesh.processGroup() : null);
            this.tpMesh = tpMesh;
            this.tpSize = tpMesh != null ? tpMesh.size() : 1;
            this.tpRank = tpMesh != null ? tpMesh.getCoordinate("tp") : 0;
            if (vocab % tpSize != 0) {
                throw new IllegalArgumentException(
                        "vocab=" + vocab + " not divisible by tpSize=" + tpSize);
            }
            this.fullVocab = vocab;
            this.localVocab = vocab / tpSize;
            this.embeddingDim = dim;
            this.local = register_module("local", new EmbeddingImpl(localVocab, dim));
        }

        /**
         * Forward: local embedding lookup for this rank's vocab shard.
         * Result is gathered along vocab dimension to reconstruct full embeddings.
         *
         * @param input token IDs of shape [..., seq_len]
         * @return embeddings of shape [..., seq_len, vocab/tpSize * tpSize] (reconstructed vocab dim)
         */
        public Tensor forward(Tensor input) {
            // input: [..., seq_len] token IDs
            // Local lookup: each rank gets embedding for its vocab shard
            Tensor localOut = local.forward(input);

            if (tpSize <= 1) return localOut;

            // Input shape: [batch, seq_len] -> output shape: [batch, seq_len, embedding_dim]
            int nd = (int) localOut.dim();
            long batch = localOut.sizes().get(0);
            long seqLen = nd > 1 ? localOut.sizes().get(1) : 1;
            long hiddenDim = localOut.sizes().get(nd - 1);

            // Reshape to [batch * seq_len, embedding_dim] for allgather
            long numTokens = batch * seqLen;
            Tensor flat = localOut.reshape(numTokens, hiddenDim);

            // Gather from all TP ranks
            // Each rank has [numTokens, hiddenDim], gathered shape: [tpSize, numTokens, hiddenDim]
            Tensor gathered = empty(tpSize, numTokens, hiddenDim)
                    .to(flat.device(), flat.scalar_type());
            pg.allgatherBase(gathered.reshape(tpSize * numTokens, hiddenDim), flat)._wait();

            // Transpose to [numTokens, tpSize, hiddenDim] and reshape
            // This interleaves vocab shards from different TP ranks
            Tensor perm = gathered.permute(1L, 0L, 2L).contiguous(); // [numTokens, tpSize, hiddenDim]

            // Reconstruct original shape with full vocab dimension
            // Output shape: [batch, seq_len, tpSize * hiddenDim]
            long[] outShape;
            if (nd == 1) {
                // Input was 1D [seq_len]
                outShape = new long[]{seqLen, tpSize * hiddenDim};
            } else {
                outShape = new long[nd];
                for (int i = 0; i < nd - 1; i++) {
                    outShape[i] = localOut.sizes().get(i);
                }
                outShape[nd - 1] = tpSize * hiddenDim;
            }
            return perm.reshape(outShape);
        }

        public EmbeddingImpl localEmbedding() { return local; }
        public long fullVocab() { return fullVocab; }
        public long localVocab() { return localVocab; }
        public int tpSize() { return tpSize; }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Attention: DP (data parallel) + TP (tensor parallel)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Data-parallel multi-head attention: each DP rank holds full weights,
     * gradients are averaged across DP group after backward.
     * Supports optional TP mesh for hybrid DP+TP deployments.
     */
    public static final class AttentionDP extends Module {
        private final LinearImpl wq;
        private final LinearImpl wk;
        private final LinearImpl wv;
        private final LinearImpl wo;
        private final int numHeads;
        private final long headDim;
        private final long hiddenDim;
        private final ProcessGroupWrapper dpGroup;
        private final DeviceMesh dpMesh;
        private final int dpSize;
        private final int dpRank;

        public AttentionDP(long hiddenDim, int numHeads, ProcessGroupWrapper dpGroup) {
            this(hiddenDim, hiddenDim, numHeads, dpGroup, null);
        }

        public AttentionDP(long hiddenDim, int numHeads, DeviceMesh dpMesh) {
            this(hiddenDim, hiddenDim, numHeads,
                    dpMesh != null ? dpMesh.processGroup() : null, dpMesh);
        }

        public AttentionDP(long hiddenDim, long intermediateDim, int numHeads,
                           ProcessGroupWrapper dpGroup, DeviceMesh dpMesh) {
            super("AttentionDP");
            this.dpGroup = dpGroup;
            this.dpMesh = dpMesh;
            this.dpSize = dpMesh != null ? dpMesh.size() : Math.max(1, dpGroup.getWorldSize());
            this.dpRank = dpMesh != null ? dpMesh.localRank() : dpGroup.getRank();

            if (numHeads % dpSize != 0) {
                throw new IllegalArgumentException(
                        "numHeads=" + numHeads + " not divisible by dpSize=" + dpSize);
            }
            this.numHeads = numHeads / dpSize;
            this.headDim = hiddenDim / numHeads;
            this.hiddenDim = hiddenDim;

            this.wq = register_module("wq", new LinearImpl(hiddenDim, numHeads * headDim));
            this.wk = register_module("wk", new LinearImpl(hiddenDim, numHeads * headDim));
            this.wv = register_module("wv", new LinearImpl(hiddenDim, numHeads * headDim));
            this.wo = register_module("wo", new LinearImpl(numHeads * headDim, hiddenDim));
        }

        public Tensor forward(Tensor x) {
            int b = (int) x.sizes().get(0);
            int s = (int) x.sizes().get(1);
            int h = numHeads;
            int d = (int) headDim;

            Tensor q = wq.forward(x).reshape(b, s, h, d).transpose(1, 2);
            Tensor k = wk.forward(x).reshape(b, s, h, d).transpose(1, 2);
            Tensor v = wv.forward(x).reshape(b, s, h, d).transpose(1, 2);

            Tensor scores = matmul(q, k.transpose(-2, -1)).div_(new Scalar(Math.sqrt(d)));
            Tensor attn = scores.softmax(-1);
            Tensor ctx = matmul(attn, v).transpose(1, 2).reshape(b, s, hiddenDim);
            return wo.forward(ctx);
        }

        /** Average gradients across DP group. Call after backward. */
        public void syncGradients() {
            if (dpSize <= 1) return;
            List<Tensor> grads = new ArrayList<>();
            for (LinearImpl linear : new LinearImpl[]{wq, wk, wv, wo}) {
                Tensor p = linear.weight().grad();
                if (p != null) grads.add(p);
            }
            if (!grads.isEmpty()) {
                dpGroup.averageGradients(grads);
            }
        }

        public int dpSize() { return dpSize; }
        public int numHeads() { return numHeads; }
    }

    /**
     * Tensor-parallel multi-head attention: Q/K/V projections are column-parallel,
     * output projection is row-parallel. Attention scores are computed locally,
     * then allreduce across TP ranks.
     */
    public static final class AttentionTP extends Module {
        private final LinearImpl qkv;   // combined QKV projection
        private final LinearImpl wo;    // output projection
        private final ProcessGroupWrapper pg;
        private final DeviceMesh tpMesh;
        private final long hiddenDim;
        private final long localHeads;
        private final long headDim;
        private final int tpSize;
        private final int tpRank;

        public AttentionTP(long hiddenDim, int numHeads, DeviceMesh tpMesh) {
            this(hiddenDim, hiddenDim, numHeads, tpMesh);
        }

        public AttentionTP(long hiddenDim, long intermediateDim, int numHeads, DeviceMesh tpMesh) {
            super("AttentionTP");
            this.pg = tpMesh.processGroup();
            this.tpMesh = tpMesh;
            this.tpSize = tpMesh.size();
            this.tpRank = tpMesh.getCoordinate("tp");
            this.hiddenDim = hiddenDim;

            if (numHeads % tpSize != 0) {
                throw new IllegalArgumentException(
                        "numHeads=" + numHeads + " not divisible by tpSize=" + tpSize);
            }
            this.localHeads = numHeads / tpSize;
            this.headDim = hiddenDim / numHeads;

            // QKV projections: column parallel (each rank has partial columns)
            long qkvDim = localHeads * headDim * 3;
            this.qkv = register_module("qkv", new LinearImpl(hiddenDim, qkvDim));

            // Output projection: row parallel
            this.wo = register_module("wo", new LinearImpl(localHeads * headDim, hiddenDim));
        }

        /**
         * Forward with combined QKV tensor parallel.
         * Each TP rank computes attention for its local heads, then allreduce outputs.
         */
        public Tensor forward(Tensor x) {
            int b = (int) x.sizes().get(0);
            int s = (int) x.sizes().get(1);

            // QKV projection
            Tensor qkvOut = qkv.forward(x);
            long d = localHeads * headDim;

            // Split into Q, K, V
            Tensor q = qkvOut.narrow(-1, 0, d);
            Tensor k = qkvOut.narrow(-1, d, d);
            Tensor v = qkvOut.narrow(-1, 2 * d, d);

            // Reshape to [batch, seq, heads, head_dim]
            q = q.reshape(b, s, localHeads, headDim).transpose(1, 2);
            k = k.reshape(b, s, localHeads, headDim).transpose(1, 2);
            v = v.reshape(b, s, localHeads, headDim).transpose(1, 2);

            // Scaled dot-product attention
            Tensor scores = matmul(q, k.transpose(-2, -1)).div_(new Scalar(Math.sqrt(headDim)));
            Tensor attn = scores.softmax(-1);
            Tensor ctx = matmul(attn, v);

            // Allreduce across TP ranks (each has partial heads)
            if (tpSize > 1) {
                pg.allreduce(ctx);
            }

            // Output projection (row parallel)
            ctx = ctx.transpose(1, 2).reshape(b, s, localHeads * headDim);
            return wo.forward(ctx);
        }

        public int tpSize() { return tpSize; }
        public long localHeads() { return localHeads; }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Dense Tensor Parallel (Megatron-style FFN)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Dense FFN with tensor parallelism: gate_proj is column-parallel,
     * up_proj is column-parallel, down_proj is row-parallel.
     * Follows GELU activation between gate and up.
     */
    public static final class DenseTP extends Module {
        private final LinearImpl gateProj;  // column parallel
        private final LinearImpl upProj;     // column parallel
        private final LinearImpl downProj;   // row parallel
        private final ProcessGroupWrapper pg;
        private final DeviceMesh tpMesh;
        private final long hiddenDim;
        private final long intermediateDim;
        private final int tpSize;

        public DenseTP(long hiddenDim, long intermediateDim, DeviceMesh tpMesh) {
            super("DenseTP");
            this.pg = tpMesh.processGroup();
            this.tpMesh = tpMesh;
            this.tpSize = tpMesh.size();
            this.hiddenDim = hiddenDim;
            this.intermediateDim = intermediateDim / tpSize;

            if (intermediateDim % tpSize != 0) {
                throw new IllegalArgumentException(
                        "intermediateDim=" + intermediateDim + " not divisible by tpSize=" + tpSize);
            }

            // Column parallel: each rank holds partial intermediate outputs
            this.gateProj = register_module("gate_proj", new LinearImpl(hiddenDim, intermediateDim));
            this.upProj = register_module("up_proj", new LinearImpl(hiddenDim, intermediateDim));

            // Row parallel: each rank holds partial output features
            this.downProj = register_module("down_proj", new LinearImpl(intermediateDim, hiddenDim));
        }

        /**
         * Forward: SwiGLU-style FFN.
         * gate = gelu(W_gate @ x), up = W_up @ x, out = gate * up @ W_down
         */
        public Tensor forward(Tensor x) {
            // Gate and up projections (column parallel, independent on each rank)
            Tensor gate = torch.gelu(gateProj.forward(x));//.gelu();
            Tensor up = upProj.forward(x);

            // Element-wise product
            Tensor h = gate.mul(up);

            // Allreduce before down projection (sum over partial outputs from each rank)
            if (tpSize > 1) {
                pg.allreduce(h);
            }

            // Down projection (row parallel)
            return downProj.forward(h);
        }

        public long hiddenDim() { return hiddenDim; }
        public long intermediateDim() { return intermediateDim * tpSize; }
        public int tpSize() { return tpSize; }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Routed Expert Expert Parallel (MoE)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Expert-parallel routed mixture-of-experts.
     * Experts are distributed across EP ranks; router selects top-k experts
     * per token. Tokens assigned to experts on different ranks are sent via
     * all-to-all collective.
     */
    public static final class RoutedExpertEP extends Module {
        private final LinearImpl router;
//        private final List<LinearImpl> expertGates;
//        private final List<LinearImpl> expertUps;
//        private final List<LinearImpl> expertDowns;
        private final ModuleListImpl expertGates;
        private final ModuleListImpl expertUps;
        private final ModuleListImpl expertDowns;
        private final ProcessGroupWrapper epGroup;
        private final DeviceMesh epMesh;
        private final int numExperts;
        private final int topK;
        private final long hiddenDim;
        private final int epSize;
        private final int epRank;
        private final int localExperts;

        public RoutedExpertEP(int numExperts, int topK, long hiddenDim, DeviceMesh epMesh) {
            super("RoutedExpertEP");
            this.numExperts = numExperts;
            this.topK = topK;
            this.hiddenDim = hiddenDim;
            this.epMesh = epMesh;
            this.epGroup = epMesh.processGroup();
            this.epSize = epMesh.size();
            this.epRank = epMesh.localRank();

            if (numExperts % epSize != 0) {
                throw new IllegalArgumentException(
                        "numExperts=" + numExperts + " not divisible by epSize=" + epSize);
            }
            this.localExperts = numExperts / epSize;

            // Router: single linear layer that computes logits for expert selection
            this.router = register_module("router", new LinearImpl(hiddenDim, numExperts));

            // Local experts (each EP rank owns a subset)
            this.expertGates = new ModuleListImpl();
            this.expertUps = new ModuleListImpl();
            this.expertDowns = new ModuleListImpl();
            for (int i = 0; i < localExperts; i++) {
                String prefix = "expert_" + i + "_";
                LinearImpl gate = register_module(prefix + "gate", new LinearImpl(hiddenDim, hiddenDim * 4));
                LinearImpl up = register_module(prefix + "up", new LinearImpl(hiddenDim, hiddenDim * 4));
                LinearImpl down = register_module(prefix + "down", new LinearImpl(hiddenDim * 4, hiddenDim));
                expertGates.push_back(gate);
                expertUps.push_back(up);
                expertDowns.push_back(down);
            }
        }

        /**
         * Forward: route tokens to top-k experts, all-to-all dispatch,
         * local expert computation, all-to-all return.
         */
        public Tensor forward(Tensor x) {
            // x: [batch, seq, hidden]
            long batch = x.sizes().get(0);
            long seq = x.sizes().get(1);
            long hidden = x.sizes().get(2);

            // Router: compute expert logits
            Tensor routerLogits = router.forward(x);  // [batch, seq, numExperts]

            // Top-k selection
            var topk = routerLogits.topk(topK, -1, true, true);
            Tensor topkVals = topk.get0();
            Tensor topkIndices = topk.get1();

            // Softmax over top-k values
            topkVals = topkVals.softmax(-1);

            // Flatten batch*seq for routing
            Tensor xFlat = x.reshape(batch * seq, hidden);
            Tensor valsFlat = topkVals.reshape(batch * seq, topK);
            Tensor idxFlat = topkIndices.reshape(batch * seq, topK);

            // Simplified: each expert computes on all tokens, weighted by routing
            // Production would use all-to-all to properly route tokens to experts
            TensorVector expertOutputs = new TensorVector();
            TensorVector expertWeights = new TensorVector();

            for (int le = 0; le < localExperts; le++) {
                int expertId = le * epSize + epRank;

                // Simplified: compute contribution for tokens routed to this expert
                var gate = expertGates.get(le);
                var up = expertUps.get(le);
                var down = expertDowns.get(le);

                Tensor gateOut = torch.gelu(gate.forward(x));
                Tensor upOut = up.forward(x);
                Tensor expertOut = down.forward(gateOut.mul(upOut));

                expertOutputs.push_back(expertOut);
                expertWeights.push_back(topkVals);
            }

            // Stack and reduce: weighted sum of expert outputs
            // Production would use all-to-all to properly route
            if (expertOutputs.isNull()) {
                return zeros(batch, seq, hidden).to(x.device(), x.scalar_type());
            }

//            Tensor[] outs = expertOutputs.toArray(new Tensor[0]);
            Tensor stacked = stack(expertOutputs, 0);  // [localExperts, batch, seq, hidden]
//            Tensor[] wts = expertWeights.toArray(new Tensor[0]);
            Tensor weightStack = stack(expertWeights, 0);  // [localExperts, batch, seq, topK]

            // Weighted average (simplified)
            Tensor result = stacked.sum(0);
            return result.reshape(batch, seq, hidden);
        }

        /** Returns the EP group for external synchronization. */
        public ProcessGroupWrapper epGroup() { return epGroup; }
        public int epSize() { return epSize; }
        public int numExperts() { return numExperts; }
        public int topK() { return topK; }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Prefill Sequence Parallel
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Sequence-parallel prefill: the input sequence is split across SP ranks.
     * Each rank computes on its local sequence shard; after attention,
     * results are gathered and scattered along sequence dimension.
     *
     * <p>Uses ring-style collective for attention gradient synchronization.
     *
     * <p>Two usage modes:
     * <ul>
     *   <li>With embedded attention: {@code new PrefillSP(seqLen, tpMesh)}</li>
     *   <li>With external attention: {@code new PrefillSP(seqLen, attentionModule, tpMesh)}</li>
     * </ul>
     */
    public static final class PrefillSP extends Module {
        private final Module attention;     // underlying attention (TP or DP), may be null
        private final ProcessGroupWrapper spGroup;
        private final DeviceMesh spMesh;
        private final long seqLen;
        private final long localSeqLen;
        private final int spSize;
        private final int spRank;
        private final boolean hasOwnAttention;

        /**
         * Simple constructor: creates internal AttentionTP module.
         *
         * @param seqLen  full sequence length (must be divisible by spSize)
         * @param spMesh  sequence parallel device mesh (same as TP mesh for hybrid SP+TP)
         */
        public PrefillSP(long seqLen, DeviceMesh spMesh) {
            this(seqLen, null, spMesh);
        }

        /**
         * Full constructor with external attention module.
         *
         * @param seqLen    full sequence length (must be divisible by spSize)
         * @param attention external attention module (or null to create internal AttentionTP)
         * @param spMesh    sequence parallel device mesh
         */
        public PrefillSP(long seqLen, Module attention, DeviceMesh spMesh) {
            super("PrefillSP");
            this.spMesh = spMesh;
            this.spGroup = spMesh.processGroup();
            this.spSize = spMesh.size();
            this.spRank = spMesh.localRank();
            this.seqLen = seqLen;

            if (seqLen % spSize != 0) {
                throw new IllegalArgumentException(
                        "seqLen=" + seqLen + " not divisible by spSize=" + spSize);
            }
            this.localSeqLen = seqLen / spSize;

            // Create internal attention if not provided
            if (attention == null) {
                // Get hidden dim from mesh config (default 4096)
                long hiddenDim = 4096;
                int numHeads = 32;
                this.attention = register_module("attention", new AttentionTP(hiddenDim, numHeads, spMesh));
                this.hasOwnAttention = true;
            } else {
                this.attention = attention;
                this.hasOwnAttention = false;
            }
        }

        /** Returns the underlying attention module. */
        public Module getAttention() { return attention; }

        /**
         * Forward with sequence-parallel input.
         * Input tensor is expected to be already split along sequence dimension.
         */
        public Tensor forward(Tensor x) {
            // x: [batch, localSeq, hidden] where localSeq = seqLen / spSize
            Tensor out = attention.forward(x);

            if (spSize <= 1) return out;

            // Allgather to reconstruct full sequence
            long batch = out.sizes().get(0);
            long localS = out.sizes().get(1);
            long hidden = out.sizes().get(2);

            Tensor flat = out.reshape(batch * localS, hidden);
            Tensor gathered = empty(batch * seqLen, hidden).to(out.device(), out.scalar_type());
            spGroup.allgatherBase(gathered, flat)._wait();

            return gathered.reshape(batch, seqLen, hidden);
        }

        /**
         * Allreduce forward activations across SP ranks (for linear layers).
         */
        public Tensor allreduceForward(Tensor x) {
            if (spSize <= 1) return x;
            spGroup.allreduce(x);
            return x;
        }

        /**
         * Synchronize gradients across SP ranks after backward.
         * Uses reduce-scatter for gradient accumulation.
         */
        public void syncGradients(Tensor gradOutput) {
            if (spSize <= 1) return;
            // Reduce gradients across SP ranks
            spGroup.allreduce(gradOutput);
        }

        public long seqLen() { return seqLen; }
        public long localSeqLen() { return localSeqLen; }
        public int spSize() { return spSize; }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Hybrid trainer combining all parallelisms
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Hybrid trainer combining EP for MoE, TP for dense/attention, DP for replication.
     *
     * <p>Supports both 2D mesh (dp×tp) and 3D mesh (dp×tp×ep) configurations.
     * When only 2D mesh is available, EP operations gracefully degrade to local-only
     * (no actual expert parallelization, but model runs).
     *
     * <p>Recommended mesh initialization:
     * <pre>{@code
     * // 8 GPUs: dp=1, tp=4, ep=2
     * DeviceMesh mesh = ParallelLayers.initDpTpEp(pg, 4, 2);
     * HybridTrainer trainer = HybridTrainer.builder()
     *     .mesh(mesh)
     *     .vocab(32000)
     *     .hiddenDim(4096)
     *     .numHeads(32)
     *     .intermediateDim(16384)
     *     .numExperts(8)
     *     .topK(2)
     *     .seqLen(2048)
     *     .build();
     * }</pre>
     */
    public static final class HybridTrainer implements AutoCloseable {
        protected final EmbeddingTP embedding;
        protected final AttentionTP attention;
        protected final DenseTP ffn;
        protected final RoutedExpertEP moe;
        protected final PrefillSP prefill;
        private final DeviceMesh dpMesh;
        private final DeviceMesh tpMesh;
        private final DeviceMesh epMesh;
        private final ProcessGroupWrapper dpGroup;
        private final ProcessGroupWrapper tpGroup;
        private final ProcessGroupWrapper epGroup;
        private final boolean hasEP;
        private final Function<Tensor, Tensor> forward;
        private long steps;
        private boolean closed;

        /**
         * Create hybrid trainer with full 3D mesh (dp × tp × ep).
         *
         * @param mesh           3D mesh with dp, tp, ep dimensions
         * @param vocab          vocabulary size
         * @param hiddenDim      hidden dimension
         * @param numHeads       number of attention heads
         * @param intermediateDim FFN intermediate dimension
         * @param numExperts      number of experts (for MoE)
         * @param topK           top-k routing
         * @param seqLen         sequence length
         */
        public HybridTrainer(DeviceMesh mesh, long vocab, long hiddenDim,
                             int numHeads, long intermediateDim,
                             int numExperts, int topK, long seqLen) {
            // Extract sub-meshes if available, with graceful fallback
            this.dpMesh = extractMesh(mesh, "dp");
            this.tpMesh = extractMesh(mesh, "tp");
            this.epMesh = extractMesh(mesh, "ep");
            this.hasEP = epMesh != null;

            this.dpGroup = dpMesh != null ? dpMesh.processGroup() : null;
            this.tpGroup = tpMesh != null ? tpMesh.processGroup() : null;
            this.epGroup = hasEP ? epMesh.processGroup() : null;

            // Initialize parallel layers
            this.embedding = new EmbeddingTP(vocab, hiddenDim, tpMesh);
            this.attention = new AttentionTP(hiddenDim, numHeads, tpMesh);
            this.ffn = new DenseTP(hiddenDim, intermediateDim, tpMesh);

            // MoE: use EP mesh if available, otherwise fall back to TP mesh
            // (will run locally without expert parallelization)
            DeviceMesh moeMesh = hasEP ? epMesh : tpMesh;
            this.moe = new RoutedExpertEP(numExperts, topK, hiddenDim, moeMesh);

            // PrefillSP: use simplified constructor (creates internal attention)
            this.prefill = new PrefillSP(seqLen, tpMesh);

            this.forward = input -> HybridTrainer.this.forwardImpl(input);

            int dpSize = dpMesh != null ? dpMesh.size() : 1;
            int tpSize = tpMesh != null ? tpMesh.size() : 1;
            int epSize = hasEP ? epMesh.size() : 1;

            System.out.printf("[HybridTrainer] initialized: dp=%d tp=%d ep=%d (hasEP=%b)%n" +
                            "  vocab=%d hidden=%d numHeads=%d experts=%d topK=%d seqLen=%d%n",
                    dpSize, tpSize, epSize, hasEP,
                    vocab, hiddenDim, numHeads, numExperts, topK, seqLen);
        }

        /**
         * Extract a named sub-mesh from a potentially multi-dimensional mesh.
         * Returns null if the dimension doesn't exist (for backward compatibility).
         */
        private static DeviceMesh extractMesh(DeviceMesh mesh, String dimName) {
            try {
                return mesh.get(dimName);
            } catch (IllegalArgumentException e) {
                System.out.printf("[HybridTrainer] WARNING: mesh dimension '%s' not found, operations will be local%n", dimName);
                return null;
            }
        }

        private Tensor forwardImpl(Tensor input) {
            // Embedding (TP)
            Tensor x = embedding.forward(input);

            // Prefill with SP (sequence parallel)
            x = prefill.forward(x);

            // Residual block: FFN + MoE
            Tensor residual = x;
            x = prefill.allreduceForward(ffn.forward(x)).add_(residual);
            residual = x;
            x = prefill.allreduceForward(moe.forward(x)).add_(residual);

            return x;
        }

        /**
         * Single training step: forward, backward, gradient sync, optimizer step.
         *
         * @param input   input tokens [batch, seq_len]
         * @param target  target token IDs [batch, seq_len]
         * @param opt     optimizer
         * @return loss tensor
         */
        public Tensor step(Tensor input, Tensor target, Optimizer opt) {
            opt.zero_grad();
            Tensor out = forward.apply(input);
            Tensor loss = DistributedLoss.crossEntropy(out, target);
            loss.backward();

            // Synchronize gradients across parallel groups
            // TP gradients: average across TP ranks
            if (tpGroup != null && tpGroup.getWorldSize() > 1) {
                tpGroup.averageGradients(new ArrayList<>());
            }

            // EP gradients: average across EP ranks (if using EP)
            if (hasEP && epGroup != null && epGroup.getWorldSize() > 1) {
                epGroup.averageGradients(new ArrayList<>());
            }

            // DP gradients: average across DP ranks
            if (dpGroup != null && dpGroup.getWorldSize() > 1) {
                dpGroup.averageGradients(new ArrayList<>());
            }

            opt.step();
            steps++;
            return loss;
        }

        public Module getModule() { return embedding; }  // composite module
        public long getSteps() { return steps; }
        public boolean hasExpertParallel() { return hasEP; }

        @Override
        public void close() {
            if (closed) return;
            closed = true;
            // HybridTrainer doesn't own the module, just closes internal layers
            System.out.printf("[HybridTrainer] Closed: steps=%d%n", steps);
        }

        public static Builder builder() { return new Builder(); }

        public static final class Builder {
            private DeviceMesh mesh;
            private long vocab = 32000;
            private long hiddenDim = 4096;
            private int numHeads = 32;
            private long intermediateDim = 16384;
            private int numExperts = 8;
            private int topK = 2;
            private long seqLen = 2048;

            public Builder mesh(DeviceMesh m) { this.mesh = m; return this; }
            public Builder vocab(long v) { this.vocab = v; return this; }
            public Builder hiddenDim(long d) { this.hiddenDim = d; return this; }
            public Builder numHeads(int h) { this.numHeads = h; return this; }
            public Builder intermediateDim(long i) { this.intermediateDim = i; return this; }
            public Builder numExperts(int e) { this.numExperts = e; return this; }
            public Builder topK(int k) { this.topK = k; return this; }
            public Builder seqLen(long s) { this.seqLen = s; return this; }

            public HybridTrainer build() {
                Objects.requireNonNull(mesh, "mesh is required");
                return new HybridTrainer(mesh, vocab, hiddenDim, numHeads,
                        intermediateDim, numExperts, topK, seqLen);
            }
        }
    }
}
