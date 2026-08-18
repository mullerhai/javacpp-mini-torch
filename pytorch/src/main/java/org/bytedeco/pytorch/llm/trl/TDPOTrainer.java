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
package org.bytedeco.pytorch.llm.trl;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.trl.config.TDPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.gather;
import static org.bytedeco.pytorch.global.torch.log_softmax;
import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * TDPO (Token-level Direct Preference Optimization) trainer.
 *
 * <p>Enterprise features:
 * <ul>
 *   <li>Token-level forward KL regularization</li>
 *   <li>PPO-style clipped surrogate (per-token advantage)</li>
 *   <li>Token-level label smoothing and γ reward shaping</li>
 *   <li>SFT NLL auxiliary loss</li>
 *   <li>Length normalization, max length clamp</li>
 *   <li>NaN/Inf guard</li>
 *   <li>Precomputed logprob fast path</li>
 * </ul>
 *
 * <p>Reference: "Token-level Direct Preference Optimization" (Dong et al., 2024)
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class TDPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "tdpo";

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;
    private final LlmForward referenceForward;
    private final TDPOConfig tdpoConfig;
    private final TensorVector params;

    private double totalTokenAcc;
    private int tokenCount;

    public TDPOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            TDPOConfig config) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.reference = reference;
        this.referenceForward = referenceForward;
        this.tdpoConfig = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        if (reference != null) freeze(reference);
        System.out.printf(
                "[TDPOTrainer v%s] beta=%.3f, clip=%.3f, fwdKL=%.3f, sftW=%.3f%n",
                VERSION, tdpoConfig.beta(), tdpoConfig.clipRange(),
                tdpoConfig.forwardKlCoef(), tdpoConfig.sftWeight());
    }

    public TDPOTrainer(
            Module policy,
            LlmForward policyForward,
            Optimizer optimizer,
            TDPOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public Module reference() { return reference; }
    public TDPOConfig config() { return tdpoConfig; }

    @Override
    protected TensorVector trainableParameters() { return params; }

    @Override
    public void train() {
        super.train();
        policy.train(true);
        if (reference != null) reference.eval();
    }

    @Override
    public void eval() {
        super.eval();
        policy.eval();
        if (reference != null) reference.eval();
    }

    @Override
    public Tensor computeLoss(Map<String, Tensor> batch) {
        // Fast path: precomputed token log-probs
        if (hasKey(batch, "chosen_token_logps") && hasKey(batch, "rejected_token_logps")) {
            Tensor chosenLp = batch.get("chosen_token_logps");
            Tensor rejectedLp = batch.get("rejected_token_logps");
            Tensor refC = batch.get("ref_chosen_token_logps");
            Tensor refR = batch.get("ref_rejected_token_logps");
            Tensor cMask = batch.get("chosen_attention_mask");
            Tensor rMask = batch.get("rejected_attention_mask");
            if (refC == null || !refC.defined()) refC = zeros_like(chosenLp);
            if (refR == null || !refR.defined()) refR = zeros_like(rejectedLp);
            return combine(batch, chosenLp, rejectedLp, refC, refR, cMask, rMask, null, null, null);
        }

        Tensor chosenIds = require(batch, "chosen_input_ids");
        Tensor rejectedIds = require(batch, "rejected_input_ids");
        Tensor chosenMask = batch.get("chosen_attention_mask");
        Tensor rejectedMask = batch.get("rejected_attention_mask");
        Tensor chosenLabels = orElse(batch.get("chosen_labels"), chosenIds);
        Tensor rejectedLabels = orElse(batch.get("rejected_labels"), rejectedIds);

        Tensor chosenLogits = policyForward.forward(chosenIds, chosenMask);
        Tensor rejectedLogits = policyForward.forward(rejectedIds, rejectedMask);

        Tensor chosenTokenLp = tokenLogProbs(chosenLogits, chosenLabels, chosenMask);
        Tensor rejectedTokenLp = tokenLogProbs(rejectedLogits, rejectedLabels, rejectedMask);

        Tensor refChosenLp;
        Tensor refRejectedLp;
        if (referenceForward != null) {
            try (NoGradGuard guard = new NoGradGuard()) {
                Tensor refCLogits = referenceForward.forward(chosenIds, chosenMask);
                Tensor refRLogits = referenceForward.forward(rejectedIds, rejectedMask);
                refChosenLp = tokenLogProbs(refCLogits, chosenLabels, chosenMask).detach();
                refRejectedLp = tokenLogProbs(refRLogits, rejectedLabels, rejectedMask).detach();
            }
        } else {
            refChosenLp = zeros_like(chosenTokenLp);
            refRejectedLp = zeros_like(rejectedTokenLp);
        }

        return combine(batch, chosenTokenLp, rejectedTokenLp, refChosenLp, refRejectedLp,
                chosenMask, rejectedMask, chosenLogits, chosenLabels, chosenMask);
    }

    private Tensor combine(Map<String, Tensor> batch,
                           Tensor chosenTokenLp, Tensor rejectedTokenLp,
                           Tensor refChosenLp, Tensor refRejectedLp,
                           Tensor chosenMask, Tensor rejectedMask,
                           Tensor chosenLogits, Tensor chosenLabels, Tensor chosenMaskForNll) {
        double beta = tdpoConfig.beta();
        double clipRange = tdpoConfig.clipRange();
        double forwardKlCoef = tdpoConfig.forwardKlCoef();
        double gamma = tdpoConfig.gamma();
        double ls = tdpoConfig.labelSmoothing();

        Tensor chosenDiff = chosenTokenLp.sub(refChosenLp);
        Tensor rejectedDiff = rejectedTokenLp.sub(refRejectedLp);

        Tensor ratio = chosenDiff.sub(rejectedDiff).exp();
        Tensor ratioClipped = ratio.clamp(
                new ScalarOptional(new Scalar(1.0 - clipRange)),
                new ScalarOptional(new Scalar(1.0 + clipRange)));

        Tensor advantage = chosenDiff.sub(rejectedDiff);

        Tensor surr1 = ratio.mul(advantage);
        Tensor surr2 = ratioClipped.mul(advantage);
        Tensor policyLoss = surr1.min(surr2).mean().neg();

        // Forward KL
        Tensor forwardKl = chosenTokenLp.sub(refChosenLp);
        if (chosenMask != null) forwardKl = forwardKl.mul(chosenMask);
        Tensor klLoss = forwardKl.mean().mul(new Scalar(forwardKlCoef));

        // Optional γ-based reward shaping: blend advantage with smoothed target
        if (gamma > 0.0) {
            Tensor weighted = advantage.mul(new Scalar(gamma))
                    .add(advantage.mean().mul(new Scalar(1.0 - gamma)));
            policyLoss = weighted.mul(surr1.min(surr2)).mean().neg();
        }

        // Label smoothing: shift advantage towards 0
        if (ls > 0.0) {
            Tensor smoothedAdv = advantage.mul(new Scalar(1.0 - ls));
            Tensor smLoss = smoothedAdv.mul(surr1.min(surr2)).mean().neg();
            policyLoss = smLoss.mul(new Scalar(1.0 - ls)).add(policyLoss.mul(new Scalar(ls)));
        }

        Tensor totalLoss = policyLoss.add(klLoss);

        // Length-normalize (per-token mean instead of sum).
        if (tdpoConfig.lengthNormalize()) {
            Tensor cCount = (chosenMask != null && chosenMask.defined())
                    ? chosenMask.sum().add(new Scalar(1e-8))
                    : torch.tensor(chosenTokenLp.size(1));
            Tensor rCount = (rejectedMask != null && rejectedMask.defined())
                    ? rejectedMask.sum().add(new Scalar(1e-8))
                    : torch.tensor(rejectedTokenLp.size(1));
            Tensor cSeq = (chosenTokenLp.mul(chosenMask != null ? chosenMask : onesLike(chosenTokenLp))).sum(1).div(cCount);
            Tensor rSeq = (rejectedTokenLp.mul(rejectedMask != null ? rejectedMask : onesLike(rejectedTokenLp))).sum(1).div(rCount);
            Tensor seqAdv = cSeq.sub(rSeq).mul(new Scalar(beta));
            Tensor seqLoss = seqAdv.sigmoid().log().neg().mean();
            totalLoss = totalLoss.mul(new Scalar(1.0 - tdpoConfig.sftWeight()))
                    .add(seqLoss.mul(new Scalar(tdpoConfig.sftWeight())));
        }

        // SFT auxiliary
        if (tdpoConfig.sftWeight() > 0.0 && chosenLogits != null && chosenLogits.defined()) {
            Tensor sft = DPOLoss.sftNll(chosenLogits, chosenLabels, chosenMaskForNll);
            totalLoss = totalLoss.add(sft.mul(new Scalar(tdpoConfig.sftWeight())));
        }

        // Aux loss coef (MoE)
        if (tdpoConfig.auxLossCoef() > 0.0) {
            // Pull auxiliary loss from batch (MoE router or other model exras)
            Tensor aux = batchAux(batch);
            if (aux != null && aux.defined()) {
                totalLoss = totalLoss.add(aux.mul(new Scalar(tdpoConfig.auxLossCoef())));
            }
        }

        double v = totalLoss.item_double();
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            System.err.println("[TDPOTrainer] WARNING: NaN/Inf loss; falling back to SFT term.");
            if (chosenLogits != null && chosenLogits.defined()) {
                totalLoss = DPOLoss.sftNll(chosenLogits, chosenLabels, chosenMaskForNll);
            }
        }

        updateTokenAccuracy(chosenTokenLp, rejectedTokenLp);
        numTrainingSteps++;
        return totalLoss;
    }

    /**
     * Auxiliary loss from the batch.  Supports conventional keys used by
     * MoE-style model wrappers: {@code aux_loss}, {@code router_loss},
     * {@code moe_loss}.  Returns {@code null} if none are present.
     */
    private static Tensor batchAux(Map<String, Tensor> batch) {
        if (batch == null) return null;
        String[] keys = {"aux_loss", "router_loss", "moe_loss", "z_loss"};
        for (String k : keys) {
            Tensor t = batch.get(k);
            if (t != null && t.defined()) return t;
        }
        return null;
    }

    private static Tensor onesLike(Tensor t) {
        return t.detach().mul(new Scalar(0.0)).add(new Scalar(1.0));
    }

    private Tensor tokenLogProbs(Tensor logits, Tensor labels, Tensor mask) {
        Tensor logProbs = log_softmax(logits, -1);
        Tensor tokenLp = gatherLogProbs(logProbs, labels);
        if (mask != null && mask.defined()) {
            tokenLp = tokenLp.mul(mask);
        }
        return tokenLp;
    }

    private Tensor gatherLogProbs(Tensor logProbs, Tensor labels) {
        // logProbs: [B, T, V], labels: [B, T]
        long B = logProbs.size(0);
        long T = logProbs.size(1);
        long V = logProbs.size(2);
        Tensor expanded = labels.reshape(new long[]{B * T}).to(org.bytedeco.pytorch.global.torch.ScalarType.Long).unsqueeze(1);
        Tensor flat = logProbs.reshape(new long[]{B * T, V});
        Tensor gathered = gather(flat, 1, expanded);
        return gathered.reshape(new long[]{B, T});
    }

    private void updateTokenAccuracy(Tensor chosenTokenLp, Tensor rejectedTokenLp) {
        try {
            Tensor tokenCorrect = chosenTokenLp.gt(rejectedTokenLp);
            totalTokenAcc += tokenCorrect.sum().item_double();
            tokenCount += (int) chosenTokenLp.size(0) * (int) chosenTokenLp.size(1);
        } catch (Exception ignored) {}
    }

    public double getTokenAccuracy() {
        return tokenCount > 0 ? totalTokenAcc / tokenCount : 0.0;
    }

    public void resetMetrics() {
        totalTokenAcc = 0.0;
        tokenCount = 0;
    }

    private static boolean hasKey(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        return t != null && t.defined();
    }

    private static void freeze(Module m) {
        TensorVector pv = m.parameters();
        for (long i = 0, n = pv.size(); i < n; i++) {
            Tensor p = pv.get(i);
            if (p != null && !p.isNull() && p.defined()) {
                p.requires_grad_(false);
            }
        }
        m.eval();
    }

    private static Tensor orElse(Tensor a, Tensor b) {
        return a != null && a.defined() ? a : b;
    }

    private static Tensor require(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        if (t == null || !t.defined()) {
            throw new IllegalArgumentException("batch missing required key: " + key);
        }
        return t;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        super.close();
        System.out.printf("[TDPOTrainer v%s] Closed: steps=%d, tokenAcc=%.2f%%%n",
                VERSION, numTrainingSteps, getTokenAccuracy() * 100);
    }

    public boolean isClosed() { return closed; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "TDPO v" + VERSION + " (Token-level Direct Preference Optimization)";
    }
}