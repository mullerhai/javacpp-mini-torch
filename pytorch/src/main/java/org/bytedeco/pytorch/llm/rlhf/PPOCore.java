/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.rlhf;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import static org.bytedeco.pytorch.global.torch.ScalarType;

/**
 * Internal helpers for the PPO algorithm — split out so PPOTrainer can be side-stepped
 * for unit testing.
 */
public final class PPOCore {

    public double policyLoss(Tensor logits, PPOTrainer.PPOBatch batch, PPOTrainer.Config cfg) {
        if (logits == null || batch.oldLogProbs == null || batch.advantages == null) return 0.0;
        Tensor logProbs = computeTokenLogProbs(logits, batch.inputIds);
        Tensor ratio = logProbs.sub(batch.oldLogProbs).exp();
        double lo = 1.0 - cfg.cliprange;
        double hi = 1.0 + cfg.cliprange;
        Tensor clipped = ratio.clamp(new ScalarOptional(new Scalar(lo)), new ScalarOptional(new Scalar(hi)));
        Tensor loss1 = ratio.mul(batch.advantages);
        Tensor loss2 = clipped.mul(batch.advantages);
        Tensor minLoss = torch.minimum(loss1, loss2);
        Tensor loss = minLoss.mean().neg();
        double v = loss.item_double();
        for (Tensor t : new Tensor[]{logProbs, ratio, clipped, loss1, loss2, minLoss, loss}) t.close();
        return v;
    }

    public double valueLoss(Tensor values, PPOTrainer.PPOBatch batch, PPOTrainer.Config cfg) {
        if (values == null || batch.returns == null) return 0.0;
        Tensor diff = values.sub(batch.returns);
        double lo = -cfg.cliprangeValue;
        double hi = cfg.cliprangeValue;
        Tensor clippedDiff = diff.clamp(new ScalarOptional(new Scalar(lo)), new ScalarOptional(new Scalar(hi)));
        Tensor loss = diff.mul(diff).mul(new Scalar(0.5));
        Tensor lossC = clippedDiff.mul(clippedDiff).mul(new Scalar(0.5));
        Tensor maxLoss = torch.maximum(loss, lossC);
        Tensor m = maxLoss.mean();
        double v = m.item_double();
        for (Tensor t : new Tensor[]{diff, clippedDiff, loss, lossC, maxLoss, m}) t.close();
        return v;
    }

    /** Compute per-token log probabilities using .select() for indexing. */
    public Tensor computeTokenLogProbs(Tensor logits, Tensor labels) {
        long[] shape = logits.sizes().vec().get();
        long bsz = shape[0];
        long seq = shape[1];
        long vocab = shape.length > 2 ? shape[2] : 0;
        Tensor logSoft = torch.log_softmax(logits, -1);
        // out[b, s] via select: take row b, then assign element s
        Tensor out = torch.zeros(new long[]{bsz, seq}, new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        for (long b = 0; b < bsz; b++) {
            Tensor outRow = out.select(0, (int) b);
            Tensor labelRow = labels.select(0, (int) b);
            Tensor logSoftRow = logSoft.select(0, (int) b);
            for (long s = 0; s < seq; s++) {
                long label = labelRow.select(0, (int) s).item_long();
                if (label < 0 || vocab > 0 && label >= vocab) continue;
                Tensor val = logSoftRow.select(0, (int) s);
                outRow.select(0, (int) s).fill_(val);
                val.close();
            }
        }
        logSoft.close();
        return out;
    }
}
