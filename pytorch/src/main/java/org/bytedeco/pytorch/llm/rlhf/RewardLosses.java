/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.rlhf;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;

import java.util.List;
import java.util.Map;

/**
 * Mirror of the pairwise logistic loss used by TRL's RewardTrainer.compute_loss.
 */
public final class RewardLosses {

    private RewardLosses() {}

    public static Tensor pairwiseLogSigmoidLoss(Tensor chosenRewards, Tensor rejectedRewards) {
        Tensor diff = chosenRewards.sub(rejectedRewards);
        Tensor neg = diff.neg();
        Tensor logSigmoid = torch.log_sigmoid(neg);
        Tensor loss = logSigmoid.neg().mean();
        diff.close(); neg.close(); logSigmoid.close();
        return loss;
    }

    public static Tensor flatten(List<Float> floats) {
        int n = floats.size();
        Tensor t = torch.zeros(new long[]{Math.max(1, n)},
                new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
        for (int i = 0; i < n; i++) {
            t.select(0, i).fill_(new Scalar(floats.get(i).doubleValue()));
        }
        return t;
    }

    public static Tensor scoreFromLogits(Tensor logits) {
        return logits.select(1, 1);
    }

    public static Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor chosen = batch.get("input_ids_j") != null ? scoreFromLogits(batch.get("logits_j")) : null;
        Tensor rejected = batch.get("input_ids_k") != null ? scoreFromLogits(batch.get("logits_k")) : null;
        if (chosen == null || rejected == null) {
            return torch.zeros(new long[]{1},
                    new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)));
        }
        return pairwiseLogSigmoidLoss(chosen, rejected);
    }
}
