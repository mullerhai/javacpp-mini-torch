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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trl.loss;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * RRHF (Rank Responses to Rank Responses) loss function.
 *
 * <p>RRHF aligns language models using a ranking loss on generated responses.
 * The key insight is that we want to match the ranking induced by the reward model.
 *
 * <p>The loss has two components:
 * <ol>
 *   <li><b>Reward loss</b>: Push model to assign higher probabilities to
 *       higher-ranked responses</li>
 *   <li><b>Ratio loss</b>: KL penalty to keep policy close to reference</li>
 * </ol>
 *
 * <p>Reward loss (sample-level):
 * <pre>
 *   L_reward = -E[sum_i π(rank_i) * log π(y_i)]
 * </pre>
 * where π(rank_i) is a probability distribution over ranks (e.g., softmax)
 *
 * <p>Ratio loss:
 * <pre>
 *   L_ratio = KL(π_θ || π_ref)
 * </pre>
 *
 * <p>Reference: "RRHF: Rank Responses to Rank Responses for Human Preference"
 * (Yuan et al., 2023)
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class RRHFLoss {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private RRHFLoss() {} // Static utility class

    /**
     * Compute RRHF loss with sample-level ranking.
     *
     * @param logProbs Log-probs for each response [B * num_responses]
     * @param rewards Rewards/scores for each response [B * num_responses]
     * @param numResponses Number of responses per prompt
     * @param temperature Temperature for softmax over rewards
     * @param rewardWeight Weight for reward loss
     * @param ratioWeight Weight for ratio loss (requires refLogProbs)
     * @return Scalar mean loss
     */
    public static Tensor computeSampleLevel(
            Tensor logProbs,
            Tensor rewards,
            int numResponses,
            double temperature,
            double rewardWeight,
            double ratioWeight) {

        int batchSize = (int) (logProbs.size(0) / numResponses);

        Tensor totalLoss = zeros(new long[]{}, logProbs.options());

        for (int i = 0; i < batchSize; i++) {
            // Get responses for this prompt
            Tensor respLp = logProbs.narrow(0, i * numResponses, numResponses);
            Tensor respRewards = rewards.narrow(0, i * numResponses, numResponses);

            // Softmax over rewards to get target distribution
            Tensor rewardDist = softmax(respRewards.div(new Scalar(temperature)), 0);

            // Reward loss: weighted cross-entropy
            Tensor rewardLoss = respLp.mul(rewardDist).neg().sum();

            // Accumulate
            totalLoss = totalLoss.add(rewardLoss);
        }

        return totalLoss.div(new Scalar(batchSize));
    }

    /**
     * Compute RRHF loss with pairwise ranking.
     *
     * @param logProbs Log-probs for each response [B * num_responses]
     * @param rewards Rewards for each response [B * num_responses]
     * @param numResponses Number of responses per prompt
     * @param rewardWeight Weight for reward loss
     * @param ratioWeight Weight for ratio loss
     * @return Scalar mean loss
     */
    public static Tensor computePairwise(
            Tensor logProbs,
            Tensor rewards,
            int numResponses,
            double rewardWeight,
            double ratioWeight) {

        int batchSize = (int) (logProbs.size(0) / numResponses);

        Tensor totalLoss = zeros(new long[]{}, logProbs.options());

        for (int i = 0; i < batchSize; i++) {
            // Get responses for this prompt
            Tensor respLp = logProbs.narrow(0, i * numResponses, numResponses);
            Tensor respRewards = rewards.narrow(0, i * numResponses, numResponses);

            // Compute pairwise losses for all pairs
            Tensor pairwiseLoss = zeros(new long[]{}, logProbs.options());

            for (int j = 0; j < numResponses; j++) {
                for (int k = j + 1; k < numResponses; k++) {
                    // If response j is ranked higher than response k
                    // Use index_select and item() for scalar comparison
                    double rewardJ = respRewards.index_select(0, torch.tensor(new long[]{j}).to(rewards.device(),rewards.dtype().toScalarType())).item_double();
                    double rewardK = respRewards.index_select(0, torch.tensor(new long[]{k}).to(rewards.device(),rewards.dtype().toScalarType())).item_double();

                    if (rewardJ > rewardK) {
                        // Compute margin loss: max(0, -log π(y_j) + log π(y_k))
                        Tensor lpJ = respLp.index_select(0, torch.tensor(new long[]{j}).to(logProbs.device(),logProbs.dtype().toScalarType()));
                        Tensor lpK = respLp.index_select(0, torch.tensor(new long[]{k}).to(logProbs.device(),logProbs.dtype().toScalarType()));
                        Tensor margin = lpK.sub(lpJ);
                        Tensor loss = relu(margin.add(new Scalar(Math.log(numResponses))));
                        pairwiseLoss = pairwiseLoss.add(loss);
                    }
                }
            }

            totalLoss = totalLoss.add(pairwiseLoss);
        }

        return totalLoss.div(new Scalar(batchSize));
    }

    /**
     * Compute combined RRHF loss with ratio penalty.
     *
     * @param logProbs Policy log-probs [B]
     * @param refLogProbs Reference log-probs [B]
     * @param rewards Rewards/scores [B]
     * @param numResponses Number of responses per prompt
     * @param temperature Temperature for softmax
     * @param rewardWeight Weight for reward loss
     * @param ratioWeight Weight for ratio loss
     * @return Scalar mean loss
     */
    public static Tensor compute(
            Tensor logProbs,
            Tensor refLogProbs,
            Tensor rewards,
            int numResponses,
            double temperature,
            double rewardWeight,
            double ratioWeight) {

        // Reward loss
        Tensor rewardLoss;
        if (rewardWeight > 0) {
            rewardLoss = computeSampleLevel(
                    logProbs, rewards, numResponses, temperature, rewardWeight, 0);
        } else {
            rewardLoss = zeros(new long[]{}, logProbs.options());
        }

        // Ratio loss (KL to reference)
        Tensor ratioLoss;
        if (ratioWeight > 0 && refLogProbs != null && refLogProbs.defined()) {
            Tensor ratio = logProbs.sub(refLogProbs);
            if (ratio.dim() > 0) {
                ratioLoss = ratio.mean();
            } else {
                ratioLoss = ratio;
            }
            ratioLoss = ratioLoss.mul(new Scalar(ratioWeight));
        } else {
            ratioLoss = zeros(new long[]{}, logProbs.options());
        }

        return rewardLoss.add(ratioLoss);
    }
}
