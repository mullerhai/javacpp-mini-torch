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
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trl;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptional;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.trl.config.EnsembleRewardConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

/**
 * Ensemble Reward Trainer for multi-objective optimization.
 *
 * <p>Trains multiple reward models simultaneously to enable Pareto-optimal
 * policy learning across competing objectives (e.g., helpfulness vs. safety).
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code input_ids} - text tokens {@code [B, T]}</li>
 *   <li>{@code attention_mask} - attention mask {@code [B, T]}</li>
 *   <li>{@code chosen_input_ids}, {@code rejected_input_ids} (preference data)</li>
 *   <li>{@code rewards} - rewards for each objective {@code [B, K]}</li>
 * </ul>
 */
public final class EnsembleRewardTrainer extends BaseTrainer {
    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module rewardModel;
    private final EnsembleRewardConfig config;
    private final TensorVector params;
    private final int numRewards;
    private final double[] currentWeights;
    private final double[] rewardMeans;
    private final double[] rewardStds;

    // Training statistics
    private double[] perRewardLosses;

    public EnsembleRewardTrainer(
            Module rewardModel,
            Optimizer optimizer,
            EnsembleRewardConfig config) {
        super(config, optimizer);
        this.rewardModel = Objects.requireNonNull(rewardModel, "rewardModel");
        this.config = Objects.requireNonNull(config, "config");
        this.params = rewardModel.parameters();
        this.numRewards = config.numRewards();
        this.currentWeights = config.initialWeights().clone();
        this.rewardMeans = new double[numRewards];
        this.rewardStds = new double[numRewards];
        this.perRewardLosses = new double[numRewards];
    }

    public Module rewardModel() { return rewardModel; }
    public EnsembleRewardConfig config() { return config; }
    public double[] currentWeights() { return currentWeights.clone(); }

    @Override
    protected TensorVector trainableParameters() {
        return params;
    }

    @Override
    public void train() {
        super.train();
        rewardModel.train(true);
    }

    @Override
    public void eval() {
        super.eval();
        rewardModel.eval();
    }

    @Override
    protected Tensor computeLoss(Map<String, Tensor> batch) {
        // Extract rewards
        Tensor targetRewards = require(batch, "rewards");
        int batchSize = (int) targetRewards.size(0);

        // Get reward model outputs
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");

        Tensor rewardOutputs = rewardModel.forward(inputIds, attentionMask);

        // Handle different output shapes
        Tensor predictedRewards;
        if (rewardOutputs.dim() == 2 && rewardOutputs.size(1) == numRewards) {
            predictedRewards = rewardOutputs;
        } else if (rewardOutputs.dim() == 1) {
            // Single scalar per sample, need to expand
            predictedRewards = rewardOutputs.unsqueeze(1);
        } else {
            throw new IllegalArgumentException(
                    "Expected reward model output shape [B, " + numRewards + "] or [B], got " +
                    "[" + rewardOutputs.size(0) + ", " + rewardOutputs.size(1) + "]");
        }

        // Compute loss for each reward head
        Tensor totalLoss = null;

        for (int r = 0; r < numRewards; r++) {
            Tensor predR = predictedRewards.select(1, r);
            Tensor targetR = targetRewards.select(1, r);

            // MSE loss for each reward
            Tensor lossR = predR.sub(targetR).pow(new Scalar(2)).mean();

            // Update running statistics
            updateRewardStatistics(r, targetR.mean().item_double(), lossR.mean().item_double());

            // Weight the loss
            double weight = currentWeights[r];
            Tensor weightedLoss = lossR.mul(new Scalar(weight));

            if (totalLoss == null) {
                totalLoss = weightedLoss;
            } else {
                totalLoss = totalLoss.add(weightedLoss);
            }
        }

        // Update adaptive weights if enabled
        if (config.adaptiveWeighting()) {
            updateAdaptiveWeights();
        }

        numTrainingSteps++;
        return totalLoss != null ? totalLoss : org.bytedeco.pytorch.global.torch.zeros(1);
    }

    /**
     * Preference loss variant for Bradley-Terry preference modeling.
     */
    protected Tensor computePreferenceLoss(Map<String, Tensor> batch, Tensor predictedRewards) {
        // Extract chosen and rejected
        Tensor chosenRewards = predictedRewards.select(1, 0);  // First reward head
        Tensor rejectedRewards = predictedRewards.select(1, 1);  // Second reward head

        // Bradley-Terry preference probability
        Tensor prefLogits = chosenRewards.sub(rejectedRewards);
        Tensor prefProb = org.bytedeco.pytorch.global.torch.sigmoid(prefLogits);

        // Binary cross-entropy with target preference
        // If chosen reward > rejected reward, target = 1, else 0
        Tensor target = org.bytedeco.pytorch.global.torch.where(
                prefLogits.gt(new Scalar(0)),
                prefLogits.mul(new Scalar(0)).add(new Scalar(1)),
                prefLogits.mul(new Scalar(0)));

        Tensor bceLoss = org.bytedeco.pytorch.global.torch.binary_cross_entropy(
                prefProb, target, new TensorOptional(new Scalar(1.0)),
                torch.Reduction.Mean.value);

        return bceLoss;
    }

    private void updateRewardStatistics(int rewardIdx, double meanReward, double loss) {
        // Exponential moving average
        double alpha = 0.1;
        rewardMeans[rewardIdx] = (1 - alpha) * rewardMeans[rewardIdx] + alpha * meanReward;
        perRewardLosses[rewardIdx] = (1 - alpha) * perRewardLosses[rewardIdx] + alpha * loss;
    }

    private void updateAdaptiveWeights() {
        double updateRate = config.weightUpdateRate();

        // Compute gradient-based weight adjustments
        // Increase weight for rewards with higher loss (harder to optimize)
        double totalLoss = 0;
        for (double loss : perRewardLosses) {
            totalLoss += Math.abs(loss);
        }

        if (totalLoss > 0) {
            double[] newWeights = new double[numRewards];
            double sumInvLoss = 0;

            for (int r = 0; r < numRewards; r++) {
                // Weight inversely proportional to loss
                double invLoss = totalLoss / (Math.abs(perRewardLosses[r]) + 1e-6);
                sumInvLoss += invLoss;
                newWeights[r] = invLoss;
            }

            // Normalize
            for (int r = 0; r < numRewards; r++) {
                newWeights[r] /= sumInvLoss;
            }

            // Smooth update
            for (int r = 0; r < numRewards; r++) {
                currentWeights[r] = (1 - updateRate) * currentWeights[r] +
                                   updateRate * newWeights[r];
            }
        }
    }

    /**
     * Compute Pareto-optimal front from current rewards.
     */
    public double[][] computeParetoFront(Tensor rewards) {
        int batchSize = (int) rewards.size(0);
        double[][] paretoPoints = new double[batchSize][];
        int paretoCount = 0;

        for (int i = 0; i < batchSize; i++) {
            boolean isPareto = true;
            for (int j = 0; j < batchSize && isPareto; j++) {
                if (i != j) {
                    boolean dominates = true;
                    for (int k = 0; k < numRewards; k++) {
                        if (rewards.select(0, i).select(0, k).item_double() >
                            rewards.select(0, j).select(0, k).item_double()) {
                            dominates = false;
                            break;
                        }
                    }
                    if (dominates) {
                        isPareto = false;
                    }
                }
            }

            if (isPareto) {
                double[] point = new double[numRewards];
                for (int k = 0; k < numRewards; k++) {
                    point[k] = rewards.select(0, i).select(0, k).item_double();
                }
                paretoPoints[paretoCount++] = point;
            }
        }

        // Trim array
        double[][] result = new double[paretoCount][];
        System.arraycopy(paretoPoints, 0, result, 0, paretoCount);
        return result;
    }

    /**
     * Get per-reward statistics.
     */
    public double[] getRewardMeans() { return rewardMeans.clone(); }
    public double[] getPerRewardLosses() { return perRewardLosses.clone(); }

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

        StringBuilder stats = new StringBuilder();
        stats.append("[EnsembleRewardTrainer] Closed: steps=").append(numTrainingSteps);
        stats.append(", weights=[");
        for (int i = 0; i < numRewards; i++) {
            if (i > 0) stats.append(", ");
            stats.append(String.format("%.3f", currentWeights[i]));
        }
        stats.append("]");
        System.out.println(stats.toString());
    }

    public boolean isClosed() { return closed; }
}
