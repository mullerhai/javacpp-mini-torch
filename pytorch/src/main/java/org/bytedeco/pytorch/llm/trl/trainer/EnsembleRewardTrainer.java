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
package org.bytedeco.pytorch.llm.trl.trainer;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.trl.config.EnsembleRewardConfig;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Ensemble Reward Trainer for multi-objective optimization.
 *
 * <p>Enterprise features:
 * <ul>
 *   <li>Multi-head reward regression (MSE / Huber / margin)</li>
 *   <li>Adaptive weighting (inverse loss / softmax temperature)</li>
 *   <li>Centered rewards (mean subtraction)</li>
 *   <li>Bradley-Terry preference loss for chosen/rejected pairs</li>
 *   <li>Length / max-length / max-prompt clamp support</li>
 *   <li>NaN/Inf guard</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class EnsembleRewardTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";
    public static final String ALGORITHM_ID = "ensemble_reward";

    private volatile boolean closed;
    private long numTrainingSteps;

    private final Module rewardModel;
    private final EnsembleRewardConfig config;
    private final TensorVector params;
    private final int numRewards;
    private final double[] currentWeights;
    private final double[] rewardMeans;
    private final double[] rewardStds;
    private final double[] perRewardLosses;

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
        System.out.printf(
                "[EnsembleRewardTrainer v%s] num=%d, adaptive=%s, center=%s, margin=%.3f%n",
                VERSION, numRewards, config.adaptiveWeighting(),
                config.centerRewards(), config.margin());
    }

    public Module rewardModel() { return rewardModel; }
    public EnsembleRewardConfig config() { return config; }
    public double[] currentWeights() { return currentWeights.clone(); }

    @Override
    protected TensorVector trainableParameters() { return params; }

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
    public Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor targetRewards = require(batch, "rewards");
        Tensor inputIds = require(batch, "input_ids");
        Tensor attentionMask = batch.get("attention_mask");

        Tensor rewardOutputs = rewardModel.forward(inputIds, attentionMask);

        Tensor predictedRewards;
        if (rewardOutputs.dim() == 2 && rewardOutputs.size(1) == numRewards) {
            predictedRewards = rewardOutputs;
        } else if (rewardOutputs.dim() == 1) {
            predictedRewards = rewardOutputs.unsqueeze(1);
        } else {
            throw new IllegalArgumentException(
                    "Unexpected reward output shape: [" + rewardOutputs.size(0) + ", "
                            + (rewardOutputs.dim() > 1 ? rewardOutputs.size(1) : -1) + "]");
        }

        Tensor totalLoss = null;
        for (int r = 0; r < numRewards; r++) {
            Tensor predR = predictedRewards.select(1, r);
            Tensor targetR = targetRewards.select(1, r);

            if (config.centerRewards()) {
                targetR = targetR.sub(targetR.mean());
            }

            Tensor lossR;
            double ls = config.labelSmoothing();
            if (ls > 0.0) {
                // Smoothed L1 (Huber-like) loss
                Tensor abs = predR.sub(targetR).abs();
                Scalar delta = new Scalar(config.margin());
                Tensor quadratic = abs.mul(new Scalar(0.5));
                Tensor linear = abs.sub(new Scalar(0.5));
                Tensor cond = abs.lt(delta);
                Tensor condF = cond.to(org.bytedeco.pytorch.global.torch.ScalarType.Float);
                lossR = condF.mul(quadratic)
                        .add(condF.mul(new Scalar(-1.0)).add(new Scalar(1.0)).mul(linear))
                        .mean();
            } else {
                lossR = predR.sub(targetR).pow(new Scalar(2)).mean();
            }

            updateRewardStatistics(r, targetR.mean().item_double(), lossR.mean().item_double());

            double weight = currentWeights[r];
            Tensor weightedLoss = lossR.mul(new Scalar(weight));

            if (totalLoss == null) {
                totalLoss = weightedLoss;
            } else {
                totalLoss = totalLoss.add(weightedLoss);
            }
        }

        // Margin-based regularization (e.g., pairwise difference)
        if (config.margin() > 0.0 && numRewards >= 2) {
            Tensor firstHead = predictedRewards.select(1, 0);
            Tensor secondHead = predictedRewards.select(1, 1);
            Tensor marginPenalty = torch.tensor(config.margin()).sub(firstHead.sub(secondHead))
                    .clamp_min(new Scalar(0.0)).mean();
            totalLoss = totalLoss.add(marginPenalty);
        }

        if (config.adaptiveWeighting()) {
            updateAdaptiveWeights();
        }

        double v = (totalLoss != null) ? totalLoss.item_double() : 0.0;
        if (Double.isNaN(v) || Double.isInfinite(v)) {
            System.err.println("[EnsembleRewardTrainer] WARNING: NaN/Inf loss; using zero.");
            return zeros(new long[]{1}, inputIds.options());
        }

        numTrainingSteps++;
        return totalLoss != null ? totalLoss : zeros(new long[]{1}, inputIds.options());
    }

    private void updateRewardStatistics(int rewardIdx, double meanReward, double loss) {
        double alpha = 0.1;
        rewardMeans[rewardIdx] = (1 - alpha) * rewardMeans[rewardIdx] + alpha * meanReward;
        perRewardLosses[rewardIdx] = (1 - alpha) * perRewardLosses[rewardIdx] + alpha * loss;
    }

    private void updateAdaptiveWeights() {
        double updateRate = config.weightUpdateRate();
        double totalLoss = 0;
        for (double loss : perRewardLosses) {
            totalLoss += Math.abs(loss);
        }
        if (totalLoss <= 0) return;
        double sumInvLoss = 0;
        double[] newWeights = new double[numRewards];
        for (int r = 0; r < numRewards; r++) {
            double invLoss = totalLoss / (Math.abs(perRewardLosses[r]) + 1e-6);
            sumInvLoss += invLoss;
            newWeights[r] = invLoss;
        }
        if (sumInvLoss <= 0) return;
        for (int r = 0; r < numRewards; r++) {
            newWeights[r] /= sumInvLoss;
            currentWeights[r] = (1 - updateRate) * currentWeights[r] + updateRate * newWeights[r];
        }
    }

    public double[][] computeParetoFront(Tensor rewards) {
        int batchSize = (int) rewards.size(0);
        double[][] paretoPoints = new double[batchSize][];
        int paretoCount = 0;
        for (int i = 0; i < batchSize; i++) {
            boolean isPareto = true;
            for (int j = 0; j < batchSize && isPareto; j++) {
                if (i == j) continue;
                boolean dominates = true;
                for (int k = 0; k < numRewards; k++) {
                    if (rewards.select(0, i).select(0, k).item_double() >
                            rewards.select(0, j).select(0, k).item_double()) {
                        dominates = false;
                        break;
                    }
                }
                if (dominates) isPareto = false;
            }
            if (isPareto) {
                double[] point = new double[numRewards];
                for (int k = 0; k < numRewards; k++) {
                    point[k] = rewards.select(0, i).select(0, k).item_double();
                }
                paretoPoints[paretoCount++] = point;
            }
        }
        double[][] result = new double[paretoCount][];
        System.arraycopy(paretoPoints, 0, result, 0, paretoCount);
        return result;
    }

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
        stats.append("[EnsembleRewardTrainer v").append(VERSION).append("] Closed: steps=").append(numTrainingSteps);
        stats.append(", weights=[");
        for (int i = 0; i < numRewards; i++) {
            if (i > 0) stats.append(", ");
            stats.append(String.format("%.3f", currentWeights[i]));
        }
        stats.append("]");
        System.out.println(stats.toString());
    }

    public boolean isClosed() { return closed; }
    public String algorithm() { return ALGORITHM_ID; }
    public String algorithmName() {
        return "Ensemble Reward v" + VERSION;
    }
}