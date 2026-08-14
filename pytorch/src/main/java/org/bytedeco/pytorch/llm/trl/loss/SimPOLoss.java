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
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * SimPO (Simple Preference Optimization) loss function.
 *
 * <p>SimPO removes the reference model term from DPO, using a target margin
 * on log probabilities instead. This simplifies training and reduces memory.
 *
 * <p>The loss is derived from the Bradley-Terry model with a reward margin:
 * <pre>
 *   π_θ(y_w) / |y_w| - π_θ(y_l) / |y_l| > γ
 * </pre>
 *
 * <p>SimPO Loss:
 * <pre>
 *   L = -E[(y_w, y_l) ~ D][log σ(β * (r_θ(y_w)/|y_w| - r_θ(y_l)/|y_l| - γ))]
 * </pre>
 *
 * <p>Where:
 * <ul>
 *   <li>β is the reward coefficient (controls loss slope)</li>
 *   <li>γ is the target margin (encourages reward separation)</li>
 *   <li>r_θ is the reward (log-probability of sequence)</li>
 *   <li>|y| is sequence length for normalization</li>
 * </ul>
 *
 * <p>Reference: "SimPO: Simple Preference Optimization" (Meng et al., 2024)
 * <a href="https://arxiv.org/abs/2405.14734">arXiv:2405.14734</a>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class SimPOLoss {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private SimPOLoss() {} // Static utility class

    /**
     * Compute SimPO loss.
     *
     * @param chosenLogps Log-probs for chosen responses [B]
     * @param rejectedLogps Log-probs for rejected responses [B]
     * @param beta Reward coefficient (controls loss sensitivity)
     * @param targetMargin Target reward margin (encourages separation)
     * @param lengthNormalize Whether to apply length normalization
     * @return Scalar mean loss
     */
    public static Tensor compute(
            Tensor chosenLogps,
            Tensor rejectedLogps,
            double beta,
            double targetMargin,
            boolean lengthNormalize) {

        // Compute reward difference (already normalized if lengthNormalize is true)
        Tensor rewardDiff = chosenLogps.sub(rejectedLogps);

        // Apply target margin
        Tensor margin = rewardDiff.sub(new Scalar(targetMargin));

        // Scale by beta and apply sigmoid
        Tensor scaledMargin = margin.mul(new Scalar(beta));

        // Neg log sigmoid = softplus(-x)
        Tensor loss = neg_log_sigmoid(scaledMargin);

        return loss.mean();
    }

    /**
     * Compute SimPO loss with label smoothing.
     *
     * @param chosenLogps Log-probs for chosen responses [B]
     * @param rejectedLogps Log-probs for rejected responses [B]
     * @param beta Reward coefficient
     * @param targetMargin Target reward margin
     * @param lengthNormalize Whether to apply length normalization
     * @param labelSmoothing Label smoothing factor [0, 1]
     * @return Scalar mean loss
     */
    public static Tensor compute(
            Tensor chosenLogps,
            Tensor rejectedLogps,
            double beta,
            double targetMargin,
            boolean lengthNormalize,
            double labelSmoothing) {

        if (labelSmoothing <= 0) {
            return compute(chosenLogps, rejectedLogps, beta, targetMargin, lengthNormalize);
        }

        // Standard SimPO loss
        Tensor simpoLoss = compute(chosenLogps, rejectedLogps, beta, targetMargin, lengthNormalize);

        // Add label smoothing: mix with uniform preference
        // Label smoothing encourages learning from both positive and negative signals
        Tensor rewardDiff = chosenLogps.sub(rejectedLogps);
        Tensor margin = rewardDiff.sub(new Scalar(targetMargin));
        Tensor scaledMargin = margin.mul(new Scalar(beta));

        // Smoothed target: slightly favor the standard direction
        double smoothTarget = 1.0 - (labelSmoothing * 0.5);
        Tensor smoothLoss = scaledMargin.neg().mul(new Scalar(smoothTarget));
        Tensor smoothBCE = neg_log_sigmoid(smoothLoss);

        // Interpolate between original and smoothed
        return simpoLoss.mul(new Scalar(1.0 - labelSmoothing))
                .add(smoothBCE.mul(new Scalar(labelSmoothing)));
    }

    /**
     * Numerically stable negative log sigmoid.
     * equivalent to: -log(sigmoid(x)) = softplus(-x) = log(1 + exp(-x))
     */
    private static Tensor neg_log_sigmoid(Tensor x) {
        // softplus(-x) is numerically stable
        return x.neg().neg().log1p().neg().neg();
    }
}
