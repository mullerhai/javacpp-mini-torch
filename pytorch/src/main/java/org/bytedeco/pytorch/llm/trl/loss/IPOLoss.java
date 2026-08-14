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

import static org.bytedeco.pytorch.global.torch.*;

/**
 * IPO (Identity Preference Optimization) loss function.
 *
 * <p>IPO adds a regularization term to DPO that makes the algorithm provably
 * converge to the optimal policy. The key insight is that the optimal policy
 * satisfies π(y_c) - π(y_r) = 1 under the Bradley-Terry model.
 *
 * <p>The loss is:
 * <pre>
 *   L = E[(π_θ(y_c) - π_θ(y_r) - 1/(2β))²]
 * </pre>
 *
 * <p>This is equivalent to DPO but with a different target and no log-sigmoid.
 * The theoretical advantage is that this formulation has better finite-sample
 * guarantees and provable convergence properties.
 *
 * <p>Reference: "A Theoretical Analysis of IPO" (Azar et al., 2024)
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class IPOLoss {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private IPOLoss() {} // Static utility class

    /**
     * Compute IPO loss.
     *
     * @param policyChosenLogps Log-probs of chosen responses under policy [B]
     * @param policyRejectedLogps Log-probs of rejected responses under policy [B]
     * @param refChosenLogps Log-probs of chosen responses under reference [B]
     * @param refRejectedLogps Log-probs of rejected responses under reference [B]
     * @param beta KL penalty coefficient
     * @return Scalar mean loss
     */
    public static Tensor compute(
            Tensor policyChosenLogps,
            Tensor policyRejectedLogps,
            Tensor refChosenLogps,
            Tensor refRejectedLogps,
            double beta) {

        // Compute log-prob differences
        Tensor piLogratios = policyChosenLogps.sub(policyRejectedLogps);
        Tensor refLogratios = refChosenLogps.sub(refRejectedLogps);

        // Compute advantage (policy - reference)
        Tensor advantage = piLogratios.sub(refLogratios);

        // IPO target: 1/(2β)
        double target = 1.0 / (2.0 * beta);

        // Squared error loss
        Tensor diff = advantage.sub(new Scalar(target));
        return diff.mul(diff).mean();
    }

    /**
     * Compute IPO loss with identity regularization.
     *
     * @param policyChosenLogps Log-probs of chosen responses [B]
     * @param policyRejectedLogps Log-probs of rejected responses [B]
     * @param refChosenLogps Reference chosen log-probs [B]
     * @param refRejectedLogps Reference rejected log-probs [B]
     * @param beta KL penalty coefficient
     * @param identityCoef Coefficient for identity regularization term
     * @return Scalar mean loss
     */
    public static Tensor compute(
            Tensor policyChosenLogps,
            Tensor policyRejectedLogps,
            Tensor refChosenLogps,
            Tensor refRejectedLogps,
            double beta,
            double identityCoef) {

        // Standard IPO loss
        Tensor ipoLoss = compute(policyChosenLogps, policyRejectedLogps,
                                refChosenLogps, refRejectedLogps, beta);

        // Identity regularization: encourages π_θ(y_c) = π_θ(y_r) + 1
        // This pushes the policy log-prob difference toward 1
        Tensor piDiff = policyChosenLogps.sub(policyRejectedLogps);
        Tensor identityTarget = piDiff.sub(new Scalar(1.0));
        Tensor identityLoss = identityTarget.mul(identityTarget).mean();

        // Combine
        return ipoLoss.add(identityLoss.mul(new Scalar(identityCoef)));
    }
}
