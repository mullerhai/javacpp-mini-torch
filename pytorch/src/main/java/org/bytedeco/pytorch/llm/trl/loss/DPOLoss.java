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
package org.bytedeco.pytorch.llm.trl.loss;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.clamp;
import static org.bytedeco.pytorch.global.torch.log;
import static org.bytedeco.pytorch.global.torch.sigmoid;
import static org.bytedeco.pytorch.global.torch.cross_entropy;
import static org.bytedeco.pytorch.global.torch.LongOptional;

/**
 * Direct Preference Optimization loss (Rafailov et al.) and its modern variants.
 *
 * <pre>
 *   π_logratios  = policy_chosen − policy_rejected
 *   ref_logratios = ref_chosen − ref_rejected
 *   logits      = π_logratios − ref_logratios
 *   sigmoid: −log σ(β · logits)
 *   hinge:   relu(1 − β · logits)
 *   ipo:     (logits − 1/(2β))²
 * </pre>
 *
 * Enterprise-grade extensions add: {@code robust}, {@code hinge}, {@code ipo},
 * {@code exo_pair}, {@code nca_pair}, {@code sppo_huber}, {@code sppo_eps},
 * {@code orpo}, {@code apos}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DPOLoss {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private DPOLoss() {}

    /**
     * Default DPO dispatcher.
     */
    public static Tensor compute(
            Tensor policyChosenLogps,
            Tensor policyRejectedLogps,
            Tensor refChosenLogps,
            Tensor refRejectedLogps,
            double beta,
            String lossType) {
        Tensor piLogratios = policyChosenLogps.sub(policyRejectedLogps);
        Tensor refLogratios = refChosenLogps.sub(refRejectedLogps);
        Tensor logits = piLogratios.sub(refLogratios);
        String type = lossType == null ? "sigmoid" : lossType.toLowerCase();
        Tensor losses;
        switch (type) {
            case "hinge": {
                Tensor t = logits.mul(new Scalar(-beta)).add(new Scalar(1.0));
                losses = clamp(t, new ScalarOptional(new Scalar(0.0)), new ScalarOptional(new Scalar(1e12)));
                break;
            }
            case "ipo": {
                double target = 1.0 / (2.0 * Math.max(beta, 1e-8));
                Tensor diff = logits.sub(new Scalar(target));
                losses = diff.mul(diff);
                break;
            }
            case "sigmoid":
            default: {
                Tensor scaled = logits.mul(new Scalar(beta));
                losses = log(sigmoid(scaled)).neg();
                break;
            }
        }
        return losses.mean();
    }

    public static Tensor compute(
            Tensor policyChosenLogps,
            Tensor policyRejectedLogps,
            Tensor refChosenLogps,
            Tensor refRejectedLogps,
            double beta) {
        return compute(policyChosenLogps, policyRejectedLogps,
                refChosenLogps, refRejectedLogps, beta, "sigmoid");
    }

    // -----------------------------------------------------------------------
    // Sigmoid (with label smoothing)
    // -----------------------------------------------------------------------

    /**
     * Standard sigmoid DPO with optional label smoothing.
     * When {@code labelSmoothing > 0}, targets are nudged toward 0.5 to encourage robustness.
     */
    public static Tensor computeSigmoid(Tensor pC, Tensor pR, Tensor rC, Tensor rR,
                                        double beta, double labelSmoothing) {
        Tensor logits = pC.sub(pR).sub(rC.sub(rR)).mul(new Scalar(beta));
        if (labelSmoothing > 0.0) {
            // HF formula: -[ (1-ls) * log σ(z) + ls * log σ(-z) ]
            Tensor pos = sigmoid(logits).log();
            Tensor neg = sigmoid(logits.neg()).log();
            return (pos.mul(new Scalar(-(1.0 - labelSmoothing)))
                    .add(neg.mul(new Scalar(-labelSmoothing)))).mean();
        }
        return logits.sigmoid().log().neg().mean();
    }

    // -----------------------------------------------------------------------
    // Robust DPO
    // -----------------------------------------------------------------------

    /**
     * Robust DPO: regular sigmoid loss + KL regularization against the reference.
     */
    public static Tensor computeRobust(Tensor pC, Tensor pR, Tensor rC, Tensor rR,
                                       double beta, double labelSmoothing) {
        Tensor base = computeSigmoid(pC, pR, rC, rR, beta, labelSmoothing);
        Tensor kl = (rC.sub(pC).mean()).add(rR.sub(pR).mean()).mul(new Scalar(0.5 * beta));
        return base.add(kl);
    }

    // -----------------------------------------------------------------------
    // Hinge
    // -----------------------------------------------------------------------

    public static Tensor computeHinge(Tensor pC, Tensor pR, Tensor rC, Tensor rR, double beta) {
        Tensor logits = pC.sub(pR).sub(rC.sub(rR)).mul(new Scalar(beta));
        Tensor t = logits.mul(new Scalar(-1.0)).add(new Scalar(1.0));
        Tensor clipped = clamp(t, new ScalarOptional(new Scalar(0.0)), new ScalarOptional(new Scalar(1e12)));
        return clipped.mean();
    }

    // -----------------------------------------------------------------------
    // IPO
    // -----------------------------------------------------------------------

    public static Tensor computeIPO(Tensor pC, Tensor pR, Tensor rC, Tensor rR, double beta) {
        Tensor logits = pC.sub(pR).sub(rC.sub(rR));
        double target = 1.0 / (2.0 * Math.max(beta, 1e-8));
        Tensor diff = logits.sub(new Scalar(target));
        return diff.mul(diff).mean();
    }

    // -----------------------------------------------------------------------
    // EXO_Pair (Exact Preference Optimization)
    // -----------------------------------------------------------------------

    /**
     * EXO_Pair uses the exact KL term {@code γ * log π(y|x) - log π_ref(y|x)} averaged
     * with the sigmoid DPO surrogate weighted by (1-γ).
     */
    public static Tensor computeExoPair(Tensor pC, Tensor pR, Tensor rC, Tensor rR,
                                        double beta, double gamma) {
        Tensor klC = rC.sub(pC).mean();
        Tensor klR = rR.sub(pR).mean();
        Tensor kl = klC.add(klR).mul(new Scalar(0.5));
        Tensor sigmoid = computeSigmoid(pC, pR, rC, rR, beta, 0.0);
        return sigmoid.mul(new Scalar(1.0 - gamma)).add(kl.mul(new Scalar(gamma)));
    }

    // -----------------------------------------------------------------------
    // NCA_Pair
    // -----------------------------------------------------------------------

    /**
     * NCA_Pair (Noise Contrastive Alignment) loss: log σ(β · (π_c − π_r − λ (π_c − π_ref_c)²))
     */
    public static Tensor computeNcaPair(Tensor pC, Tensor pR, Tensor rC, Tensor rR, double beta) {
        Tensor cDiff = pC.sub(rC);
        Tensor logits = pC.sub(pR).add(cDiff.mul(cDiff).mul(new Scalar(-0.5))).mul(new Scalar(beta));
        return logits.sigmoid().log().neg().mean();
    }

    // -----------------------------------------------------------------------
    // SPPO / SPO
    // -----------------------------------------------------------------------

    public static Tensor computeSpppoHuber(Tensor pC, Tensor pR, Tensor rC, Tensor rR,
                                           double beta, double labelSmoothing) {
        // SPPO-Huber: β·(π_logratio − ref_logratio) − γ, with huber loss around 0.
        Tensor z = pC.sub(pR).sub(rC.sub(rR)).mul(new Scalar(beta));
        Tensor huberLoss = huber(z, 1.0);
        return huberLoss.mean();
    }

    public static Tensor computeSpppoEps(Tensor pC, Tensor pR, Tensor rC, Tensor rR,
                                         double beta, double labelSmoothing) {
        Tensor z = pC.sub(pR).sub(rC.sub(rR)).mul(new Scalar(beta));
        Tensor sigmoidL = z.sigmoid().log();
        Tensor negSigmoidL = z.neg().sigmoid().log();
        Tensor loss = sigmoidL.mul(new Scalar(-1.0)).sub(negSigmoidL);
        return loss.mean();
    }

    /**
     * Huber loss with smooth-quadratic falloff past {@code delta}.
     * Returns an elementwise tensor (not reduced).
     */
    private static Tensor huber(Tensor input, double delta) {
        Tensor abs = input.abs();
        Tensor quadratic = abs.mul(new Scalar(0.5));
        Tensor linear = abs.mul(new Scalar(delta)).sub(new Scalar(0.5 * delta * delta));
        Tensor cond = abs.lt(new Scalar(delta));
        // torch.where(cond, x_if_true, x_if_false) — using cast to float.
        Tensor condF = cond.to(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        return condF.mul(quadratic).add(condF.mul(new Scalar(-1.0)).add(new Scalar(1.0)).mul(linear));
    }

    // -----------------------------------------------------------------------
    // ORPO
    // -----------------------------------------------------------------------

    /**
     * ORPO loss: {@code -log σ(beta · (logodds_chosen − logodds_rejected)) + log_ratio}.
     * Reference-free by construction.
     */
    public static Tensor computeORPO(Tensor pC, Tensor pR, double beta, boolean lengthNormalize) {
        // log odds = log p / log (1-p). We approximate using log p directly.
        // The HF formulation uses log(π/ (1 − π)). Without explicit probabilities we
        // use the difference of log-probs as a monotonic proxy.
        Tensor oddsDiff = pC.sub(pR).mul(new Scalar(beta));
        Tensor sigmoidLoss = oddsDiff.sigmoid().log().neg();
        // log_ratio term acts as NLL on chosen completion.
        Tensor nll = pC.mul(new Scalar(-1.0)).mean();
        return sigmoidLoss.add(nll);
    }

    // -----------------------------------------------------------------------
    // APOS
    // -----------------------------------------------------------------------

    /**
     * APOS (Adaptive Preference Optimization) — combines sigmoid DPO with adaptive γ-based
     * down-weighting of "easy" examples.
     */
    public static Tensor computeApos(Tensor pC, Tensor pR, Tensor rC, Tensor rR,
                                     double beta, double gamma) {
        Tensor sigmoidLoss = computeSigmoid(pC, pR, rC, rR, beta, 0.0);
        if (gamma <= 0.0) return sigmoidLoss;
        // Adaptive weight: 1 / (1 + exp(-gamma * beta * (p_c − p_r)))
        Tensor z = pC.sub(pR).mul(new Scalar(gamma * beta));
        Tensor weight = z.sigmoid();
        return sigmoidLoss.mul(weight).mean().div(weight.mean().add(new Scalar(1e-8)));
    }

    // -----------------------------------------------------------------------
    // SFT auxiliary loss
    // -----------------------------------------------------------------------

    /**
     * Standard NLL on chosen tokens. Used for the SFT auxiliary term and for
     * "sft" loss type.
     */
    public static Tensor sftNll(Tensor logits, Tensor labels, Tensor mask) {
        // cross_entropy requires Long labels
        Tensor labelLong = labels.to(org.bytedeco.pytorch.global.torch.ScalarType.Long);
        // shift: predict token t+1 from position t
        long t = logits.size(1);
        Tensor shiftLogits = logits.slice(1, LongOptional(0), LongOptional(t - 1), 1);
        Tensor shiftLabels = labelLong.slice(1, LongOptional(1), LongOptional(labelLong.size(1)), 1);
        Tensor ce = cross_entropy(
                shiftLogits.reshape(shiftLogits.size(0) * shiftLogits.size(1), shiftLogits.size(2)),
                shiftLabels.reshape(shiftLabels.size(0) * shiftLabels.size(1)));
        if (mask != null && mask.defined()) {
            Tensor m = mask.to(org.bytedeco.pytorch.global.torch.ScalarType.Float);
            Tensor mShift = m.slice(1, LongOptional(1), LongOptional(m.size(1)), 1).reshape(shiftLabels.size(0) * shiftLabels.size(1));
            Tensor weighted = ce.mul(mShift);
            return weighted.sum().div(mShift.sum().add(new Scalar(1e-8)));
        }
        return ce.mean();
    }

    private static org.bytedeco.pytorch.LongOptional LongOptional(long v) {
        return new org.bytedeco.pytorch.LongOptional(v);
    }
}