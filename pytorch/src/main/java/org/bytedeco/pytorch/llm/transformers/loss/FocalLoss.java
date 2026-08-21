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
package org.bytedeco.pytorch.llm.transformers.loss;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Focal loss for dense object detection / segmentation.
 *
 * <p>Reduces the relative loss for well-classified examples, focusing training
 * on hard negatives. See Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017).
 *
 * <pre>
 * FL(p) = -alpha * (1 - p)^gamma * log(p)
 * </pre>
 */
public class FocalLoss implements Loss {

    private final double gamma;
    private final double alpha;

    /**
     * Construct a FocalLoss.
     *
     * @param gamma focusing parameter (higher = more focus on hard examples)
     * @param alpha class balancing weight
     */
    public FocalLoss(double gamma, double alpha) {
        if (gamma < 0) throw new IllegalArgumentException("gamma must be non-negative");
        this.gamma = gamma;
        this.alpha = alpha;
    }

    @Override
    public Tensor compute(Tensor logits, Tensor labels, Map<String, Object> kwargs) {
        // Compute probabilities via softmax
        Tensor probs = softmax(logits, logits.dim() - 1);

        // Get probability for the true class
        Tensor pt = probs.gather(logits.dim() - 1, labels.to(ScalarType.Long), true)
                .clamp(new ScalarOptional(new Scalar(1e-7f)), new ScalarOptional(new Scalar(1.0f - 1e-7f)));

        // Focal weight: (1 - p_t)^gamma  — Tensor.pow(Scalar) is the only overload.
        // sub(Scalar, Tensor) is not available; use rsub (Scalar - Tensor).
        Tensor focalWeight = org.bytedeco.pytorch.global.torch.rsub(pt, new Scalar(1.0f)).pow(new Scalar((float) gamma));

        // Cross-entropy log term
        Tensor ceLoss = nll_loss(neg(log(pt)), labels, null);

        // mul(Scalar, Tensor) not available; use Tensor * Scalar (Tensor.mul(Scalar)).
        return mul(mul(focalWeight, new Scalar((float) alpha)), ceLoss).mean();
    }
}
