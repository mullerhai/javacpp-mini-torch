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

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * Huber loss — less sensitive to outliers than MSE.
 *
 * <p>Uses smooth L1 loss: quadratic for |x| &le; delta, linear for |x| &gt; delta.
 * See Huber, "Robust Estimation of a Location Parameter" (1964).
 */
public class HuberLoss implements Loss {

    private final double delta;

    /**
     * Construct a HuberLoss.
     *
     * @param delta the transition point between L2 and L1 loss
     */
    public HuberLoss(double delta) {
        if (delta <= 0) throw new IllegalArgumentException("delta must be positive");
        this.delta = delta;
    }

    @Override
    public Tensor compute(Tensor logits, Tensor labels, Map<String, Object> kwargs) {
        Tensor diff = sub(logits, labels);
        Tensor absDiff = abs(diff);
        // Tensor.pow(Scalar) is the only overload — use the instance form.
        Tensor sqDiff = diff.mul(diff);
        Scalar deltaScalar = new Scalar((float) delta);

        // L2 region: |diff| <= delta
        Tensor l2Region = mul(sqDiff, new Scalar(0.5f));
        // L1 region: |diff| > delta
        // halfDeltaSq = delta * delta * 0.5  — compute as a plain Scalar since
        // there's no mul(Scalar, Scalar) overload.
        Scalar halfDeltaSq = new Scalar(0.5f * (float) delta * (float) delta);
        Tensor l1Region = sub(mul(absDiff, deltaScalar), halfDeltaSq);

        // Select per-element: where(absDiff <= delta, l2Region, l1Region)
        Tensor mask = absDiff.le(deltaScalar).to(diff.scalar_type());
        // (1 - mask) → use neg() of (mask - 1)
        Tensor invMask = sub(mask, new Scalar(1.0f)).neg();
        Tensor loss = add(mul(mask, l2Region), mul(invMask, l1Region));
        return loss.mean();
    }
}
