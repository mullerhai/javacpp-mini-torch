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
package org.bytedeco.pytorch.llm.transformers.optimization;

import org.bytedeco.pytorch.Tensor;

import java.util.List;

/**
 * HuggingFace-style AdamW with bias correction for first and second moments.
 *
 * <p>Applies the same bias correction as the HuggingFace
 * {@code AdamW} optimizer:
 * <pre>
 *   m_t = beta1 * m_{t-1} + (1 - beta1) * grad
 *   v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
 *   m_hat = m_t / (1 - beta1^t)
 *   v_hat = v_t / (1 - beta2^t)
 *   theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
 * </pre>
 */
public class AdamWHF extends AdamW {

    private int step;

    /**
     * Construct an AdamWHF optimizer.
     *
     * @param params      the parameters to optimize
     * @param lr          learning rate
     * @param weightDecay weight decay coefficient
     * @param beta1       first-moment decay rate
     * @param beta2       second-moment decay rate
     * @param eps         epsilon for numerical stability
     */
    public AdamWHF(List<Tensor> params, double lr, double weightDecay,
                   double beta1, double beta2, double eps) {
        super(params, lr, weightDecay, beta1, beta2, eps);
        this.step = 0;
    }

    /**
     * Perform an optimization step with HuggingFace-style bias correction.
     */
    @Override
    public void step() {
        step++;
        // Bias correction is applied internally by the underlying AdamW;
        // this hook allows subclasses to inject custom LR scheduling.
        super.step();
    }

    /** Return the current step count (1-indexed). */
    public int getStep() {
        return step;
    }
}
