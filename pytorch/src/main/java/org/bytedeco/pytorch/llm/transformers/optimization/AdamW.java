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

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.optim.options.AdamWOptions;

import java.util.List;
import java.util.Objects;

/**
 * AdamW optimizer wrapper bridging JavaCPP-generated
 * {@link org.bytedeco.pytorch.optim.AdamW} with a HuggingFace-style constructor.
 *
 * <p>Mirrors the API expected by HuggingFace {@code Trainer}.
 */
public class AdamW {
    private final org.bytedeco.pytorch.optim.AdamW optimizer;

    /**
     * Construct an AdamW optimizer over the given parameter list.
     *
     * @param params       the parameters to optimize
     * @param lr           learning rate
     * @param weightDecay  weight decay coefficient
     * @param beta1        first-moment decay rate
     * @param beta2        second-moment decay rate
     * @param eps          epsilon for numerical stability
     */
    public AdamW(List<Tensor> params, double lr, double weightDecay,
                 double beta1, double beta2, double eps) {
        Objects.requireNonNull(params, "params");
        TensorVector tv = new TensorVector();
        for (Tensor t : params) tv.add(t);
        AdamWOptions opts = new AdamWOptions(lr);
        opts.weight_decay().put(weightDecay);
        DoublePointer betas = new DoublePointer(2);
        betas.put(0, beta1);
        betas.put(1, beta2);
        opts.betas(betas);
        opts.eps().put(eps);
        this.optimizer = new org.bytedeco.pytorch.optim.AdamW(tv, opts);
    }

    /** Perform a single optimization step. */
    public void step() {
        optimizer.step();
    }

    /** Clear accumulated gradients for all parameters. */
    public void zero_grad() {
        optimizer.zero_grad();
    }

    /** Zero out gradients for a specific set of parameters. */
    public void zero_grad(boolean setToNone) {
        optimizer.zero_grad(setToNone);
    }

    /** Access the underlying JavaCPP optimizer. */
    public org.bytedeco.pytorch.optim.AdamW unwrap() {
        return optimizer;
    }
}
