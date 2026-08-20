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
package org.bytedeco.pytorch.llm.peft;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import static org.bytedeco.pytorch.global.torch.ones;

/**
 * IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations).
 *
 * <p>HuggingFace: attention targets scale the <em>output</em> of the linear;
 * feedforward targets scale the <em>input</em>. Vectors initialise to 1 so the
 * untrained adapter is an identity.
 *
 * <pre>
 *   feedforward:  y = base(l ⊙ x)
 *   attention:    y = l ⊙ base(x)
 * </pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class IA3Linear extends Module implements AutoCloseable {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final LinearImpl base;
    private final IA3Config config;
    private final boolean feedforward;
    private Tensor scale;
    private volatile boolean closed;

    public static IA3Linear borrowBase(LinearImpl base, IA3Config config, boolean feedforward) {
        return new IA3Linear(base, config, feedforward, false);
    }

    public IA3Linear(LinearImpl base, IA3Config config, boolean feedforward) {
        this(base, config, feedforward, true);
    }

    private IA3Linear(LinearImpl base, IA3Config config, boolean feedforward, boolean registerBase) {
        super("IA3Linear");
        if (base == null) throw new IllegalArgumentException("base must not be null");
        if (config == null) throw new IllegalArgumentException("config must not be null");
        this.base = registerBase ? register_module("base", base) : base;
        this.config = config;
        this.feedforward = feedforward;
        long dim = feedforward ? base.weight().size(1) : base.weight().size(0);
        Tensor init = ones(new long[]{dim}).contiguous().clone();
        init.requires_grad_(true);
        register_parameter("ia3_l", init, true);
        this.scale = init;
        try {
            base.weight().requires_grad_(false);
            if (base.bias() != null && !base.bias().isNull() && base.bias().defined()) {
                base.bias().requires_grad_(false);
            }
        } catch (Exception ignored) {}
    }

    public LinearImpl base() { return base; }
    public IA3Config config() { return config; }
    public boolean isFeedforward() { return feedforward; }
    public Tensor scale() { return scale; }

    public Tensor forward(Tensor input) {
        if (feedforward) {
            return base.forward(input.mul(broadcast(scale, input)));
        }
        Tensor y = base.forward(input);
        return y.mul(broadcast(scale, y));
    }

    /** Align a 1-D scale vector with the last dim of {@code ref}. */
    private static Tensor broadcast(Tensor scale, Tensor ref) {
        long[] shape = new long[(int) ref.dim()];
        java.util.Arrays.fill(shape, 1L);
        shape[shape.length - 1] = scale.numel();
        return scale.reshape(shape);
    }

    public TensorVector ia3Parameters() {
        TensorVector v = new TensorVector();
        v.push_back(scale);
        return v;
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (scale != null && scale.defined()) {
            try { scale.close(); } catch (Throwable ignored) {}
            scale = null;
        }
    }
}
