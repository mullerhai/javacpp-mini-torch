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
package org.bytedeco.pytorch.llm.transformers.trainer.utils;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.transformers.trainer.TrainingArguments;

/**
 * Factory for PyTorch optimizers mirroring HF's {@code transformers.optimization}.
 *
 * <p>Creates AdamW, Adam, SGD, and Adafactor optimizers from model parameters.
 */
public final class optimizers {

    private optimizers() {}

    /**
     * Create an optimizer for the given model based on {@code args.optim}.
     *
     * @param model the model whose parameters to optimise
     * @param args  training arguments containing learning rate, weight decay, optim type
     * @return a PyTorch optimizer (or null if wiring is not yet complete)
     */
    public static org.bytedeco.pytorch.optim.Optimizer create_optimizer(Module model, TrainingArguments args) {
        optimizer_types type = optimizer_types.from(args.optim());
        return switch (type) {
            case ADAM   -> create_adam(model, args.learningRate());
            case SGD    -> create_sgd(model, args.learningRate());
            case ADAFACTOR -> create_adafactor(model, args.learningRate());
            default     -> create_adamw(model, args.learningRate(), args.weightDecay());
        };
    }

    /**
     * Create AdamW optimizer: {@code torch.optim.AdamW(model.parameters(), lr, weight_decay)}.
     */
    public static org.bytedeco.pytorch.optim.Optimizer create_adamw(Module model, float lr, float weightDecay) {
        // TODO: wire torch.optim.AdamW once the PyTorch Java API is available
        System.out.println("[optimizers] AdamW not yet wired — returning null");
        return null;
    }

    /**
     * Create Adam optimizer: {@code torch.optim.Adam(model.parameters(), lr)}.
     */
    public static org.bytedeco.pytorch.optim.Optimizer create_adam(Module model, float lr) {
        // TODO: wire torch.optim.Adam once the PyTorch Java API is available
        return null;
    }

    /**
     * Create SGD optimizer: {@code torch.optim.SGD(model.parameters(), lr)}.
     */
    public static org.bytedeco.pytorch.optim.Optimizer create_sgd(Module model, float lr) {
        // TODO: wire torch.optim.SGD
        return null;
    }

    /**
     * Create Adafactor optimizer (sharded, memory-efficient).
     */
    public static org.bytedeco.pytorch.optim.Optimizer create_adafactor(Module model, float lr) {
        // TODO: wire Adafactor (requires transformers-specific LR schedule)
        return null;
    }
}
