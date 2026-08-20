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
import org.bytedeco.pytorch.optim.schedulers.*;

import org.bytedeco.pytorch.optim.Optimizer;

/**
 * Learning-rate scheduler factory mirroring HF's
 * {@code transformers.optimization.get_scheduler}.
 *
 * <p>Creates {@code torch.optim.lr_scheduler.LambdaLR} with a decay lambda
 * based on the scheduler type.
 */
public final class scheduler {

    private scheduler() {}

    /**
     * Build a learning-rate scheduler.
     *
     * @param name               scheduler type (e.g. {@code "linear"}, {@code "cosine"})
     * @param optimizer          PyTorch optimizer
     * @param numWarmupSteps     number of warm-up steps
     * @param numTrainingSteps   total training steps
     * @return a learning-rate scheduler stub (returns null — real implementation pending)
     */
    public static Object get_scheduler(String name, Optimizer optimizer,
                                     int numWarmupSteps, int numTrainingSteps) {
        scheduler_type type = scheduler_type.from(name);
        float warmupRatio = numTrainingSteps > 0
                ? (float) numWarmupSteps / numTrainingSteps
                : 0.1f;

        // Compute the decay lambda inline for simple cases
        switch (type) {
            case LINEAR:
                return makeLinearDecay(numWarmupSteps, numTrainingSteps);
            case COSINE:
                return makeCosineDecay(numWarmupSteps, numTrainingSteps);
            case CONSTANT:
                return (step) -> step < numWarmupSteps ? (float) step / Math.max(1, numWarmupSteps) : 1.0f;
            case CONSTANT_WITH_WARMUP:
                return makeConstantWithWarmup(numWarmupSteps);
            default:
                return makeLinearDecay(numWarmupSteps, numTrainingSteps);
        }
    }

    private static FloatUnaryOperator makeLinearDecay(int numWarmupSteps, int numTrainingSteps) {
        return (step) -> {
            if (step < numWarmupSteps) {
                return (float) step / Math.max(1, numWarmupSteps);
            }
            float decaySteps = numTrainingSteps - numWarmupSteps;
            if (decaySteps <= 0) return 1.0f;
            return Math.max(0.0f, 1.0f - (float) (step - numWarmupSteps) / decaySteps);
        };
    }

    private static FloatUnaryOperator makeCosineDecay(int numWarmupSteps, int numTrainingSteps) {
        return (step) -> {
            if (step < numWarmupSteps) {
                return (float) step / Math.max(1, numWarmupSteps);
            }
            float progress = (float) (step - numWarmupSteps) / Math.max(1, numTrainingSteps - numWarmupSteps);
            return (float) (0.5 * (1.0 + Math.cos(Math.PI * progress)));
        };
    }

    private static FloatUnaryOperator makeConstantWithWarmup(int numWarmupSteps) {
        return (step) -> step < numWarmupSteps ? (float) step / Math.max(1, numWarmupSteps) : 1.0f;
    }

    @FunctionalInterface
    public interface FloatUnaryOperator {
        float applyAsFloat(float operand);
    }
}
