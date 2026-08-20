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

import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Objects;
import java.util.function.Function;

/**
 * Factory for HuggingFace-style learning rate schedulers.
 *
 * <p>Given a scheduler name, warmup steps, and total training steps,
 * returns a {@link Function} mapping a current step (Long) to a learning rate
 * multiplier (Double).
 *
 * <p>Supported names: {@code linear}, {@code cosine}, {@code cosine_with_restarts},
 * {@code polynomial}, {@code constant}, {@code constant_with_warmup}.
 */
public final class get_scheduler {

    private get_scheduler() {} // static utility

    /**
     * Build a LR scheduler lambda for the given name.
     *
     * @param name              scheduler name (case-insensitive)
     * @param optimizer         the optimizer whose LR will be adjusted (not modified)
     * @param numWarmupSteps    number of warmup steps
     * @param numTrainingSteps  total number of training steps
     * @return a function {@code step -> lrMultiplier}
     */
    public static Lambda get(String name, Optimizer optimizer,
                             int numWarmupSteps, int numTrainingSteps) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(optimizer, "optimizer");
        if (numTrainingSteps <= 0) {
            throw new IllegalArgumentException("numTrainingSteps must be positive");
        }

        String lower = name.trim().toLowerCase(java.util.Locale.ROOT);
        switch (lower) {
            case "linear":
                return new LinearScheduler(numWarmupSteps, numTrainingSteps);
            case "cosine":
                return new CosineScheduler(numWarmupSteps, numTrainingSteps, 1.0, 0.0);
            case "cosine_with_restarts":
                return new CosineWithRestartsScheduler(numWarmupSteps, numTrainingSteps);
            case "polynomial":
                return new PolynomialScheduler(numWarmupSteps, numTrainingSteps);
            case "constant":
                return new ConstantScheduler(numWarmupSteps, 1.0);
            case "constant_with_warmup":
                return new ConstantWithWarmupScheduler(numWarmupSteps);
            default:
                throw new IllegalArgumentException("Unknown scheduler: " + name);
        }
    }

    /** Functional interface: step -> LR multiplier. */
    @FunctionalInterface
    public interface Lambda extends Function<Long, Double> {}

    // ---- Schedulers ------------------------------------------------------------

    static class LinearScheduler implements Lambda {
        private final int warmup;
        private final int total;

        LinearScheduler(int warmup, int total) {
            this.warmup = warmup;
            this.total = total;
        }

        @Override
        public Double apply(Long step) {
            double t = step.doubleValue();
            if (t < warmup) {
                return t / Math.max(1, warmup);
            }
            return Math.max(0.0, (total - t) / Math.max(1, total - warmup));
        }
    }

    static class CosineScheduler implements Lambda {
        private final int warmup;
        private final double maxRate;
        private final double minRate;

        CosineScheduler(int warmup, int total, double maxRate, double minRate) {
            this.warmup = warmup;
            this.maxRate = maxRate;
            this.minRate = minRate;
        }

        @Override
        public Double apply(Long step) {
            if (step < warmup) {
                return step.doubleValue() / Math.max(1, warmup);
            }
            double progress = (step - warmup) / Math.max(1, total - warmup);
            return minRate + 0.5 * (maxRate - minRate)
                    * (1.0 + Math.cos(Math.PI * progress));
        }
    }

    static class CosineWithRestartsScheduler implements Lambda {
        private final int warmup;
        private final int total;
        private static final int NUM_RESTARTS = 1;

        CosineWithRestartsScheduler(int warmup, int total) {
            this.warmup = warmup;
            this.total = total;
        }

        @Override
        public Double apply(Long step) {
            if (step < warmup) {
                return step.doubleValue() / Math.max(1, warmup);
            }
            double progress = (step - warmup) / Math.max(1, total - warmup);
            double angle = Math.PI * NUM_RESTARTS * progress;
            return Math.max(0.0, 0.5 * (1.0 + Math.cos(angle)));
        }
    }

    static class PolynomialScheduler implements Lambda {
        private final int warmup;
        private final int total;

        PolynomialScheduler(int warmup, int total) {
            this.warmup = warmup;
            this.total = total;
        }

        @Override
        public Double apply(Long step) {
            if (step < warmup) {
                return step.doubleValue() / Math.max(1, warmup);
            }
            double progress = (step - warmup) / Math.max(1, total - warmup);
            return Math.pow(1.0 - progress, 2.0);
        }
    }

    static class ConstantScheduler implements Lambda {
        private final int warmup;
        private final double multiplier;

        ConstantScheduler(int warmup, double multiplier) {
            this.warmup = warmup;
            this.multiplier = multiplier;
        }

        @Override
        public Double apply(Long step) {
            if (step < warmup) {
                return step.doubleValue() / Math.max(1, warmup);
            }
            return multiplier;
        }
    }

    static class ConstantWithWarmupScheduler implements Lambda {
        private final int warmup;

        ConstantWithWarmupScheduler(int warmup) {
            this.warmup = warmup;
        }

        @Override
        public Double apply(Long step) {
            if (step < warmup) {
                return step.doubleValue() / Math.max(1, warmup);
            }
            return 1.0;
        }
    }
}
