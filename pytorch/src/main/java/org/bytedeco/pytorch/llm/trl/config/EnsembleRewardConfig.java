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
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trl.config;

import java.util.Objects;

/**
 * Configuration for Ensemble Reward Trainer.
 *
 * <p>Ensemble Reward Trainer trains multiple reward models simultaneously for
 * multi-objective optimization. Supports:
 * <ul>
 *   <li>Multiple reward heads with shared backbone</li>
 *   <li>Adaptive weighting based on objective importance</li>
 *   <li>Pareto-optimal training support</li>
 *   <li>Uncertainty-aware reward aggregation</li>
 * </ul>
 *
 * <p>Reference: Multi-objective optimization research
 *
 * <pre>{@code
 * EnsembleRewardConfig config = EnsembleRewardConfig.builder()
 *     .numRewards(3)
 *     .rewardNames("helpfulness", "safety", "coherence")
 *     .adaptiveWeighting(true)
 *     .useParetoTraining(true)
 *     .build();
 * }</pre>
 */
public final class EnsembleRewardConfig extends TrainerConfig {
    private final int numRewards;
    private final String rewardNames;
    private final double[] initialWeights;
    private final boolean adaptiveWeighting;
    private final double weightUpdateRate;
    private final boolean useParetoTraining;
    private final boolean useUncertainty;
    private final double uncertaintyThreshold;

    private EnsembleRewardConfig(Builder b) {
        super(b);
        this.numRewards = b.numRewards;
        this.rewardNames = b.rewardNames;
        this.initialWeights = b.initialWeights;
        this.adaptiveWeighting = b.adaptiveWeighting;
        this.weightUpdateRate = b.weightUpdateRate;
        this.useParetoTraining = b.useParetoTraining;
        this.useUncertainty = b.useUncertainty;
        this.uncertaintyThreshold = b.uncertaintyThreshold;
    }

    public int numRewards() { return numRewards; }
    public String rewardNames() { return rewardNames; }
    public double[] initialWeights() { return initialWeights; }
    public boolean adaptiveWeighting() { return adaptiveWeighting; }
    public double weightUpdateRate() { return weightUpdateRate; }
    public boolean useParetoTraining() { return useParetoTraining; }
    public boolean useUncertainty() { return useUncertainty; }
    public double uncertaintyThreshold() { return uncertaintyThreshold; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private int numRewards = 2;
        private String rewardNames = "reward0,reward1";
        private double[] initialWeights = new double[]{0.5, 0.5};
        private boolean adaptiveWeighting = true;
        private double weightUpdateRate = 0.01;
        private boolean useParetoTraining = false;
        private boolean useUncertainty = true;
        private double uncertaintyThreshold = 0.1;

        public Builder numRewards(int numRewards) {
            this.numRewards = Math.max(1, numRewards);
            return this;
        }

        public Builder rewardNames(String rewardNames) {
            this.rewardNames = rewardNames;
            return this;
        }

        public Builder initialWeights(double[] initialWeights) {
            this.initialWeights = initialWeights.clone();
            return this;
        }

        public Builder adaptiveWeighting(boolean adaptiveWeighting) {
            this.adaptiveWeighting = adaptiveWeighting;
            return this;
        }

        public Builder weightUpdateRate(double weightUpdateRate) {
            this.weightUpdateRate = Math.max(0.001, Math.min(0.1, weightUpdateRate));
            return this;
        }

        public Builder useParetoTraining(boolean useParetoTraining) {
            this.useParetoTraining = useParetoTraining;
            return this;
        }

        public Builder useUncertainty(boolean useUncertainty) {
            this.useUncertainty = useUncertainty;
            return this;
        }

        public Builder uncertaintyThreshold(double uncertaintyThreshold) {
            this.uncertaintyThreshold = Math.max(0, uncertaintyThreshold);
            return this;
        }

        @Override
        public EnsembleRewardConfig build() {
            // Normalize weights
            if (initialWeights == null || initialWeights.length != numRewards) {
                initialWeights = new double[numRewards];
                double w = 1.0 / numRewards;
                for (int i = 0; i < numRewards; i++) {
                    initialWeights[i] = w;
                }
            }
            double sum = 0;
            for (double w : initialWeights) sum += w;
            if (Math.abs(sum - 1.0) > 0.01) {
                for (int i = 0; i < initialWeights.length; i++) {
                    initialWeights[i] /= sum;
                }
            }
            return new EnsembleRewardConfig(this);
        }
    }

    @Override
    public String toString() {
        return "EnsembleRewardConfig{" +
                "numRewards=" + numRewards +
                ", rewardNames=" + rewardNames +
                ", adaptiveWeighting=" + adaptiveWeighting +
                ", useParetoTraining=" + useParetoTraining +
                '}';
    }
}
