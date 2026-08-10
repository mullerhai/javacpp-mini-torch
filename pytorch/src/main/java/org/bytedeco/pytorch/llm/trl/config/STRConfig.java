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
 * Configuration for Self-Taught Reasoning (STR) trainer.
 *
 * <p>STR enables models to learn reasoning capabilities through self-generated
 * chains of thought, self-critique, and iterative refinement.
 *
 * <p>Features:
 * <ul>
 *   <li>Chain-of-thought generation</li>
 *   <li>Self-critique and verification</li>
 *   <li>Iterative refinement</li>
 *   <li>Process reward modeling</li>
 * </ul>
 *
 * <p>Reference: "Self-Taught Reasoning" (Kimi/DeepSeek research)
 */
public final class STRConfig extends TrainerConfig {
    private final int maxReasoningSteps;
    private final double reasoningTemperature;
    private final boolean useProcessReward;
    private final boolean useSelfCritique;
    private final double critiqueWeight;
    private final int beamSize;
    private final double reasoningAlpha;  // Weight for reasoning bonus
    private final boolean useCoT;

    private STRConfig(Builder b) {
        super(b);
        this.maxReasoningSteps = b.maxReasoningSteps;
        this.reasoningTemperature = b.reasoningTemperature;
        this.useProcessReward = b.useProcessReward;
        this.useSelfCritique = b.useSelfCritique;
        this.critiqueWeight = b.critiqueWeight;
        this.beamSize = b.beamSize;
        this.reasoningAlpha = b.reasoningAlpha;
        this.useCoT = b.useCoT;
    }

    public int maxReasoningSteps() { return maxReasoningSteps; }
    public double reasoningTemperature() { return reasoningTemperature; }
    public boolean useProcessReward() { return useProcessReward; }
    public boolean processReward() { return useProcessReward; }  // Alias
    public boolean useSelfCritique() { return useSelfCritique; }
    public double critiqueWeight() { return critiqueWeight; }
    public int beamSize() { return beamSize; }
    public double reasoningAlpha() { return reasoningAlpha; }
    public boolean useCoT() { return useCoT; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private int maxReasoningSteps = 8;
        private double reasoningTemperature = 0.8;
        private boolean useProcessReward = true;
        private boolean useSelfCritique = true;
        private double critiqueWeight = 0.3;
        private int beamSize = 4;
        private double reasoningAlpha = 0.5;
        private boolean useCoT = true;
        private boolean processReward = true;  // Alias for useProcessReward

        public Builder maxReasoningSteps(int maxReasoningSteps) {
            this.maxReasoningSteps = Math.max(1, maxReasoningSteps);
            return this;
        }

        public Builder reasoningTemperature(double reasoningTemperature) {
            this.reasoningTemperature = Math.max(0.1, Math.min(2.0, reasoningTemperature));
            return this;
        }

        public Builder useProcessReward(boolean useProcessReward) {
            this.useProcessReward = useProcessReward;
            return this;
        }

        public Builder useSelfCritique(boolean useSelfCritique) {
            this.useSelfCritique = useSelfCritique;
            return this;
        }

        public Builder critiqueWeight(double critiqueWeight) {
            this.critiqueWeight = Math.max(0, Math.min(1, critiqueWeight));
            return this;
        }

        public Builder beamSize(int beamSize) {
            this.beamSize = Math.max(1, beamSize);
            return this;
        }

        public Builder reasoningAlpha(double reasoningAlpha) {
            this.reasoningAlpha = Math.max(0, Math.min(1, reasoningAlpha));
            return this;
        }

        public Builder useCoT(boolean useCoT) {
            this.useCoT = useCoT;
            return this;
        }

        @Override
        public STRConfig build() {
            return new STRConfig(this);
        }
    }

    @Override
    public String toString() {
        return "STRConfig{" +
                "maxReasoningSteps=" + maxReasoningSteps +
                ", useProcessReward=" + useProcessReward +
                ", useSelfCritique=" + useSelfCritique +
                ", beamSize=" + beamSize +
                ", useCoT=" + useCoT +
                '}';
    }
}
