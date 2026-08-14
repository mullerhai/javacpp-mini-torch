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
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trl.config;

/**
 * Configuration for RRHF (Rank Responses to Rank Responses) trainer.
 *
 * <p>RRHF aligns language models using a ranking loss on generated responses.
 * Unlike DPO, it doesn't require preference pairs - instead, it uses a reward
 * model to score multiple generated responses and learns to match the ranking.
 *
 * <p>Key features:
 * <ul>
 *   <li>Only requires a reward model (no reference model needed)</li>
 *   <li>Can use any number of responses per prompt</li>
 *   <li>Compatible with any reward model scoring function</li>
 *   <li>Simpler data requirements than pairwise methods</li>
 * </ul>
 *
 * <p>Reference: "RRHF: Rank Responses to Rank Responses for Human Preference"
 * (Yuan et al., 2023)
 *
 * @see org.bytedeco.pytorch.llm.trl.RRHFTrainer
 */
public final class RRHFConfig extends TrainerConfig {

    // ==================== RRHF-specific Parameters ====================

    /** Reward model weight in combined loss (default: 1.0) */
    private final double rewardWeight;

    /** Ratio loss weight for response ranking (default: 1.0) */
    private final double ratioWeight;

    /** Number of responses to generate/rank per prompt (default: 4) */
    private final int numResponses;

    /** Use sample-level ranking vs token-level (default: true) */
    private final boolean sampleLevelRanking;

    /** Use pairwise ranking loss (default: true) */
    private final boolean pairwiseLoss;

    /** Temperature for softmax over rewards (default: 1.0) */
    private final double rewardTemperature;

    // ==================== Constructor ====================

    private RRHFConfig(Builder b) {
        super(b);
        this.rewardWeight = b.rewardWeight;
        this.ratioWeight = b.ratioWeight;
        this.numResponses = b.numResponses;
        this.sampleLevelRanking = b.sampleLevelRanking;
        this.pairwiseLoss = b.pairwiseLoss;
        this.rewardTemperature = b.rewardTemperature;
    }

    // ==================== Getters ====================

    /**
     * Weight for reward model loss term.
     * Default: 1.0
     */
    public double rewardWeight() { return rewardWeight; }

    /**
     * Weight for ratio/importance sampling loss term.
     * Default: 1.0
     */
    public double ratioWeight() { return ratioWeight; }

    /**
     * Number of responses to generate/rank per prompt.
     * Default: 4
     */
    public int numResponses() { return numResponses; }

    /**
     * Whether to use sample-level vs token-level ranking.
     * Default: true (sample-level)
     */
    public boolean sampleLevelRanking() { return sampleLevelRanking; }

    /**
     * Whether to use pairwise ranking loss.
     * Default: true
     */
    public boolean pairwiseLoss() { return pairwiseLoss; }

    /**
     * Temperature for softmax over rewards.
     * Default: 1.0
     */
    public double rewardTemperature() { return rewardTemperature; }

    // ==================== Builder ====================

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {

        private double rewardWeight = 1.0;
        private double ratioWeight = 1.0;
        private int numResponses = 4;
        private boolean sampleLevelRanking = true;
        private boolean pairwiseLoss = true;
        private double rewardTemperature = 1.0;

        public Builder rewardWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("rewardWeight must be non-negative");
            this.rewardWeight = v;
            return this;
        }

        public Builder ratioWeight(double v) {
            if (v < 0) throw new IllegalArgumentException("ratioWeight must be non-negative");
            this.ratioWeight = v;
            return this;
        }

        public Builder numResponses(int v) {
            if (v < 2) throw new IllegalArgumentException("numResponses must be >= 2");
            this.numResponses = v;
            return this;
        }

        public Builder sampleLevelRanking(boolean v) {
            this.sampleLevelRanking = v;
            return this;
        }

        public Builder pairwiseLoss(boolean v) {
            this.pairwiseLoss = v;
            return this;
        }

        public Builder rewardTemperature(double v) {
            if (v <= 0) throw new IllegalArgumentException("rewardTemperature must be positive");
            this.rewardTemperature = v;
            return this;
        }

        @Override
        public RRHFConfig build() { return new RRHFConfig(this); }
    }

    @Override
    public String toString() {
        return "RRHFConfig{" +
                "rewardWeight=" + rewardWeight +
                ", ratioWeight=" + ratioWeight +
                ", numResponses=" + numResponses +
                ", sampleLevelRanking=" + sampleLevelRanking +
                '}';
    }
}
