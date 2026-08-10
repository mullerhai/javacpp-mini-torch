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
 * Configuration for MultiModal GRPO trainer.
 *
 * <p>MultiModal GRPO extends standard GRPO to handle multimodal inputs (text + images
 * + audio + video) with modality-specific reward shaping.
 *
 * <p>Features:
 * <ul>
 *   <li>Modality-specific encoding support</li>
 *   <li>Cross-modal attention reward shaping</li>
 *   <li>Modality dropout for robustness</li>
 *   <li>Hierarchical reward aggregation</li>
 * </ul>
 *
 * <p>Reference: ByteDance multimodal research
 *
 * <pre>{@code
 * MultiModalGRPOConfig config = MultiModalGRPOConfig.builder()
 *     .groupSize(4)
 *     .rewardWeights(Map.of("text", 0.3, "image", 0.4, "audio", 0.3))
 *     .modalityDropout(0.1)
 *     .crossModalReward(true)
 *     .build();
 * }</pre>
 */
public final class MultiModalGRPOConfig extends TrainerConfig {
    private final int groupSize;
    private final double beta;
    private final double rewardNormScale;
    private final double entropyCoeff;
    private final double modalityDropout;
    private final boolean crossModalReward;
    private final String modalityConfig;
    private final double imageWeight;
    private final double audioWeight;
    private final double textWeight;
    private final boolean adaptiveGrouping;

    private MultiModalGRPOConfig(Builder b) {
        super(b);
        this.groupSize = b.groupSize;
        this.beta = b.beta;
        this.rewardNormScale = b.rewardNormScale;
        this.entropyCoeff = b.entropyCoeff;
        this.modalityDropout = b.modalityDropout;
        this.crossModalReward = b.crossModalReward;
        this.modalityConfig = b.modalityConfig;
        this.imageWeight = b.imageWeight;
        this.audioWeight = b.audioWeight;
        this.textWeight = b.textWeight;
        this.adaptiveGrouping = b.adaptiveGrouping;
    }

    public int groupSize() { return groupSize; }
    public double beta() { return beta; }
    public double rewardNormScale() { return rewardNormScale; }
    public double entropyCoeff() { return entropyCoeff; }
    public double modalityDropout() { return modalityDropout; }
    public boolean crossModalReward() { return crossModalReward; }
    public String modalityConfig() { return modalityConfig; }
    public double imageWeight() { return imageWeight; }
    public double audioWeight() { return audioWeight; }
    public double textWeight() { return textWeight; }
    public boolean adaptiveGrouping() { return adaptiveGrouping; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private int groupSize = 4;
        private double beta = 0.1;
        private double rewardNormScale = 0.0;  // 0 means no normalization
        private double entropyCoeff = 0.01;
        private double modalityDropout = 0.1;
        private boolean crossModalReward = true;
        private String modalityConfig = "text,image";
        private double imageWeight = 0.5;
        private double audioWeight = 0.25;
        private double textWeight = 0.25;
        private boolean adaptiveGrouping = true;

        public Builder groupSize(int groupSize) {
            this.groupSize = Math.max(2, groupSize);
            return this;
        }

        public Builder beta(double beta) {
            this.beta = Math.max(0.001, beta);
            return this;
        }

        public Builder rewardNormScale(double rewardNormScale) {
            this.rewardNormScale = Math.max(0, rewardNormScale);
            return this;
        }

        public Builder entropyCoeff(double entropyCoeff) {
            this.entropyCoeff = Math.max(0, entropyCoeff);
            return this;
        }

        public Builder modalityDropout(double modalityDropout) {
            this.modalityDropout = Math.max(0, Math.min(0.5, modalityDropout));
            return this;
        }

        public Builder crossModalReward(boolean crossModalReward) {
            this.crossModalReward = crossModalReward;
            return this;
        }

        public Builder modalityConfig(String modalityConfig) {
            this.modalityConfig = modalityConfig;
            return this;
        }

        public Builder imageWeight(double imageWeight) {
            this.imageWeight = Math.max(0, Math.min(1, imageWeight));
            return this;
        }

        public Builder audioWeight(double audioWeight) {
            this.audioWeight = Math.max(0, Math.min(1, audioWeight));
            return this;
        }

        public Builder textWeight(double textWeight) {
            this.textWeight = Math.max(0, Math.min(1, textWeight));
            return this;
        }

        public Builder adaptiveGrouping(boolean adaptiveGrouping) {
            this.adaptiveGrouping = adaptiveGrouping;
            return this;
        }

        @Override
        public MultiModalGRPOConfig build() {
            // Normalize weights
            double total = imageWeight + audioWeight + textWeight;
            if (Math.abs(total - 1.0) > 0.01) {
                imageWeight /= total;
                audioWeight /= total;
                textWeight /= total;
            }
            return new MultiModalGRPOConfig(this);
        }
    }

    @Override
    public String toString() {
        return "MultiModalGRPOConfig{" +
                "groupSize=" + groupSize +
                ", beta=" + beta +
                ", modalityDropout=" + modalityDropout +
                ", crossModalReward=" + crossModalReward +
                ", weights(text=" + textWeight +
                ", image=" + imageWeight +
                ", audio=" + audioWeight +
                ")}";
    }
}
