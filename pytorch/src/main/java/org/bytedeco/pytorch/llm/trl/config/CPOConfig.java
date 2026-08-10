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
 * Configuration for CPO (Contrastive Preference Optimization) trainer.
 *
 * <p>CPO combines DPO-style preference learning with contrastive loss to better
 * separate chosen and rejected responses. Inspired by Meta AI research on
 * contrastive alignment.
 *
 * <p>Reference: "Contrastive Preference Learning: Learning from Human Feedback
 * without RL" (Meta AI)
 *
 * <pre>{@code
 * CPOConfig config = CPOConfig.builder()
 *     .beta(0.1)
 *     .contrastiveAlpha(0.5)
 *     .margin(0.5)
 *     .build();
 * }</pre>
 */
public final class CPOConfig extends TrainerConfig {
    private final double beta;
    private final double contrastiveAlpha;
    private final double margin;
    private final boolean useReferenceModel;
    private final int contrastiveSteps;

    private CPOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.contrastiveAlpha = b.contrastiveAlpha;
        this.margin = b.margin;
        this.useReferenceModel = b.useReferenceModel;
        this.contrastiveSteps = b.contrastiveSteps;
    }

    /** KL divergence coefficient for reference model. */
    public double beta() { return beta; }

    /** Weight for contrastive loss component. */
    public double contrastiveAlpha() { return contrastiveAlpha; }

    /** Margin for contrastive loss (margin > 0 separates chosen > rejected). */
    public double margin() { return margin; }

    /** Whether to use reference model for KL penalty. */
    public boolean useReferenceModel() { return useReferenceModel; }

    /** Number of contrastive steps per DPO step. */
    public int contrastiveSteps() { return contrastiveSteps; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double beta = 0.1;
        private double contrastiveAlpha = 0.5;
        private double margin = 0.5;
        private boolean useReferenceModel = true;
        private int contrastiveSteps = 1;

        public Builder beta(double beta) {
            this.beta = Math.max(0, beta);
            return this;
        }

        public Builder contrastiveAlpha(double contrastiveAlpha) {
            this.contrastiveAlpha = Math.max(0, contrastiveAlpha);
            return this;
        }

        public Builder margin(double margin) {
            this.margin = margin;
            return this;
        }

        public Builder useReferenceModel(boolean useReferenceModel) {
            this.useReferenceModel = useReferenceModel;
            return this;
        }

        public Builder contrastiveSteps(int contrastiveSteps) {
            this.contrastiveSteps = Math.max(1, contrastiveSteps);
            return this;
        }

        @Override
        public CPOConfig build() {
            return new CPOConfig(this);
        }
    }

    @Override
    public String toString() {
        return "CPOConfig{" +
                "beta=" + beta +
                ", contrastiveAlpha=" + contrastiveAlpha +
                ", margin=" + margin +
                ", useReferenceModel=" + useReferenceModel +
                ", contrastiveSteps=" + contrastiveSteps +
                '}';
    }
}
