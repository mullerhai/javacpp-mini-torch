/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version ( collectively, the "License");
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

import java.util.Arrays;
import java.util.Objects;

/**
 * Configuration for Constitutional AI (CAI) trainer.
 *
 * <p>Constitutional AI is Anthropic's approach to AI safety that uses:
 * <ul>
 *   <li>A set of constitutional principles</li>
 *   <li>Self-critique and revision</li>
 *   <li>RLHF with helpful and harmless objectives</li>
 * </ul>
 *
 * <p>Reference: "Constitutional AI: Harmlessness from AI Feedback" (Anthropic)
 *
 * <pre>{@code
 * ConstitutionalAIConfig config = ConstitutionalAIConfig.builder()
 *     .principles(Arrays.asList("helpful", "harmless", "honest"))
 *     .critiqueWeight(0.5)
 *     .revisionSteps(2)
 *     .build();
 * }</pre>
 */
public final class ConstitutionalAIConfig extends TrainerConfig {
    private final String principles;
    private final double critiqueWeight;
    private final int revisionSteps;
    private final boolean useSLICF;  // Supervised Learning from AI Feedback
    private final boolean useRLAIF;  // RL from AI Feedback
    private final double harmlessnessWeight;
    private final double helpfulnessWeight;
    private final double honestyWeight;

    private ConstitutionalAIConfig(Builder b) {
        super(b);
        this.principles = b.principles;
        this.critiqueWeight = b.critiqueWeight;
        this.revisionSteps = b.revisionSteps;
        this.useSLICF = b.useSLICF;
        this.useRLAIF = b.useRLAIF;
        this.harmlessnessWeight = b.harmlessnessWeight;
        this.helpfulnessWeight = b.helpfulnessWeight;
        this.honestyWeight = b.honestyWeight;
    }

    public String principles() { return principles; }
    public double critiqueWeight() { return critiqueWeight; }
    public int revisionSteps() { return revisionSteps; }
    public boolean useSLICF() { return useSLICF; }
    public boolean useRLAIF() { return useRLAIF; }
    public double harmlessnessWeight() { return harmlessnessWeight; }
    public double helpfulnessWeight() { return helpfulnessWeight; }
    public double honestyWeight() { return honestyWeight; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private String principles = "Helpful, Harmless, Honest";
        private double critiqueWeight = 0.5;
        private int revisionSteps = 2;
        private boolean useSLICF = true;
        private boolean useRLAIF = true;
        private double harmlessnessWeight = 0.4;
        private double helpfulnessWeight = 0.3;
        private double honestyWeight = 0.3;

        public Builder principles(String principles) {
            this.principles = principles;
            return this;
        }

        public Builder principles(String[] principles) {
            this.principles = String.join(", ", principles);
            return this;
        }

        public Builder critiqueWeight(double critiqueWeight) {
            this.critiqueWeight = Math.max(0, Math.min(1, critiqueWeight));
            return this;
        }

        public Builder revisionSteps(int revisionSteps) {
            this.revisionSteps = Math.max(1, revisionSteps);
            return this;
        }

        public Builder useSLICF(boolean useSLICF) {
            this.useSLICF = useSLICF;
            return this;
        }

        public Builder useRLAIF(boolean useRLAIF) {
            this.useRLAIF = useRLAIF;
            return this;
        }

        public Builder harmlessnessWeight(double harmlessnessWeight) {
            this.harmlessnessWeight = Math.max(0, Math.min(1, harmlessnessWeight));
            return this;
        }

        public Builder helpfulnessWeight(double helpfulnessWeight) {
            this.helpfulnessWeight = Math.max(0, Math.min(1, helpfulnessWeight));
            return this;
        }

        public Builder honestyWeight(double honestyWeight) {
            this.honestyWeight = Math.max(0, Math.min(1, honestyWeight));
            return this;
        }

        @Override
        public ConstitutionalAIConfig build() {
            // Normalize weights
            double total = harmlessnessWeight + helpfulnessWeight + honestyWeight;
            if (Math.abs(total - 1.0) > 0.01) {
                harmlessnessWeight /= total;
                helpfulnessWeight /= total;
                honestyWeight /= total;
            }
            return new ConstitutionalAIConfig(this);
        }
    }

    @Override
    public String toString() {
        return "ConstitutionalAIConfig{" +
                "principles=" + principles +
                ", critiqueWeight=" + critiqueWeight +
                ", revisionSteps=" + revisionSteps +
                ", weights(harmless=" + harmlessnessWeight +
                ", helpful=" + helpfulnessWeight +
                ", honest=" + honestyWeight +
                ")}";
    }
}
