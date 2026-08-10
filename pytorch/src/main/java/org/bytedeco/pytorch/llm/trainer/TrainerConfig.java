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
package org.bytedeco.pytorch.llm.trainer;

import java.util.List;
import java.util.Map;

/**
 * Enterprise trainer configuration.
 */
public class TrainerConfig {

    public static final String VERSION = "2.0";

    // Training parameters
    private final int numTrain_epochs;
    private final int maxSteps;
    private final int gradientAccumulationSteps;
    private final int trainBatchSize;
    private final int evalBatchSize;
    private final int maxSeqLength;

    // Optimizer parameters
    private final float learningRate;
    private final float weightDecay;
    private final float warmupRatio;
    private final float lrEnd;

    // Gradient parameters
    private final float maxGradNorm;
    private final boolean clipGradNorm;

    // AMP parameters
    private final boolean useAmp;
    private final float ampInitScale;
    private final float ampGrowthFactor;
    private final float ampBackoffFactor;

    // Checkpoint parameters
    private final int saveSteps;
    private final int evalSteps;
    private final int loggingSteps;
    private final String saveStrategy;
    private final String outputDir;

    // Distributed training
    private final int worldSize;
    private final int rank;
    private final boolean distributed;

    // Precision training
    private final String precision;  // fp32, fp16, bf16, fp8

    // Long context
    private final boolean useLongContext;
    private final int maxContextLength;

    private TrainerConfig(Builder b) {
        this.numTrain_epochs = b.numTrain_epochs;
        this.maxSteps = b.maxSteps;
        this.gradientAccumulationSteps = b.gradientAccumulationSteps;
        this.trainBatchSize = b.trainBatchSize;
        this.evalBatchSize = b.evalBatchSize;
        this.maxSeqLength = b.maxSeqLength;
        this.learningRate = b.learningRate;
        this.weightDecay = b.weightDecay;
        this.warmupRatio = b.warmupRatio;
        this.lrEnd = b.lrEnd;
        this.maxGradNorm = b.maxGradNorm;
        this.clipGradNorm = b.clipGradNorm;
        this.useAmp = b.useAmp;
        this.ampInitScale = b.ampInitScale;
        this.ampGrowthFactor = b.ampGrowthFactor;
        this.ampBackoffFactor = b.ampBackoffFactor;
        this.saveSteps = b.saveSteps;
        this.evalSteps = b.evalSteps;
        this.loggingSteps = b.loggingSteps;
        this.saveStrategy = b.saveStrategy;
        this.outputDir = b.outputDir;
        this.worldSize = b.worldSize;
        this.rank = b.rank;
        this.distributed = b.distributed;
        this.precision = b.precision;
        this.useLongContext = b.useLongContext;
        this.maxContextLength = b.maxContextLength;
    }

    public static TrainerConfig defaults() {
        return builder().build();
    }

    public static Builder builder() { return new Builder(); }

    // Getters
    public int numTrain_epochs() { return numTrain_epochs; }
    public int maxSteps() { return maxSteps; }
    public int gradientAccumulationSteps() { return gradientAccumulationSteps; }
    public int trainBatchSize() { return trainBatchSize; }
    public int evalBatchSize() { return evalBatchSize; }
    public int maxSeqLength() { return maxSeqLength; }
    public float learningRate() { return learningRate; }
    public float weightDecay() { return weightDecay; }
    public float warmupRatio() { return warmupRatio; }
    public float lrEnd() { return lrEnd; }
    public float maxGradNorm() { return maxGradNorm; }
    public boolean clipGradNorm() { return clipGradNorm; }
    public boolean useAmp() { return useAmp; }
    public float ampInitScale() { return ampInitScale; }
    public float ampGrowthFactor() { return ampGrowthFactor; }
    public float ampBackoffFactor() { return ampBackoffFactor; }
    public int saveSteps() { return saveSteps; }
    public int evalSteps() { return evalSteps; }
    public int loggingSteps() { return loggingSteps; }
    public String saveStrategy() { return saveStrategy; }
    public String outputDir() { return outputDir; }
    public int worldSize() { return worldSize; }
    public int rank() { return rank; }
    public boolean distributed() { return distributed; }
    public String precision() { return precision; }
    public boolean useLongContext() { return useLongContext; }
    public int maxContextLength() { return maxContextLength; }

    /**
     * Builder for TrainerConfig.
     */
    public static class Builder {
        private int numTrain_epochs = 3;
        private int maxSteps = 1000;
        private int gradientAccumulationSteps = 1;
        private int trainBatchSize = 1;
        private int evalBatchSize = 1;
        private int maxSeqLength = 2048;

        private float learningRate = 1e-4f;
        private float weightDecay = 0.01f;
        private float warmupRatio = 0.1f;
        private float lrEnd = 0.0f;

        private float maxGradNorm = 1.0f;
        private boolean clipGradNorm = true;

        private boolean useAmp = true;
        private float ampInitScale = 65536.0f;
        private float ampGrowthFactor = 1.01f;
        private float ampBackoffFactor = 0.5f;

        private int saveSteps = 500;
        private int evalSteps = 500;
        private int loggingSteps = 10;
        private String saveStrategy = "steps";
        private String outputDir = "output";

        private int worldSize = 1;
        private int rank = 0;
        private boolean distributed = false;

        private String precision = "bf16";  // fp32, fp16, bf16, fp8
        private boolean useLongContext = false;
        private int maxContextLength = 32768;

        public Builder numTrain_epochs(int v) { this.numTrain_epochs = v; return this; }
        public Builder maxSteps(int v) { this.maxSteps = v; return this; }
        public Builder gradientAccumulationSteps(int v) { this.gradientAccumulationSteps = v; return this; }
        public Builder trainBatchSize(int v) { this.trainBatchSize = v; return this; }
        public Builder evalBatchSize(int v) { this.evalBatchSize = v; return this; }
        public Builder maxSeqLength(int v) { this.maxSeqLength = v; return this; }
        public Builder learningRate(float v) { this.learningRate = v; return this; }
        public Builder weightDecay(float v) { this.weightDecay = v; return this; }
        public Builder warmupRatio(float v) { this.warmupRatio = v; return this; }
        public Builder lrEnd(float v) { this.lrEnd = v; return this; }
        public Builder maxGradNorm(float v) { this.maxGradNorm = v; return this; }
        public Builder clipGradNorm(boolean v) { this.clipGradNorm = v; return this; }
        public Builder useAmp(boolean v) { this.useAmp = v; return this; }
        public Builder ampInitScale(float v) { this.ampInitScale = v; return this; }
        public Builder ampGrowthFactor(float v) { this.ampGrowthFactor = v; return this; }
        public Builder ampBackoffFactor(float v) { this.ampBackoffFactor = v; return this; }
        public Builder saveSteps(int v) { this.saveSteps = v; return this; }
        public Builder evalSteps(int v) { this.evalSteps = v; return this; }
        public Builder loggingSteps(int v) { this.loggingSteps = v; return this; }
        public Builder saveStrategy(String v) { this.saveStrategy = v; return this; }
        public Builder outputDir(String v) { this.outputDir = v; return this; }
        public Builder worldSize(int v) { this.worldSize = v; return this; }
        public Builder rank(int v) { this.rank = v; return this; }
        public Builder distributed(boolean v) { this.distributed = v; return this; }
        public Builder precision(String v) { this.precision = v; return this; }
        public Builder useLongContext(boolean v) { this.useLongContext = v; return this; }
        public Builder maxContextLength(int v) { this.maxContextLength = v; return this; }

        /**
         * Configure for BF16 training (recommended).
         */
        public Builder bf16() {
            this.precision = "bf16";
            this.useAmp = true;
            return this;
        }

        /**
         * Configure for FP16 training.
         */
        public Builder fp16() {
            this.precision = "fp16";
            this.useAmp = true;
            return this;
        }

        /**
         * Configure for FP8 training (H100/H200).
         */
        public Builder fp8() {
            this.precision = "fp8";
            this.useAmp = true;
            return this;
        }

        /**
         * Configure for distributed training.
         */
        public Builder distributed(int worldSize, int rank) {
            this.distributed = true;
            this.worldSize = worldSize;
            this.rank = rank;
            return this;
        }

        /**
         * Configure for long context models.
         */
        public Builder longContext(int maxLength) {
            this.useLongContext = true;
            this.maxContextLength = maxLength;
            return this;
        }

        public TrainerConfig build() {
            return new TrainerConfig(this);
        }
    }

    @Override
    public String toString() {
        return String.format(
                "TrainerConfig{epochs=%d, steps=%d, batchSize=%d, seqLen=%d, " +
                "lr=%.2e, precision=%s, distributed=%b}",
                numTrain_epochs, maxSteps, trainBatchSize, maxSeqLength,
                learningRate, precision, distributed);
    }
}
