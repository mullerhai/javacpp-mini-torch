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
package org.bytedeco.pytorch.llm.transformers.trainer;

import java.util.ArrayList;
import java.util.List;

/**
 * HuggingFace-style training arguments mirroring {@code transformers.TrainingArguments}.
 *
 * <p>Use the {@link Builder} to configure training hyperparameters, logging,
 * checkpointing, and Hub upload options.
 *
 * <pre>{@code
 * TrainingArguments args = TrainingArguments.builder()
 *     .outputDir("./output")
 *     .numTrainEpochs(3)
 *     .perDeviceTrainBatchSize(8)
 *     .learningRate(5e-5f)
 *     .build();
 * }</pre>
 */
public final class TrainingArguments {

    private final String outputDir;
    private final double numTrainEpochs;
    private final int perDeviceTrainBatchSize;
    private final int perDeviceEvalBatchSize;
    private final float learningRate;
    private final float weightDecay;
    private final int warmupSteps;
    private final float warmupRatio;
    private final String lrSchedulerType;
    private final int loggingSteps;
    private final int saveSteps;
    private final int evalSteps;
    private final String evalStrategy;
    private final String saveStrategy;
    private final int saveTotalLimit;
    private final boolean fp16;
    private final boolean bf16;
    private final int gradientAccumulationSteps;
    private final int maxSteps;
    private final int seed;
    private final int dataloaderNumWorkers;
    private final boolean removeUnusedColumns;
    private final float labelSmoothingFactor;
    private final String optim;
    private final List<String> reportTo;
    private final boolean loadBestModelAtEnd;
    private final String metricForBestModel;
    private final boolean greaterIsBetter;
    private final boolean pushToHub;
    private final String hubModelId;
    private final String hubStrategy;
    private final String hubToken;

    private TrainingArguments(Builder b) {
        this.outputDir = b.outputDir;
        this.numTrainEpochs = b.numTrainEpochs;
        this.perDeviceTrainBatchSize = b.perDeviceTrainBatchSize;
        this.perDeviceEvalBatchSize = b.perDeviceEvalBatchSize;
        this.learningRate = b.learningRate;
        this.weightDecay = b.weightDecay;
        this.warmupSteps = b.warmupSteps;
        this.warmupRatio = b.warmupRatio;
        this.lrSchedulerType = b.lrSchedulerType;
        this.loggingSteps = b.loggingSteps;
        this.saveSteps = b.saveSteps;
        this.evalSteps = b.evalSteps;
        this.evalStrategy = b.evalStrategy;
        this.saveStrategy = b.saveStrategy;
        this.saveTotalLimit = b.saveTotalLimit;
        this.fp16 = b.fp16;
        this.bf16 = b.bf16;
        this.gradientAccumulationSteps = b.gradientAccumulationSteps;
        this.maxSteps = b.maxSteps;
        this.seed = b.seed;
        this.dataloaderNumWorkers = b.dataloaderNumWorkers;
        this.removeUnusedColumns = b.removeUnusedColumns;
        this.labelSmoothingFactor = b.labelSmoothingFactor;
        this.optim = b.optim;
        this.reportTo = List.copyOf(b.reportTo);
        this.loadBestModelAtEnd = b.loadBestModelAtEnd;
        this.metricForBestModel = b.metricForBestModel;
        this.greaterIsBetter = b.greaterIsBetter;
        this.pushToHub = b.pushToHub;
        this.hubModelId = b.hubModelId;
        this.hubStrategy = b.hubStrategy;
        this.hubToken = b.hubToken;
    }

    // Getters
    public String outputDir() { return outputDir; }
    public double numTrainEpochs() { return numTrainEpochs; }
    public int perDeviceTrainBatchSize() { return perDeviceTrainBatchSize; }
    public int perDeviceEvalBatchSize() { return perDeviceEvalBatchSize; }
    public float learningRate() { return learningRate; }
    public float weightDecay() { return weightDecay; }
    public int warmupSteps() { return warmupSteps; }
    public float warmupRatio() { return warmupRatio; }
    public String lrSchedulerType() { return lrSchedulerType; }
    public int loggingSteps() { return loggingSteps; }
    public int saveSteps() { return saveSteps; }
    public int evalSteps() { return evalSteps; }
    public String evalStrategy() { return evalStrategy; }
    public String saveStrategy() { return saveStrategy; }
    public int saveTotalLimit() { return saveTotalLimit; }
    public boolean fp16() { return fp16; }
    public boolean bf16() { return bf16; }
    public int gradientAccumulationSteps() { return gradientAccumulationSteps; }
    public int maxSteps() { return maxSteps; }
    public int seed() { return seed; }
    public int dataloaderNumWorkers() { return dataloaderNumWorkers; }
    public boolean removeUnusedColumns() { return removeUnusedColumns; }
    public float labelSmoothingFactor() { return labelSmoothingFactor; }
    public String optim() { return optim; }
    public List<String> reportTo() { return reportTo; }
    public boolean loadBestModelAtEnd() { return loadBestModelAtEnd; }
    public String metricForBestModel() { return metricForBestModel; }
    public boolean greaterIsBetter() { return greaterIsBetter; }
    public boolean pushToHub() { return pushToHub; }
    public String hubModelId() { return hubModelId; }
    public String hubStrategy() { return hubStrategy; }
    public String hubToken() { return hubToken; }

    /** Validate the argument set; throw {@link IllegalStateException} on inconsistencies. */
    public void validate() {
        if (perDeviceTrainBatchSize <= 0) {
            throw new IllegalStateException("perDeviceTrainBatchSize must be > 0, got: " + perDeviceTrainBatchSize);
        }
        if (perDeviceEvalBatchSize <= 0) {
            throw new IllegalStateException("perDeviceEvalBatchSize must be > 0, got: " + perDeviceEvalBatchSize);
        }
        if (learningRate <= 0) {
            throw new IllegalStateException("learningRate must be > 0, got: " + learningRate);
        }
        if (weightDecay < 0) {
            throw new IllegalStateException("weightDecay must be >= 0, got: " + weightDecay);
        }
        if (warmupSteps < 0) {
            throw new IllegalStateException("warmupSteps must be >= 0, got: " + warmupSteps);
        }
        if (warmupRatio < 0 || warmupRatio > 1) {
            throw new IllegalStateException("warmupRatio must be in [0,1], got: " + warmupRatio);
        }
        if (gradientAccumulationSteps <= 0) {
            throw new IllegalStateException("gradientAccumulationSteps must be > 0, got: " + gradientAccumulationSteps);
        }
        if (saveTotalLimit < 0) {
            throw new IllegalStateException("saveTotalLimit must be >= 0, got: " + saveTotalLimit);
        }
        if ("no".equalsIgnoreCase(evalStrategy) && loadBestModelAtEnd) {
            throw new IllegalStateException(
                    "loadBestModelAtEnd=true requires evalStrategy != 'no', got: '" + evalStrategy + "'");
        }
        if (fp16 && bf16) {
            throw new IllegalStateException("fp16 and bf16 cannot both be true");
        }
        if (labelSmoothingFactor < 0 || labelSmoothingFactor > 1) {
            throw new IllegalStateException("labelSmoothingFactor must be in [0,1], got: " + labelSmoothingFactor);
        }
        if (outputDir == null || outputDir.isBlank()) {
            throw new IllegalStateException("outputDir must be set");
        }
    }

    public Builder toBuilder() {
        return new Builder()
                .outputDir(outputDir)
                .numTrainEpochs(numTrainEpochs)
                .perDeviceTrainBatchSize(perDeviceTrainBatchSize)
                .perDeviceEvalBatchSize(perDeviceEvalBatchSize)
                .learningRate(learningRate)
                .weightDecay(weightDecay)
                .warmupSteps(warmupSteps)
                .warmupRatio(warmupRatio)
                .lrSchedulerType(lrSchedulerType)
                .loggingSteps(loggingSteps)
                .saveSteps(saveSteps)
                .evalSteps(evalSteps)
                .evalStrategy(evalStrategy)
                .saveStrategy(saveStrategy)
                .saveTotalLimit(saveTotalLimit)
                .fp16(fp16)
                .bf16(bf16)
                .gradientAccumulationSteps(gradientAccumulationSteps)
                .maxSteps(maxSteps)
                .seed(seed)
                .dataloaderNumWorkers(dataloaderNumWorkers)
                .removeUnusedColumns(removeUnusedColumns)
                .labelSmoothingFactor(labelSmoothingFactor)
                .optim(optim)
                .reportTo(new ArrayList<>(reportTo))
                .loadBestModelAtEnd(loadBestModelAtEnd)
                .metricForBestModel(metricForBestModel)
                .greaterIsBetter(greaterIsBetter)
                .pushToHub(pushToHub)
                .hubModelId(hubModelId)
                .hubStrategy(hubStrategy)
                .hubToken(hubToken);
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String outputDir = "./output";
        private double numTrainEpochs = 3.0;
        private int perDeviceTrainBatchSize = 8;
        private int perDeviceEvalBatchSize = 8;
        private float learningRate = 5e-5f;
        private float weightDecay = 0.0f;
        private int warmupSteps = 0;
        private float warmupRatio = 0.0f;
        private String lrSchedulerType = "linear";
        private int loggingSteps = 500;
        private int saveSteps = 500;
        private int evalSteps = 500;
        private String evalStrategy = "steps";
        private String saveStrategy = "steps";
        private int saveTotalLimit = 5;
        private boolean fp16 = false;
        private boolean bf16 = false;
        private int gradientAccumulationSteps = 1;
        private int maxSteps = -1;
        private int seed = 42;
        private int dataloaderNumWorkers = 0;
        private boolean removeUnusedColumns = true;
        private float labelSmoothingFactor = 0.0f;
        private String optim = "adamw_torch";
        private List<String> reportTo = List.of("none");
        private boolean loadBestModelAtEnd = false;
        private String metricForBestModel = null;
        private boolean greaterIsBetter = true;
        private boolean pushToHub = false;
        private String hubModelId = null;
        private String hubStrategy = "best";
        private String hubToken = null;

        public Builder outputDir(String v) { this.outputDir = v; return this; }
        public Builder numTrainEpochs(double v) { this.numTrainEpochs = v; return this; }
        public Builder perDeviceTrainBatchSize(int v) { this.perDeviceTrainBatchSize = v; return this; }
        public Builder perDeviceEvalBatchSize(int v) { this.perDeviceEvalBatchSize = v; return this; }
        public Builder learningRate(float v) { this.learningRate = v; return this; }
        public Builder weightDecay(float v) { this.weightDecay = v; return this; }
        public Builder warmupSteps(int v) { this.warmupSteps = v; return this; }
        public Builder warmupRatio(float v) { this.warmupRatio = v; return this; }
        public Builder lrSchedulerType(String v) { this.lrSchedulerType = v; return this; }
        public Builder loggingSteps(int v) { this.loggingSteps = v; return this; }
        public Builder saveSteps(int v) { this.saveSteps = v; return this; }
        public Builder evalSteps(int v) { this.evalSteps = v; return this; }
        public Builder evalStrategy(String v) { this.evalStrategy = v; return this; }
        public Builder saveStrategy(String v) { this.saveStrategy = v; return this; }
        public Builder saveTotalLimit(int v) { this.saveTotalLimit = v; return this; }
        public Builder fp16(boolean v) { this.fp16 = v; return this; }
        public Builder bf16(boolean v) { this.bf16 = v; return this; }
        public Builder gradientAccumulationSteps(int v) { this.gradientAccumulationSteps = v; return this; }
        public Builder maxSteps(int v) { this.maxSteps = v; return this; }
        public Builder seed(int v) { this.seed = v; return this; }
        public Builder dataloaderNumWorkers(int v) { this.dataloaderNumWorkers = v; return this; }
        public Builder removeUnusedColumns(boolean v) { this.removeUnusedColumns = v; return this; }
        public Builder labelSmoothingFactor(float v) { this.labelSmoothingFactor = v; return this; }
        public Builder optim(String v) { this.optim = v; return this; }
        public Builder reportTo(List<String> v) { this.reportTo = v; return this; }
        public Builder loadBestModelAtEnd(boolean v) { this.loadBestModelAtEnd = v; return this; }
        public Builder metricForBestModel(String v) { this.metricForBestModel = v; return this; }
        public Builder greaterIsBetter(boolean v) { this.greaterIsBetter = v; return this; }
        public Builder pushToHub(boolean v) { this.pushToHub = v; return this; }
        public Builder hubModelId(String v) { this.hubModelId = v; return this; }
        public Builder hubStrategy(String v) { this.hubStrategy = v; return this; }
        public Builder hubToken(String v) { this.hubToken = v; return this; }

        public TrainingArguments build() {
            return new TrainingArguments(this);
        }
    }
}
