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
package org.bytedeco.pytorch.benchmark;

import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer;
import org.bytedeco.pytorch.llm.trainer.TrainerConfig;
import org.bytedeco.pytorch.llm.trainer.MultiModalTrainer;
import org.bytedeco.pytorch.llm.modules.LongContextRoPE;
import org.bytedeco.pytorch.Tensor;

import java.util.concurrent.TimeUnit;

/**
 * JMH benchmarks for LLM training performance.
 */
@org.openjdk.jmh.annotations.State(org.openjdk.jmh.annotations.Scope.Thread)
@org.openjdk.jmh.annotations.OutputTimeUnit(TimeUnit.MILLISECONDS)
@org.openjdk.jmh.annotations.Measurement(iterations = 5, time = 1)
@org.openjdk.jmh.annotations.Warmup(iterations = 3, time = 1)
public class TrainingBenchmark {

    private Module model;
    private TrainerConfig config;

    @org.openjdk.jmh.annotations.Setup
    public void setup() {
        // Initialize model
        model = torch.nn.linear(4096, 4096);

        // Initialize config
        config = TrainerConfig.builder()
                .bf16()
                .trainBatchSize(1)
                .maxSeqLength(512)
                .build();
    }

    @org.openjdk.jmh.annotations.TearDown
    public void teardown() {
        if (model != null) model.close();
    }

    /**
     * Benchmark EnterpriseTrainer initialization.
     */
    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public EnterpriseTrainer trainerInitialization() {
        return EnterpriseTrainer.builder()
                .model(model)
                .config(config)
                .build();
    }

    /**
     * Benchmark training step (dummy).
     */
    @org.openjdk.jmh.annotations.Benchmark
    public double trainingStep() {
        Tensor loss = torch.randn(new long[]{1}).abs().add(0.1);
        return loss.item_double();
    }

    /**
     * Benchmark MultiModalTrainer.
     */
    @org.openjdk.jmh.annotations.Benchmark
    public MultiModalTrainer multiModalTrainerInitialization() {
        return MultiModalTrainer.builder()
                .model(model)
                .build();
    }

    /**
     * Benchmark with different precisions.
     */
    @org.openjdk.jmh.annotations.Param({"fp32", "fp16", "bf16"})
    public String precision;

    @org.openjdk.jmh.annotations.Setup
    public void setupPrecision() {
        switch (precision) {
            case "fp32":
                config = TrainerConfig.builder()
                        .trainBatchSize(1)
                        .maxSeqLength(512)
                        .build();
                break;
            case "fp16":
                config = TrainerConfig.builder()
                        .fp16()
                        .trainBatchSize(1)
                        .maxSeqLength(512)
                        .build();
                break;
            case "bf16":
                config = TrainerConfig.builder()
                        .bf16()
                        .trainBatchSize(1)
                        .maxSeqLength(512)
                        .build();
                break;
        }
    }

    @org.openjdk.jmh.annotations.Benchmark
    public EnterpriseTrainer precisionTrainerInit() {
        return EnterpriseTrainer.builder()
                .model(model)
                .config(config)
                .build();
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
