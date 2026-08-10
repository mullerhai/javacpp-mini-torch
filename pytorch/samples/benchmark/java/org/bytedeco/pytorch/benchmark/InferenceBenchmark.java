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
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.benchmark;

import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.Tensor;

import java.util.concurrent.TimeUnit;

/**
 * JMH benchmarks for LLM inference performance.
 *
 * <p>Run with:
 * <pre>
 * mvn clean install
 * java -jar target/benchmarks.jar ".*InferenceBenchmark.*" -f 3 -wi 5 -i 10
 * </pre>
 *
 * <p>Reference: JMH (Java Microbenchmark Harness)
 */
@org.openjdk.jmh.annotations.State(org.openjdk.jmh.annotations.Scope.Thread)
@org.openjdk.jmh.annotations.OutputTimeUnit(TimeUnit.MILLISECONDS)
@org.openjdk.jmh.annotations.Measurement(iterations = 5, time = 1)
@org.openjdk.jmh.annotations.Warmup(iterations = 3, time = 1)
public class InferenceBenchmark {

    private Module model;
    private Tensor dummyInput;
    private int seqLength = 512;
    private int hiddenSize = 4096;
    private int batchSize = 1;

    @org.openjdk.jmh.annotations.Setup
    public void setup() {
        // Initialize model
        model = torch.nn.linear(hiddenSize, hiddenSize);

        // Create dummy input
        dummyInput = torch.randn(new long[]{batchSize, seqLength, hiddenSize});
    }

    @org.openjdk.jmh.annotations.TearDown
    public void teardown() {
        if (model != null) model.close();
        if (dummyInput != null) dummyInput.close();
    }

    /**
     * Benchmark linear layer forward pass.
     */
    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public Tensor linearForward() {
        return model.forward(dummyInput);
    }

    /**
     * Benchmark with varying batch sizes.
     */
    @org.openjdk.jmh.annotations.Param({"1", "8", "16", "32"})
    public int batchSizeParam;

    @org.openjdk.jmh.annotations.Setup
    public void setupBatchSize() {
        dummyInput = torch.randn(new long[]{batchSizeParam, seqLength, hiddenSize});
    }

    @org.openjdk.jmh.annotations.Benchmark
    public Tensor batchedForward() {
        return model.forward(dummyInput);
    }

    /**
     * Benchmark with varying sequence lengths.
     */
    @org.openjdk.jmh.annotations.Param({"128", "512", "2048", "4096"})
    public int seqLengthParam;

    @org.openjdk.jmh.annotations.Setup
    public void setupSeqLength() {
        seqLength = seqLengthParam;
        dummyInput = torch.randn(new long[]{batchSize, seqLength, hiddenSize});
    }

    @org.openjdk.jmh.annotations.Benchmark
    public Tensor seqForward() {
        return model.forward(dummyInput);
    }

    /**
     * Benchmark matrix operations.
     */
    @org.openjdk.jmh.annotations.Benchmark
    public Tensor matmul() {
        return torch.matmul(dummyInput, dummyInput.transpose(1, 2));
    }

    /**
     * Main method to run benchmarks.
     */
    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
