/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
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
 * or as provided under the License is would be distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.benchmark;

import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.utils.tensor.TensorConverter;
import org.bytedeco.pytorch.utils.buffer.ZeroCopyBuffer;
import org.bytedeco.pytorch.utils.multimodal.MultiModalDataLoader;
import org.bytedeco.pytorch.utils.stream.StreamProcessor;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.Tensor;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmarks for Utils module performance.
 *
 * <p>Run with:
 * <pre>
 * mvn clean install
 * java -jar target/benchmarks.jar ".*UtilsBenchmark.*" -f 3 -wi 5 -i 10
 * </pre>
 */
@org.openjdk.jmh.annotations.State(org.openjdk.jmh.annotations.Scope.Thread)
@org.openjdk.jmh.annotations.OutputTimeUnit(TimeUnit.MILLISECONDS)
@org.openjdk.jmh.annotations.Measurement(iterations = 5, time = 1)
@org.openjdk.jmh.annotations.Warmup(iterations = 3, time = 1)
public class UtilsBenchmark {

    @org.openjdk.jmh.annotations.Param({"1024", "4096", "16384"})
    public int batchSize;

    private TensorConverter converter;
    private ZeroCopyBuffer zeroCopyBuffer;
    private StreamProcessor streamProcessor;
    private Tensor dummyTensor;

    @org.openjdk.jmh.annotations.Setup
    public void setup() {
        converter = TensorConverter.builder()
                .zeroCopy(true)
                .batchSize(batchSize)
                .build();

        zeroCopyBuffer = ZeroCopyBuffer.builder()
                .medium()
                .build();

        streamProcessor = StreamProcessor.builder()
                .windowSize(100)
                .maxBatchSize(batchSize)
                .build();

        dummyTensor = torch.randn(batchSize, 512);
    }

    @org.openjdk.jmh.annotations.TearDown
    public void teardown() {
        if (converter != null) converter.close();
        if (zeroCopyBuffer != null) zeroCopyBuffer.close();
        if (streamProcessor != null) streamProcessor.close();
        if (dummyTensor != null) dummyTensor.close();
    }

    // ============= TensorConverter Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public TensorConverter.DataType[] tensorConversion() {
        // Just return the enum values as a dummy result
        return TensorConverter.DataType.values();
    }

    // ============= ZeroCopyBuffer Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public long bufferAllocation() {
        try (ZeroCopyBuffer buf = ZeroCopyBuffer.builder().small().build()) {
            return buf.capacity();
        }
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public long bufferIntWrite() {
        for (int i = 0; i < 1000; i++) {
            zeroCopyBuffer.putInt(i * 4, i);
        }
        return 1000;
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public long bufferFloatWrite() {
        for (int i = 0; i < 1000; i++) {
            zeroCopyBuffer.putFloat(i * 4, (float) i);
        }
        return 1000;
    }

    // ============= StreamProcessor Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    public void streamProcessorStats() {
        streamProcessor.getStats();
    }

    // ============= Torch Operation Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double tensorClone() {
        return dummyTensor.clone().sum().item_double();
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double tensorMatmul() {
        Tensor other = torch.randn(512, 256);
        Tensor result = torch.matmul(dummyTensor, other);
        return result.sum().item_double();
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double tensorSoftmax() {
        Tensor result = torch.softmax(dummyTensor, -1);
        return result.sum().item_double();
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double tensorRelu() {
        Tensor result = torch.relu(dummyTensor);
        return result.sum().item_double();
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double tensorAdd() {
        Tensor result = dummyTensor.add(dummyTensor);
        return result.sum().item_double();
    }

    // ============= DataFrame Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public DataFrame dataFrameCreation() {
        // Create a simple DataFrame
        org.bytedeco.pytorch.dataframe.Column col = 
                org.bytedeco.pytorch.dataframe.Column.ofInts("id", batchSize);
        return org.bytedeco.pytorch.dataframe.DataFrame.of(col);
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
