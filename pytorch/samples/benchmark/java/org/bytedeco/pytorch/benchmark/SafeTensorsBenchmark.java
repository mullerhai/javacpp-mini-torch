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

import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.safetensors.SafeTensorsLoader;
import org.bytedeco.pytorch.llm.transformers.processor.ImageProcessor;
import org.bytedeco.pytorch.llm.transformers.processor.AudioProcessor;
import org.bytedeco.pytorch.llm.transformers.processor.VideoProcessor;
import org.bytedeco.pytorch.llm.modules.LongContextRoPE;
import org.bytedeco.pytorch.Tensor;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmarks for SafeTensors loading performance.
 */
@org.openjdk.jmh.annotations.State(org.openjdk.jmh.annotations.Scope.Thread)
@org.openjdk.jmh.annotations.OutputTimeUnit(TimeUnit.MILLISECONDS)
@org.openjdk.jmh.annotations.Measurement(iterations = 5, time = 1)
@org.openjdk.jmh.annotations.Warmup(iterations = 3, time = 1)
public class SafeTensorsBenchmark {

    private File tempFile;
    private SafeTensorsLoader loader;

    @org.openjdk.jmh.annotations.Setup
    public void setup() throws IOException {
        // Create a test file with random tensors
        Map<String, Tensor> tensors = new java.util.LinkedHashMap<>();
        tensors.put("embedding", createRandomTensor(1000, 4096));
        tensors.put("layer1.weight", createRandomTensor(4096, 4096));
        tensors.put("layer1.bias", createRandomTensor(4096));
        tensors.put("layer2.weight", createRandomTensor(4096, 4096));
        tensors.put("layer2.bias", createRandomTensor(4096));

        tempFile = File.createTempFile("bench", ".safetensors");
        SafeTensors.save(tensors, tempFile);

        loader = SafeTensorsLoader.builder()
                .zeroCopy(true)
                .build();
    }

    @org.openjdk.jmh.annotations.TearDown
    public void teardown() {
        if (tempFile != null && tempFile.exists()) {
            tempFile.delete();
        }
        if (loader != null) {
            loader.close();
        }
    }

    private Tensor createRandomTensor(long... shape) {
        return org.bytedeco.pytorch.global.torch.rand(shape);
    }

    /**
     * Benchmark SafeTensors.loadAsTensors.
     */
    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public Map<String, Tensor> loadAsTensors() throws IOException {
        return SafeTensors.loadAsTensors(tempFile, false);
    }

    /**
     * Benchmark SafeTensorsLoader.
     */
    @org.openjdk.jmh.annotations.Benchmark
    public org.bytedeco.pytorch.data.safetensors.SafeTensorsLoader.SafeTensorsLoadResult loaderLoad() throws IOException {
        return loader.load(tempFile.toPath());
    }

    /**
     * Benchmark async loading.
     */
    @org.openjdk.jmh.annotations.Benchmark
    public java.util.concurrent.CompletableFuture<org.bytedeco.pytorch.data.safetensors.SafeTensorsLoader.SafeTensorsLoadResult> asyncLoad() throws Exception {
        SafeTensorsLoader asyncLoader = SafeTensorsLoader.builder()
                .asyncLoad(true)
                .numThreads(4)
                .build();
        return asyncLoader.loadAsync(tempFile.toPath());
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
