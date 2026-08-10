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
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.benchmark;

import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.vision.processing.ImageProcessor;
import org.bytedeco.pytorch.vision.processing.VideoProcessor;
import org.bytedeco.pytorch.audio.processing.AudioProcessor;
import org.bytedeco.pytorch.feature.MultiModalFeatureStore;
import org.bytedeco.pytorch.Tensor;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * JMH benchmarks for Vision, Audio, and Feature modules.
 *
 * <p>Run with:
 * <pre>
 * mvn clean install
 * java -jar target/benchmarks.jar ".*MultimodalBenchmark.*" -f 3 -wi 5 -i 10
 * </pre>
 */
@org.openjdk.jmh.annotations.State(org.openjdk.jmh.annotations.Scope.Thread)
@org.openjdk.jmh.annotations.OutputTimeUnit(TimeUnit.MILLISECONDS)
@org.openjdk.jmh.annotations.Measurement(iterations = 5, time = 1)
@org.openjdk.jmh.annotations.Warmup(iterations = 3, time = 1)
public class MultimodalBenchmark {

    @org.openjdk.jmh.annotations.Param({"32", "64", "128"})
    public int batchSize;

    private ImageProcessor imageProcessor;
    private VideoProcessor videoProcessor;
    private AudioProcessor audioProcessor;
    private MultiModalFeatureStore featureStore;
    private Tensor dummyImage;
    private Tensor dummyEmbedding;

    @org.openjdk.jmh.annotations.Setup
    public void setup() {
        // Initialize processors
        imageProcessor = ImageProcessor.builder()
                .batchSize(batchSize)
                .numWorkers(4)
                .targetSize(224)
                .build();

        videoProcessor = VideoProcessor.builder()
                .numFrames(16)
                .samplingStrategy(VideoProcessor.SamplingStrategy.UNIFORM)
                .targetSize(224)
                .build();

        audioProcessor = AudioProcessor.builder()
                .sampleRate(16000)
                .nMels(80)
                .nFft(512)
                .build();

        featureStore = MultiModalFeatureStore.builder()
                .name("test_store")
                .inMemory()
                .build();

        // Create dummy tensors
        dummyImage = torch.randn(3, 224, 224);
        dummyEmbedding = torch.randn(512);
    }

    @org.openjdk.jmh.annotations.TearDown
    public void teardown() {
        if (imageProcessor != null) imageProcessor.close();
        if (videoProcessor != null) videoProcessor.close();
        if (audioProcessor != null) audioProcessor.close();
        if (featureStore != null) featureStore.close();
        if (dummyImage != null) dummyImage.close();
        if (dummyEmbedding != null) dummyEmbedding.close();
    }

    // ============= ImageProcessor Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    public ImageProcessor.ImageProcessorStats imageProcessorStats() {
        return imageProcessor.getStats();
    }

    // ============= VideoProcessor Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    public VideoProcessor.VideoProcessorStats videoProcessorStats() {
        return videoProcessor.getStats();
    }

    // ============= AudioProcessor Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    public AudioProcessor.AudioProcessorStats audioProcessorStats() {
        return audioProcessor.getStats();
    }

    // ============= FeatureStore Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public void featureStoreWrite() {
        featureStore.putEmbedding("user_" + System.nanoTime(), "emb", dummyEmbedding.clone());
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public void featureStoreRead() {
        featureStore.putEmbedding("user_test", "emb", dummyEmbedding.clone());
        featureStore.getEmbedding("user_test", "emb");
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public void featureStoreMultiModal() {
        String key = "user_" + System.nanoTime();
        featureStore.putImageEmbedding(key, dummyEmbedding.clone());
        featureStore.putAudioEmbedding(key, dummyEmbedding.clone());
        featureStore.putTextEmbedding(key, dummyEmbedding.clone());
        featureStore.getMultiModalEmbeddings(key);
    }

    @org.openjdk.jmh.annotations.Benchmark
    public MultiModalFeatureStore.MultiModalFeatureStoreStats featureStoreStats() {
        return featureStore.getStats();
    }

    // ============= Torch Operation Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double tensorImageNormalize() {
        Tensor mean = torch.tensor(new float[]{0.485f, 0.456f, 0.406f}).view(3, 1, 1);
        Tensor std = torch.tensor(new float[]{0.229f, 0.224f, 0.225f}).view(3, 1, 1);
        return dummyImage.sub(mean).div(std).sum().item_double();
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double tensorImageResize() {
        // Simulate resize
        return dummyImage.upsample_bilinear2d(
                new long[]{3, 224, 224}, false).sum().item_double();
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double tensorEmbeddingConcat() {
        Tensor a = torch.randn(512);
        Tensor b = torch.randn(512);
        Tensor c = torch.randn(512);
        Tensor d = torch.randn(512);
        return torch.cat(new org.bytedeco.pytorch.TensorVector(a, b, c, d), 0).sum().item_double();
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double tensorMelSpectrogram() {
        // Simulate mel spectrogram computation
        Tensor x = torch.randn(batchSize, 80, 100);
        return x.sum().item_double();
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
