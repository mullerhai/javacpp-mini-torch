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
package org.bytedeco.pytorch.recommend.benchmark;

import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.recommend.monitoring.RecommendMetricsCollector;
import org.bytedeco.pytorch.Tensor;

import java.util.concurrent.TimeUnit;

/**
 * JMH benchmarks for recommendation system models.
 */
@org.openjdk.jmh.annotations.State(org.openjdk.jmh.annotations.Scope.Thread)
@org.openjdk.jmh.annotations.OutputTimeUnit(TimeUnit.MILLISECONDS)
@org.openjdk.jmh.annotations.Measurement(iterations = 5, time = 1)
@org.openjdk.jmh.annotations.Warmup(iterations = 3, time = 1)
public class RecommendBenchmark {

    @org.openjdk.jmh.annotations.Param({"1024", "4096", "16384"})
    public int vocabSize;

    @org.openjdk.jmh.annotations.Param({"128", "512", "2048"})
    public int embedDim;

    @org.openjdk.jmh.annotations.Param({"16", "64", "256"})
    public int batchSize;

    private Tensor denseInput;
    private Tensor sparseInput;
    private RecommendMetricsCollector metricsCollector;

    @org.openjdk.jmh.annotations.Setup
    public void setup() {
        denseInput = torch.randn(new long[]{batchSize, 10});
        sparseInput = torch.randint(0, vocabSize, new long[]{batchSize, 20});
        metricsCollector = RecommendMetricsCollector.builder()
                .name("recommend-benchmark")
                .enableAucTracking(true)
                .enableLatencyTracking(true)
                .build();
    }

    @org.openjdk.jmh.annotations.TearDown
    public void teardown() {
        if (denseInput != null) denseInput.close();
        if (sparseInput != null) sparseInput.close();
        if (metricsCollector != null) metricsCollector.close();
    }

    @org.openjdk.jmh.annotations.Benchmark
    public void ctrMetrics() {
        metricsCollector.recordCtr(true, true);
    }

    @org.openjdk.jmh.annotations.Benchmark
    public void rankingMetrics() {
        metricsCollector.recordRankingQuality(0.75, 0.65, 0.55, 0.85);
    }

    @org.openjdk.jmh.annotations.Benchmark
    public String prometheusExport() {
        return metricsCollector.exportPrometheus();
    }

    @org.openjdk.jmh.annotations.Benchmark
    public String jsonExport() {
        return metricsCollector.exportJson();
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.AverageTime)
    public double batchLatency() {
        long start = System.nanoTime();
        metricsCollector.recordPrediction(batchSize, 1.0);
        return (System.nanoTime() - start) / 1_000_000.0;
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double embeddingLookup() {
        try {
            Module embedding = torch.nn.embedding(vocabSize, embedDim);
            return embedding.forward(sparseInput).sum().item_double();
        } catch (Exception e) {
            return 0.0;
        }
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double linearLayer() {
        try {
            Module linear = torch.nn.linear(10, 128);
            return linear.forward(denseInput).sum().item_double();
        } catch (Exception e) {
            return 0.0;
        }
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double mlpForward() {
        try {
            Module mlp = torch.nn.Sequential(
                torch.nn.linear(10, 256),
                torch.nn.relu(),
                torch.nn.linear(256, 128),
                torch.nn.relu(),
                torch.nn.linear(128, 1)
            );
            return mlp.forward(denseInput).sum().item_double();
        } catch (Exception e) {
            return 0.0;
        }
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
