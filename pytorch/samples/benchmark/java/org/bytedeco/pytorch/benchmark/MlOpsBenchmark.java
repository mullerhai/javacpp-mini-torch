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

import org.bytedeco.pytorch.mlops.tracking.ModelTracker;
import org.bytedeco.pytorch.mlops.experiment.HyperparameterOptimizer;
import org.bytedeco.pytorch.mlops.experiment.SearchSpace;
import org.bytedeco.pytorch.mlops.serving.ModelServer;
import org.bytedeco.pytorch.mlops.pipeline.Pipeline;
import org.bytedeco.pytorch.mlops.pipeline.PipelineExecutor;
import org.bytedeco.pytorch.mlops.monitoring.ResourceMonitor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.Tensor;

import java.util.concurrent.TimeUnit;

/**
 * JMH benchmarks for MLOps platform components.
 *
 * <p>Run with:
 * <pre>
 * mvn clean install
 * java -jar target/benchmarks.jar ".*MlOpsBenchmark.*" -f 3 -wi 3 -i 5
 * </pre>
 */
@org.openjdk.jmh.annotations.State(org.openjdk.jmh.annotations.Scope.Thread)
@org.openjdk.jmh.annotations.OutputTimeUnit(TimeUnit.MILLISECONDS)
@org.openjdk.jmh.annotations.Measurement(iterations = 5, time = 1)
@org.openjdk.jmh.annotations.Warmup(iterations = 3, time = 1)
public class MlOpsBenchmark {

    @org.openjdk.jmh.annotations.Param({"100", "1000", "10000"})
    public int numMetrics;

    private ModelTracker modelTracker;
    private ResourceMonitor resourceMonitor;
    private ModelServer modelServer;
    private HyperparameterOptimizer optimizer;
    private Pipeline pipeline;
    private PipelineExecutor executor;

    @org.openjdk.jmh.annotations.Setup
    public void setup() {
        // Initialize components
        modelTracker = ModelTracker.builder()
                .experimentName("benchmark")
                .build();

        resourceMonitor = ResourceMonitor.builder()
                .collectInterval(1, TimeUnit.SECONDS)
                .build();

        modelServer = ModelServer.builder()
                .port(8080)
                .maxBatchSize(32)
                .build();

        // Initialize optimizer
        SearchSpace space = new SearchSpace();
        space.addLogUniform("lr", 1e-4, 1e-1);
        space.add("batch_size", 16, 32, 64, 128);

        optimizer = HyperparameterOptimizer.builder()
                .searchSpace(space)
                .maxTrials(10)
                .maxConcurrency(2)
                .build();

        // Initialize pipeline
        pipeline = Pipeline.builder("test-pipeline")
                .step("preprocess", () -> { /* preprocess */ })
                .step("train", () -> { /* train */ })
                .step("evaluate", () -> { /* evaluate */ })
                .after("train", "preprocess")
                .after("evaluate", "train")
                .build();

        executor = PipelineExecutor.builder()
                .numThreads(4)
                .build();
    }

    @org.openjdk.jmh.annotations.TearDown
    public void teardown() {
        if (modelTracker != null) modelTracker.close();
        if (resourceMonitor != null) resourceMonitor.close();
        if (modelServer != null) modelServer.close();
        if (optimizer != null) optimizer.close();
        if (pipeline != null) pipeline.close();
        if (executor != null) executor.close();
    }

    // ============= ModelTracker Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    public void modelTrackerStartRun() {
        try (ModelTracker.Run run = modelTracker.startRun("test-run")) {
            // Log some metrics
            for (int i = 0; i < 100; i++) {
                run.logMetric("loss", Math.random(), i);
                run.logMetric("accuracy", Math.random(), i);
            }
        }
    }

    @org.openjdk.jmh.annotations.Benchmark
    public void modelTrackerLogMetrics() {
        try (ModelTracker.Run run = modelTracker.startRun("test-run")) {
            for (int i = 0; i < numMetrics; i++) {
                run.logMetric("metric_" + i, Math.random());
            }
        }
    }

    @org.openjdk.jmh.annotations.Benchmark
    public ModelTracker.ModelTrackerStats modelTrackerStats() {
        return modelTracker.getStats();
    }

    // ============= ResourceMonitor Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    public void resourceMonitorRegisterGauge() {
        resourceMonitor.registerGauge("custom_gauge_" + System.nanoTime(),
                () -> Math.random() * 100);
    }

    @org.openjdk.jmh.annotations.Benchmark
    public void resourceMonitorRecordCounter() {
        resourceMonitor.incrementCounter("custom_counter_" + System.nanoTime());
    }

    @org.openjdk.jmh.annotations.Benchmark
    public String resourceMonitorPrometheusExport() {
        return resourceMonitor.exportPrometheus();
    }

    @org.openjdk.jmh.annotations.Benchmark
    public ResourceMonitor.ResourceMonitorStats resourceMonitorStats() {
        return resourceMonitor.getStats();
    }

    // ============= ModelServer Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public void modelServerRegisterModel() {
        // Register a dummy model
        // Note: In real scenario, this would load an actual model
    }

    @org.openjdk.jmh.annotations.Benchmark
    public ModelServer.ModelServerStats modelServerStats() {
        return modelServer.getStats();
    }

    // ============= Pipeline Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    public void pipelineTopologicalSort() {
        pipeline.topologicalOrder();
    }

    @org.openjdk.jmh.annotations.Benchmark
    public Pipeline.PipelineResult pipelineExecute() {
        return executor.execute(pipeline);
    }

    // ============= Torch Operation Benchmarks =============

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double tensorMatmul() {
        Tensor a = torch.randn(512, 512);
        Tensor b = torch.randn(512, 512);
        return torch.matmul(a, b).sum().item_double();
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double tensorSoftmax() {
        Tensor x = torch.randn(1, 1000);
        return torch.softmax(x, -1).sum().item_double();
    }

    @org.openjdk.jmh.annotations.Benchmark
    @org.openjdk.jmh.annotations.Mode(org.openjdk.jmh.annotations.Mode.Throughput)
    public double tensorRelu() {
        Tensor x = torch.randn(1024, 1024);
        return torch.relu(x).sum().item_double();
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
