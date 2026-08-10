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
package org.bytedeco.pytorch.mlops.pipeline;
import org.bytedeco.pytorch.distributed.*;
import org.bytedeco.pytorch.c10.*;

import java.io.Closeable;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Pipeline executor with parallel execution support.
 */
public class PipelineExecutor implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    private final int maxParallelism;
    private final boolean enableCaching;
    private final int maxRetries;
    private final ExecutorService executor;
    private final ScheduledExecutorService scheduler;

    private final Map<String, PipelineResult.StepResult> stepResults = new ConcurrentHashMap<>();
    private final Map<String, Object> sharedArtifacts = new ConcurrentHashMap<>();

    public static Builder builder() {
        return new Builder();
    }

    private PipelineExecutor(Builder builder) {
        this.maxParallelism = builder.maxParallelism;
        this.enableCaching = builder.enableCaching;
        this.maxRetries = builder.maxRetries;

        this.executor = Executors.newFixedThreadPool(builder.numThreads);
        this.scheduler = Executors.newScheduledThreadPool(2);
    }

    /**
     * Execute a pipeline.
     */
    public Pipeline.PipelineResult execute(Pipeline pipeline) {
        return execute(pipeline, null);
    }

    /**
     * Execute a pipeline with initial context.
     */
    public Pipeline.PipelineResult execute(Pipeline pipeline, Map<String, Object> initialContext) {
        long startTime = System.currentTimeMillis();

        // Get execution order
        List<String> order = pipeline.topologicalOrder();

        // Initialize context
        Map<String, Object> context = new ConcurrentHashMap<>();
        if (initialContext != null) {
            context.putAll(initialContext);
        }

        // Track completed steps
        Set<String> completed = Collections.synchronizedSet(new HashSet<>());
        Map<String, Future<Pipeline.PipelineResult.StepResult>> futures = new ConcurrentHashMap<>();

        // Track which steps are waiting for dependencies
        Map<String, Set<String>> pending = new ConcurrentHashMap<>();
        for (String stepName : order) {
            pending.put(stepName, new HashSet<>(pipeline.getDependencies(stepName)));
        }

        // Execute steps as dependencies are satisfied
        while (!pending.isEmpty()) {
            // Find steps ready to run
            List<String> ready = new ArrayList<>();
            for (Map.Entry<String, Set<String>> entry : pending.entrySet()) {
                if (entry.getValue().isEmpty() && !futures.containsKey(entry.getKey())) {
                    ready.add(entry.getKey());
                }
            }

            if (ready.isEmpty()) {
                // Deadlock - check for cycles
                if (pending.values().stream().noneMatch(Set::isEmpty)) {
                    throw new IllegalStateException("Pipeline execution deadlock - possible cycle");
                }
            }

            // Limit parallelism
            int toRun = Math.min(ready.size(), maxParallelism - futures.size());
            for (int i = 0; i < toRun; i++) {
                String stepName = ready.get(i);
                Pipeline.PipelineStep step = pipeline.getStep(stepName).orElse(null);

                if (step != null) {
                    // Set inputs from context
                    for (Map.Entry<String, Object> entry : context.entrySet()) {
                        step.setInput(entry.getKey(), entry.getValue());
                    }

                    // Submit step
                    final String name = stepName;
                    futures.put(stepName, executor.submit(() -> executeStep(name, step)));
                }
            }
        }

        // Wait for all steps to complete
        Map<String, Pipeline.PipelineResult.StepResult> results = new LinkedHashMap<>();
        for (String stepName : order) {
            Future<Pipeline.PipelineResult.StepResult> future = futures.get(stepName);
            if (future != null) {
                try {
                    Pipeline.PipelineResult.StepResult result = future.get();
                    results.put(stepName, result);

                    // Store outputs in context
                    if (result.outputs != null) {
                        context.putAll(result.outputs);
                    }

                } catch (Exception e) {
                    results.put(stepName, new Pipeline.PipelineResult.StepResult(
                            stepName, Pipeline.StepStatus.FAILED, e.getMessage(), 0, null));
                }
            }
        }

        long duration = System.currentTimeMillis() - startTime;

        return new Pipeline.PipelineResult(pipeline.name(), results, duration);
    }

    /**
     * Execute a single step with retry logic.
     */
    private Pipeline.PipelineResult.StepResult executeStep(String name, Pipeline.PipelineStep step) {
        int attempts = 0;
        Exception lastError = null;

        while (attempts <= maxRetries) {
            try {
                step.run();

                if (step.getStatus() == Pipeline.StepStatus.SUCCESS) {
                    return new Pipeline.PipelineResult.StepResult(
                            name,
                            step.getStatus(),
                            step.getError(),
                            step.getDurationMs(),
                            step.getOutputs()
                    );
                } else if (step.getStatus() == Pipeline.StepStatus.FAILED) {
                    lastError = new Exception(step.getError());
                }
            } catch (Exception e) {
                lastError = e;
            }

            attempts++;
            if (attempts <= maxRetries) {
                try {
                    Thread.sleep((long) Math.pow(2, attempts) * 1000);  // Exponential backoff
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }

        return new Pipeline.PipelineResult.StepResult(
                name,
                Pipeline.StepStatus.FAILED,
                lastError != null ? lastError.getMessage() : "Max retries exceeded",
                step.getDurationMs(),
                null
        );
    }

    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        executor.shutdown();
        scheduler.shutdown();

        try {
            executor.awaitTermination(10, TimeUnit.SECONDS);
            scheduler.awaitTermination(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            executor.shutdownNow();
            scheduler.shutdownNow();
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private int numThreads = 4;
        private int maxParallelism = 4;
        private boolean enableCaching = true;
        private int maxRetries = 3;

        public Builder numThreads(int threads) { this.numThreads = threads; return this; }
        public Builder maxParallelism(int max) { this.maxParallelism = max; return this; }
        public Builder enableCaching(boolean enable) { this.enableCaching = enable; return this; }
        public Builder maxRetries(int retries) { this.maxRetries = retries; return this; }

        public PipelineExecutor build() {
            return new PipelineExecutor(this);
        }
    }
}
