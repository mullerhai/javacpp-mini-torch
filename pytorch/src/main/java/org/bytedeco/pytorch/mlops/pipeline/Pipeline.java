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

import java.io.Closeable;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Consumer;

/**
 * Enterprise-grade pipeline orchestrator for ML workflows.
 *
 * <p>Features:
 * <ul>
 *   <li>DAG-based workflow</li>
 *   <li>Parallel execution</li>
 *   <li>Step caching</li>
 *   <li>Error handling and retry</li>
 *   <li>Conditional branching</li>
 * </ul>
 *
 * <p>Reference: Kubeflow Pipelines, Airflow, Prefect
 *
 * <pre>{@code
 * Pipeline pipeline = Pipeline.builder("training-pipeline")
 *     .step("preprocess", () -> preprocess(data))
 *     .step("train", () -> train(model, data))
 *     .step("evaluate", () -> evaluate(model, test))
 *     .after("train", "preprocess")
 *     .after("evaluate", "train")
 *     .build();
 *
 * PipelineExecutor executor = PipelineExecutor.builder().build();
 * PipelineResult result = executor.execute(pipeline);
 * }</pre>
 */
public class Pipeline implements AutoCloseable {

    public static final String VERSION = "2.0";

    private final String name;
    private final String description;
    private final Map<String, PipelineStep> steps = new LinkedHashMap<>();
    private final Map<String, Set<String>> dependencies = new HashMap<>();
    private final Map<String, Set<String>> dependents = new HashMap<>();
    private final Map<String, Object> artifacts = new ConcurrentHashMap<>();

    private volatile boolean closed;

    public static Builder builder(String name) {
        return new Builder(name);
    }

    private Pipeline(Builder builder) {
        this.name = builder.name;
        this.description = builder.description;

        // Build step graph
        for (Map.Entry<String, PipelineStep> entry : builder.steps.entrySet()) {
            addStep(entry.getKey(), entry.getValue());
        }

        for (Map.Entry<String, List<String>> entry : builder.dependencies.entrySet()) {
            for (String dep : entry.getValue()) {
                after(entry.getKey(), dep);
            }
        }
    }

    private void addStep(String name, PipelineStep step) {
        steps.put(name, step);
        dependencies.put(name, new HashSet<>());
        dependents.put(name, new HashSet<>());
    }

    /**
     * Add a dependency - step runs after dependencies.
     */
    public Pipeline after(String step, String... dependencies) {
        for (String dep : dependencies) {
            this.dependencies.get(step).add(dep);
            this.dependents.get(dep).add(step);
        }
        return this;
    }

    /**
     * Get step by name.
     */
    public Optional<PipelineStep> getStep(String name) {
        return Optional.ofNullable(steps.get(name));
    }

    /**
     * Get all steps.
     */
    public Map<String, PipelineStep> getSteps() {
        return new LinkedHashMap<>(steps);
    }

    /**
     * Get dependencies of a step.
     */
    public Set<String> getDependencies(String stepName) {
        return new HashSet<>(dependencies.getOrDefault(stepName, Collections.emptySet()));
    }

    /**
     * Get dependents of a step.
     */
    public Set<String> getDependents(String stepName) {
        return new HashSet<>(dependents.getOrDefault(stepName, Collections.emptySet()));
    }

    /**
     * Get topological order.
     */
    public List<String> topologicalOrder() {
        List<String> order = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        Set<String> temp = new HashSet<>();

        for (String step : steps.keySet()) {
            if (!visited.contains(step)) {
                topologicalSort(step, visited, temp, order);
            }
        }

        Collections.reverse(order);
        return order;
    }

    private void topologicalSort(String step, Set<String> visited, Set<String> temp, List<String> order) {
        if (temp.contains(step)) {
            throw new IllegalStateException("Cycle detected in pipeline: " + step);
        }

        if (visited.contains(step)) {
            return;
        }

        temp.add(step);

        for (String dep : dependencies.getOrDefault(step, Collections.emptySet())) {
            topologicalSort(dep, visited, temp, order);
        }

        temp.remove(step);
        visited.add(step);
        order.add(step);
    }

    /**
     * Check if pipeline is valid DAG.
     */
    public boolean isValid() {
        try {
            topologicalOrder();
            return true;
        } catch (IllegalStateException e) {
            return false;
        }
    }

    /**
     * Set artifact.
     */
    public void setArtifact(String key, Object value) {
        artifacts.put(key, value);
    }

    /**
     * Get artifact.
     */
    public Object getArtifact(String key) {
        return artifacts.get(key);
    }

    public String name() { return name; }
    public String description() { return description; }

    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        closed = true;
        steps.values().forEach(step -> {
            if (step instanceof Closeable) {
                try { ((Closeable) step).close(); } catch (Exception ignored) {}
            }
        });
    }

    // ============= Pipeline Step =============

    /**
     * Pipeline step.
     */
    public interface PipelineStep extends Runnable, Closeable {
        String getName();
        StepStatus getStatus();
        String getError();
        long getDurationMs();
        Map<String, Object> getOutputs();
        void setInput(String key, Object value);
        Object getInput(String key);
    }

    /**
     * Step status.
     */
    public enum StepStatus {
        PENDING,
        RUNNING,
        SUCCESS,
        FAILED,
        SKIPPED
    }

    /**
     * Simple step implementation.
     */
    public static class SimpleStep implements PipelineStep {
        private final String name;
        private final Runnable action;
        private volatile StepStatus status = StepStatus.PENDING;
        private volatile String error;
        private volatile long durationMs;
        private final Map<String, Object> inputs = new ConcurrentHashMap<>();
        private final Map<String, Object> outputs = new ConcurrentHashMap<>();

        public SimpleStep(String name, Runnable action) {
            this.name = name;
            this.action = action;
        }

        @Override
        public String getName() { return name; }
        @Override
        public StepStatus getStatus() { return status; }
        @Override
        public String getError() { return error; }
        @Override
        public long getDurationMs() { return durationMs; }
        @Override
        public Map<String, Object> getOutputs() { return outputs; }

        @Override
        public void setInput(String key, Object value) {
            inputs.put(key, value);
        }

        @Override
        public Object getInput(String key) {
            return inputs.get(key);
        }

        @Override
        public void run() {
            long start = System.currentTimeMillis();
            status = StepStatus.RUNNING;

            try {
                action.run();
                status = StepStatus.SUCCESS;
            } catch (Exception e) {
                status = StepStatus.FAILED;
                error = e.getMessage();
            }

            durationMs = System.currentTimeMillis() - start;
        }

        @Override
        public void close() {}
    }

    /**
     * Step that returns a value.
     */
    public static class ValueStep<T> implements PipelineStep {
        private final String name;
        private final java.util.function.Supplier<T> action;
        private volatile T result;
        private volatile StepStatus status = StepStatus.PENDING;
        private volatile String error;
        private volatile long durationMs;
        private final Map<String, Object> inputs = new ConcurrentHashMap<>();
        private final Map<String, Object> outputs = new ConcurrentHashMap<>();

        public ValueStep(String name, java.util.function.Supplier<T> action) {
            this.name = name;
            this.action = action;
        }

        @Override
        public String getName() { return name; }
        @Override
        public StepStatus getStatus() { return status; }
        @Override
        public String getError() { return error; }
        @Override
        public long getDurationMs() { return durationMs; }
        @Override
        public Map<String, Object> getOutputs() { return outputs; }

        @Override
        public void setInput(String key, Object value) {
            inputs.put(key, value);
        }

        @Override
        public Object getInput(String key) {
            return inputs.get(key);
        }

        @Override
        public void run() {
            long start = System.currentTimeMillis();
            status = StepStatus.RUNNING;

            try {
                result = action.get();
                outputs.put("result", result);
                status = StepStatus.SUCCESS;
            } catch (Exception e) {
                status = StepStatus.FAILED;
                error = e.getMessage();
            }

            durationMs = System.currentTimeMillis() - start;
        }

        public T getResult() { return result; }

        @Override
        public void close() {}
    }

    /**
     * Builder.
     */
    public static class Builder {
        private final String name;
        private String description = "";
        private final Map<String, PipelineStep> steps = new LinkedHashMap<>();
        private final Map<String, List<String>> dependencies = new HashMap<>();

        Builder(String name) {
            this.name = name;
        }

        public Builder description(String desc) { this.description = desc; return this; }

        public Builder step(String name, Runnable action) {
            steps.put(name, new SimpleStep(name, action));
            return this;
        }

        public <T> Builder step(String name, java.util.function.Supplier<T> action,
                               java.util.function.Consumer<T> output) {
            steps.put(name, new ValueStep<>(name, action));
            return this;
        }

        public Builder after(String step, String... dependencies) {
            this.dependencies.put(step, Arrays.asList(dependencies));
            return this;
        }

        public Pipeline build() {
            if (!isValid()) {
                throw new IllegalStateException("Invalid pipeline configuration");
            }
            return new Pipeline(this);
        }

        private boolean isValid() {
            // Basic validation
            return !steps.isEmpty();
        }
    }

    /**
     * Pipeline result.
     */
    public static class PipelineResult {
        private final String pipelineName;
        private final Map<String, StepResult> stepResults;
        private final long totalDurationMs;
        private final boolean success;

        public PipelineResult(String pipelineName, Map<String, StepResult> stepResults,
                            long totalDurationMs) {
            this.pipelineName = pipelineName;
            this.stepResults = stepResults;
            this.totalDurationMs = totalDurationMs;
            this.success = stepResults.values().stream()
                    .allMatch(r -> r.status == StepStatus.SUCCESS);
        }

        public String pipelineName() { return pipelineName; }
        public Map<String, StepResult> stepResults() { return stepResults; }
        public long totalDurationMs() { return totalDurationMs; }
        public boolean success() { return success; }

        public StepResult getStepResult(String stepName) {
            return stepResults.get(stepName);
        }
    }

    /**
     * Step result.
     */
    public static class StepResult {
        public final String stepName;
        public final StepStatus status;
        public final String error;
        public final long durationMs;
        public final Map<String, Object> outputs;

        public StepResult(String stepName, StepStatus status, String error,
                        long durationMs, Map<String, Object> outputs) {
            this.stepName = stepName;
            this.status = status;
            this.error = error;
            this.durationMs = durationMs;
            this.outputs = outputs;
        }
    }
}
