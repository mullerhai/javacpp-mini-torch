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
package org.bytedeco.pytorch.mlops.tracking;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.Closeable;
import java.nio.file.Path;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * Enterprise-grade model tracking and experiment management platform.
 *
 * <p>Features:
 * <ul>
 *   <li>Experiment and run management</li>
 *   <li>Metrics and parameter logging</li>
 *   <li>Model versioning and registry</li>
 *   <li>Artifact tracking</li>
 *   <li>Distributed training support</li>
 * </ul>
 *
 * <p>Reference: MLflow, ClearML, Weights & Biases
 *
 * <pre>{@code
 * // Start tracking
 * ModelTracker tracker = ModelTracker.builder()
 *     .trackingUri("file:/mlops/tracking")
 *     .experimentName("image-classification")
 *     .build();
 *
 * // Log experiment
 * try (Run run = tracker.startRun("vgg16-training")) {
 *     run.logParam("learning_rate", 0.001);
 *     run.logParam("batch_size", 32);
 *     run.logMetric("accuracy", 0.95);
 *     run.logModel(model, "vgg16");
 * }
 * }</pre>
 */
public class ModelTracker implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final String trackingUri;
    private final String experimentName;
    private final String artifactRoot;
    private final int maxRunsInMemory;

    // State
    private final Map<String, Experiment> experiments = new ConcurrentHashMap<>();
    private final Map<String, Run> activeRuns = new ConcurrentHashMap<>();
    private final ExecutorService executor;

    // Statistics
    private final AtomicLong totalRuns = new AtomicLong(0);
    private final AtomicLong totalMetrics = new AtomicLong(0);
    private final AtomicLong totalModels = new AtomicLong(0);

    public static Builder builder() {
        return new Builder();
    }

    private ModelTracker(Builder builder) {
        this.trackingUri = builder.trackingUri;
        this.experimentName = builder.experimentName;
        this.artifactRoot = builder.artifactRoot;
        this.maxRunsInMemory = builder.maxRunsInMemory;

        this.executor = Executors.newScheduledThreadPool(4);

        // Initialize default experiment
        getOrCreateExperiment("Default");
    }

    // ============= Experiment Management =============

    /**
     * Get or create an experiment.
     */
    public Experiment getOrCreateExperiment(String name) {
        return experiments.computeIfAbsent(name, n -> new Experiment(n, this));
    }

    /**
     * Get experiment by name.
     */
    public Optional<Experiment> getExperiment(String name) {
        return Optional.ofNullable(experiments.get(name));
    }

    /**
     * List all experiments.
     */
    public List<Experiment> listExperiments() {
        return new ArrayList<>(experiments.values());
    }

    /**
     * Set active experiment.
     */
    public void setActiveExperiment(String name) {
        getOrCreateExperiment(name);
    }

    // ============= Run Management =============

    /**
     * Start a new run.
     */
    public Run startRun(String runName) {
        return startRun(runName, experimentName);
    }

    /**
     * Start a new run in specified experiment.
     */
    public Run startRun(String runName, String experimentName) {
        Experiment exp = getOrCreateExperiment(experimentName);
        Run run = new Run(runName, exp, this);
        activeRuns.put(run.getRunId(), run);
        totalRuns.incrementAndGet();
        return run;
    }

    /**
     * Get active run.
     */
    public Optional<Run> getActiveRun() {
        return activeRuns.isEmpty() ? Optional.empty() :
                Optional.of(activeRuns.values().iterator().next());
    }

    /**
     * End a run.
     */
    public void endRun(String runId) {
        Run run = activeRuns.remove(runId);
        if (run != null) {
            run.end();
        }
    }

    /**
     * Search runs.
     */
    public List<Run> searchRuns(String experimentName, String filter) {
        Experiment exp = experiments.get(experimentName);
        if (exp == null) return Collections.emptyList();
        return exp.searchRuns(filter);
    }

    // ============= Model Registry =============

    /**
     * Register a model.
     */
    public RegisteredModel registerModel(String name, String version, Path modelPath) {
        RegisteredModel model = new RegisteredModel(name, version, modelPath, this);
        totalModels.incrementAndGet();
        return model;
    }

    /**
     * Get latest model version.
     */
    public Optional<RegisteredModel> getLatestModel(String name) {
        // Implementation would search storage
        return Optional.empty();
    }

    // ============= Statistics =============

    public ModelTrackerStats getStats() {
        return new ModelTrackerStats(
                trackingUri,
                experimentName,
                experiments.size(),
                activeRuns.size(),
                totalRuns.get(),
                totalMetrics.get(),
                totalModels.get()
        );
    }

    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // End all active runs
        activeRuns.values().forEach(Run::end);
        activeRuns.clear();

        executor.shutdown();
        try {
            executor.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }

        System.out.printf(
                "[ModelTracker] Closed: experiments=%d, runs=%d, models=%d%n",
                experiments.size(), totalRuns.get(), totalModels.get());
    }

    // ============= Inner Types =============

    /**
     * Experiment - a collection of runs.
     */
    public static class Experiment implements Closeable {
        private final String name;
        private final ModelTracker tracker;
        private final List<Run> runs = Collections.synchronizedList(new ArrayList<>());
        private final Instant createdAt;
        private String description;
        private Map<String, String> tags;

        Experiment(String name, ModelTracker tracker) {
            this.name = name;
            this.tracker = tracker;
            this.createdAt = Instant.now();
            this.tags = new ConcurrentHashMap<>();
        }

        public String name() { return name; }
        public Instant createdAt() { return createdAt; }
        public String description() { return description; }
        public void setDescription(String desc) { this.description = desc; }
        public Map<String, String> tags() { return tags; }

        public void setTag(String key, String value) {
            tags.put(key, value);
        }

        public Run startRun(String runName) {
            Run run = new Run(runName, this, tracker);
            runs.add(run);
            tracker.activeRuns.put(run.getRunId(), run);
            return run;
        }

        public List<Run> getRuns() {
            return new ArrayList<>(runs);
        }

        public List<Run> searchRuns(String filter) {
            // Simple filter implementation
            return runs.stream()
                    .filter(r -> filter == null || filter.isEmpty() ||
                            r.getRunName().contains(filter))
                    .toList();
        }

        public Run getBestRun(String metric) {
            return runs.stream()
                    .filter(r -> r.getMetrics().containsKey(metric))
                    .max(Comparator.comparingDouble(r -> r.getMetrics().get(metric).latest()))
                    .orElse(null);
        }

        @Override
        public void close() {}
    }

    /**
     * Run - a single execution of a model training/inference.
     */
    public static class Run implements AutoCloseable {
        private final String runId;
        private final String runName;
        private final Experiment experiment;
        private final ModelTracker tracker;
        private final Instant startTime;
        private volatile Instant endTime;
        private volatile RunStatus status;

        // Data
        private final Map<String, Object> params = new ConcurrentHashMap<>();
        private final Map<String, Metric> metrics = new ConcurrentHashMap<>();
        private final Map<String, Artifact> artifacts = new ConcurrentHashMap<>();
        private final Map<String, String> tags = new ConcurrentHashMap<>();

        Run(String runName, Experiment experiment, ModelTracker tracker) {
            this.runId = UUID.randomUUID().toString();
            this.runName = runName;
            this.experiment = experiment;
            this.tracker = tracker;
            this.startTime = Instant.now();
            this.status = RunStatus.RUNNING;
        }

        public String getRunId() { return runId; }
        public String getRunName() { return runName; }
        public Experiment getExperiment() { return experiment; }
        public Instant getStartTime() { return startTime; }
        public Instant getEndTime() { return endTime; }
        public RunStatus getStatus() { return status; }
        public Map<String, Object> getParams() { return params; }
        public Map<String, Metric> getMetrics() { return metrics; }
        public Map<String, Artifact> getArtifacts() { return artifacts; }
        public Map<String, String> getTags() { return tags; }

        public double getDurationSeconds() {
            Instant end = endTime != null ? endTime : Instant.now();
            return (end.toEpochMilli() - startTime.toEpochMilli()) / 1000.0;
        }

        // ============= Logging Methods =============

        /**
         * Log a parameter.
         */
        public void logParam(String key, Object value) {
            params.put(key, value);
        }

        /**
         * Log multiple parameters.
         */
        public void logParams(Map<String, Object> params) {
            this.params.putAll(params);
        }

        /**
         * Log a metric.
         */
        public void logMetric(String key, double value) {
            logMetric(key, value, 0, null);
        }

        /**
         * Log a metric with step.
         */
        public void logMetric(String key, double value, long step) {
            logMetric(key, value, step, null);
        }

        /**
         * Log a metric with timestamp.
         */
        public void logMetric(String key, double value, long step, Instant timestamp) {
            Metric metric = metrics.computeIfAbsent(key, k -> new Metric(key));
            metric.add(value, step, timestamp != null ? timestamp : Instant.now());
            tracker.totalMetrics.incrementAndGet();
        }

        /**
         * Log a metric batch.
         */
        public void logMetricBatch(String key, List<Double> values, List<Long> steps) {
            for (int i = 0; i < values.size(); i++) {
                long step = steps != null && i < steps.size() ? steps.get(i) : i;
                logMetric(key, values.get(i), step);
            }
        }

        /**
         * Log an artifact.
         */
        public void logArtifact(String name, Path path) {
            artifacts.put(name, new Artifact(name, path, ArtifactType.FILE));
        }

        /**
         * Log a model as artifact.
         */
        public void logModel(Object model, String name) {
            artifacts.put(name, new Artifact(name, null, ArtifactType.MODEL));
        }

        /**
         * Log a tensor as artifact.
         */
        public void logTensor(String name, Tensor tensor) {
            artifacts.put(name, new Artifact(name, null, ArtifactType.TENSOR));
        }

        /**
         * Log a DataFrame as artifact.
         */
        public void logDataFrame(String name, DataFrame df) {
            artifacts.put(name, new Artifact(name, null, ArtifactType.DATAFRAME));
        }

        /**
         * Set a tag.
         */
        public void setTag(String key, String value) {
            tags.put(key, value);
        }

        // ============= Status Management =============

        /**
         * Mark run as completed.
         */
        public void end() {
            end(RunStatus.FINISHED);
        }

        /**
         * Mark run as completed with status.
         */
        public void end(RunStatus status) {
            this.endTime = Instant.now();
            this.status = status;
        }

        /**
         * Mark run as failed.
         */
        public void fail() {
            end(RunStatus.FAILED);
        }

        /**
         * Mark run as killed.
         */
        public void kill() {
            end(RunStatus.KILLED);
        }

        @Override
        public void close() {
            if (status == RunStatus.RUNNING) {
                end();
            }
        }
    }

    /**
     * Metric with history.
     */
    public static class Metric {
        private final String key;
        private final List<Double> values = Collections.synchronizedList(new ArrayList<>());
        private final List<Long> steps = Collections.synchronizedList(new ArrayList<>());
        private final List<Instant> timestamps = Collections.synchronizedList(new ArrayList<>());

        Metric(String key) {
            this.key = key;
        }

        public String key() { return key; }
        public List<Double> values() { return new ArrayList<>(values); }
        public List<Long> steps() { return new ArrayList<>(steps); }
        public List<Instant> timestamps() { return new ArrayList<>(timestamps); }

        public void add(double value, long step, Instant timestamp) {
            values.add(value);
            steps.add(step);
            timestamps.add(timestamp);
        }

        public double latest() {
            return values.isEmpty() ? 0 : values.get(values.size() - 1);
        }

        public double min() {
            return values.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        }

        public double max() {
            return values.stream().mapToDouble(Double::doubleValue).max().orElse(0);
        }

        public double mean() {
            return values.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        }
    }

    /**
     * Artifact.
     */
    public static class Artifact {
        private final String name;
        private final Path path;
        private final ArtifactType type;
        private final Instant createdAt;
        private long sizeBytes;

        Artifact(String name, Path path, ArtifactType type) {
            this.name = name;
            this.path = path;
            this.type = type;
            this.createdAt = Instant.now();
        }

        public String name() { return name; }
        public Path path() { return path; }
        public ArtifactType type() { return type; }
        public Instant createdAt() { return createdAt; }
        public long sizeBytes() { return sizeBytes; }
    }

    /**
     * Artifact types.
     */
    public enum ArtifactType {
        FILE,
        MODEL,
        TENSOR,
        DATAFRAME,
        IMAGE,
        TABLE
    }

    /**
     * Run status.
     */
    public enum RunStatus {
        RUNNING,
        FINISHED,
        FAILED,
        KILLED
    }

    /**
     * Registered model.
     */
    public static class RegisteredModel {
        private final String name;
        private final String version;
        private final Path path;
        private final ModelTracker tracker;
        private final Instant createdAt;
        private String stage;
        private String description;

        RegisteredModel(String name, String version, Path path, ModelTracker tracker) {
            this.name = name;
            this.version = version;
            this.path = path;
            this.tracker = tracker;
            this.createdAt = Instant.now();
            this.stage = "None";
        }

        public String name() { return name; }
        public String version() { return version; }
        public Path path() { return path; }
        public Instant createdAt() { return createdAt; }
        public String stage() { return stage; }
        public String description() { return description; }

        public void setStage(String stage) { this.stage = stage; }
        public void setDescription(String desc) { this.description = desc; }

        public void promote() {
            switch (stage) {
                case "None" -> setStage("Staging");
                case "Staging" -> setStage("Production");
            }
        }

        public void archive() {
            setStage("Archived");
        }
    }

    /**
     * Statistics.
     */
    public static class ModelTrackerStats {
        public final String trackingUri;
        public final String experimentName;
        public final int numExperiments;
        public final int activeRuns;
        public final long totalRuns;
        public final long totalMetrics;
        public final long totalModels;

        public ModelTrackerStats(String trackingUri, String experimentName,
                                int numExperiments, int activeRuns, long totalRuns,
                                long totalMetrics, long totalModels) {
            this.trackingUri = trackingUri;
            this.experimentName = experimentName;
            this.numExperiments = numExperiments;
            this.activeRuns = activeRuns;
            this.totalRuns = totalRuns;
            this.totalMetrics = totalMetrics;
            this.totalModels = totalModels;
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private String trackingUri = "file:/mlops/tracking";
        private String experimentName = "Default";
        private String artifactRoot = "/mlops/artifacts";
        private int maxRunsInMemory = 1000;

        public Builder trackingUri(String uri) { this.trackingUri = uri; return this; }
        public Builder experimentName(String name) { this.experimentName = name; return this; }
        public Builder artifactRoot(String root) { this.artifactRoot = root; return this; }
        public Builder maxRunsInMemory(int max) { this.maxRunsInMemory = max; return this; }

        public ModelTracker build() {
            return new ModelTracker(this);
        }
    }
}
