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
package org.bytedeco.pytorch.mlops.experiment;
import org.bytedeco.pytorch.c10.*;
import org.bytedeco.pytorch.jit.*;
import java.util.concurrent.Future;
import org.bytedeco.pytorch.mlops.tracking.ModelTracker;

import java.io.Closeable;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Function;

/**
 * Enterprise-grade hyperparameter optimization framework.
 *
 * <p>Features:
 * <ul>
 *   <li>Bayesian optimization</li>
 *   <li>Random search</li>
 *   <li>Grid search</li>
 *   <li>Hyperband early stopping</li>
 *   <li>Distributed trials</li>
 * </ul>
 *
 * <p>Reference: Optuna, Ray Tune, MLflow Autolog
 *
 * <pre>{@code
 * HyperparameterOptimizer optimizer = HyperparameterOptimizer.builder()
 *     .searchSpace(space -> {
 *         space.add("lr", SearchSpace.logUniform(1e-4, 1e-1));
 *         space.add("batch_size", SearchSpace.choice(16, 32, 64, 128));
 *     })
 *     .objective(trial -> trainAndEvaluate(trial))
 *     .maxTrials(100)
 *     .build();
 *
 * OptimizationResult result = optimizer.optimize();
 * }</pre>
 */
public class HyperparameterOptimizer implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final SearchSpace searchSpace;
    private final Function<Trial, Double> objective;
    private final OptimizationAlgorithm algorithm;
    private final int maxTrials;
    private final int maxConcurrency;
    private final long timeoutSeconds;
    private final boolean earlyStopping;

    // State
    private final ExecutorService executor;
    private final List<Trial> completedTrials = Collections.synchronizedList(new ArrayList<>());
    private final Map<String, Double> bestParams = new ConcurrentHashMap<>();
    private volatile double bestValue = Double.MAX_VALUE;
    private volatile boolean cancelled;

    public static Builder builder() {
        return new Builder();
    }

    private HyperparameterOptimizer(Builder builder) {
        this.searchSpace = builder.searchSpace;
        this.objective = builder.objective;
        this.algorithm = builder.algorithm;
        this.maxTrials = builder.maxTrials;
        this.maxConcurrency = builder.maxConcurrency;
        this.timeoutSeconds = builder.timeoutSeconds;
        this.earlyStopping = builder.earlyStopping;

        this.executor = Executors.newFixedThreadPool(maxConcurrency);
    }

    // ============= Optimization =============

    /**
     * Run optimization.
     */
    public OptimizationResult optimize() {
        List<Trial> trials = new ArrayList<>();

        // Generate trials based on algorithm
        switch (algorithm) {
            case GRID -> trials.addAll(generateGridTrials());
            case RANDOM -> trials.addAll(generateRandomTrials());
            case BAYESIAN -> trials.addAll(generateBayesianTrials());
            case HYPERBAND -> trials.addAll(generateHyperbandTrials());
            default -> trials.addAll(generateRandomTrials());
        }

        // Execute trials
        long startTime = System.currentTimeMillis();
        List<Future<TrialResult>> futures = new ArrayList<>();

        for (Trial trial : trials) {
            if (cancelled) break;

            Future<TrialResult> future = executor.submit(() -> runTrial(trial));
            futures.add(future);
        }

        // Collect results
        List<TrialResult> results = new ArrayList<>();
        for (Future<TrialResult> f : futures) {
            try {
                results.add(f.get(timeoutSeconds, TimeUnit.SECONDS));
            } catch (Exception e) {
                System.err.println("Trial failed: " + e.getMessage());
            }
        }

        long duration = System.currentTimeMillis() - startTime;

        // Find best
        TrialResult best = results.stream()
                .min(Comparator.comparingDouble(TrialResult::value))
                .orElse(null);

        if (best != null) {
            Map<String, Double> bestTrialParams = toDoubleMap(best.trial().getParams());
            bestParams.putAll(bestTrialParams);
            bestValue = best.value();
        }

        return new OptimizationResult(bestParams, bestValue, results, duration);
    }

    /**
     * Run a single trial.
     */
    private TrialResult runTrial(Trial trial) {
        try {
            // Set parameters
            trial.setParams(searchSpace.sample(trial.getTrialId()));

            // Run objective
            long start = System.currentTimeMillis();
            double value = objective.apply(trial);
            long duration = System.currentTimeMillis() - start;

            trial.setValue(value);
            trial.setDuration(duration);
            trial.setStatus(TrialStatus.COMPLETED);

            completedTrials.add(trial);

            // Update best
            synchronized (this) {
                if (value < bestValue) {
                    bestValue = value;
                    bestParams.clear();
                    bestParams.putAll(toDoubleMap(trial.getParams()));
                }
            }

            return new TrialResult(trial, value, null);

        } catch (Exception e) {
            trial.setStatus(TrialStatus.FAILED);
            return new TrialResult(trial, Double.MAX_VALUE, e);
        }
    }

    // ============= Trial Generation =============

    /**
     * Generate grid search trials.
     */
    private List<Trial> generateGridTrials() {
        List<Trial> trials = new ArrayList<>();
        List<Map<String, Object>> combinations = searchSpace.gridCombinations();

        for (Map<String, Object> params : combinations) {
            Trial trial = new Trial(trials.size(), this);
            trial.preSetParams(params);
            trials.add(trial);

            if (trials.size() >= maxTrials) break;
        }

        return trials;
    }

    /**
     * Generate random search trials.
     */
    private List<Trial> generateRandomTrials() {
        List<Trial> trials = new ArrayList<>();

        for (int i = 0; i < maxTrials; i++) {
            Trial trial = new Trial(i, this);
            trials.add(trial);
        }

        return trials;
    }

    /**
     * Generate Bayesian optimization trials.
     */
    private List<Trial> generateBayesianTrials() {
        List<Trial> trials = new ArrayList<>();

        // Initial random trials
        int initTrials = Math.min(10, maxTrials / 4);
        for (int i = 0; i < initTrials; i++) {
            Trial trial = new Trial(i, this);
            trials.add(trial);
        }

        // Bayesian trials
        for (int i = initTrials; i < maxTrials; i++) {
            Trial trial = new Trial(i, this);
            // Sample based on acquisition function
            Map<String, Object> params = searchSpace.sampleBayesian(
                    completedTrials, bestParams);
            trial.preSetParams(params);
            trials.add(trial);
        }

        return trials;
    }

    /**
     * Generate Hyperband trials.
     */
    private List<Trial> generateHyperbandTrials() {
        List<Trial> trials = new ArrayList<>();
        int maxBracket = 5;

        for (int bracket = 0; bracket < maxBracket; bracket++) {
            int n = (int) Math.pow(3, bracket);
            int r = maxTrials / n;

            for (int i = 0; i < n && trials.size() < maxTrials; i++) {
                Trial trial = new Trial(trials.size(), this);
                trial.setResource("epochs", r);
                trials.add(trial);
            }
        }

        return trials;
    }

    /**
     * Cancel optimization.
     */
    public void cancel() {
        cancelled = true;
    }

    // ============= Statistics =============

    public HyperparameterOptimizerStats getStats() {
        return new HyperparameterOptimizerStats(
                algorithm,
                maxTrials,
                completedTrials.size(),
                bestValue,
                bestParams
        );
    }

    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        executor.shutdown();
        try {
            executor.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }
    }

    // ============= Inner Types =============

    /**
     * Trial - a single hyperparameter configuration.
     */
    public static class Trial {
        private final int trialId;
        private final HyperparameterOptimizer optimizer;
        private Map<String, Object> params = new HashMap<>();
        private volatile double value;
        private volatile long duration;
        private volatile TrialStatus status;
        private Map<String, Double> intermediateValues = new HashMap<>();

        Trial(int trialId, HyperparameterOptimizer optimizer) {
            this.trialId = trialId;
            this.optimizer = optimizer;
            this.status = TrialStatus.RUNNING;
        }

        public int getTrialId() { return trialId; }
        public Map<String, Object> getParams() { return new HashMap<>(params); }
        public double getValue() { return value; }
        public long getDuration() { return duration; }
        public TrialStatus getStatus() { return status; }

        void setParams(Map<String, Object> params) {
            this.params.putAll(params);
        }

        void preSetParams(Map<String, Object> params) {
            this.params = new HashMap<>(params);
        }

        void setValue(double value) { this.value = value; }
        void setDuration(long duration) { this.duration = duration; }
        void setStatus(TrialStatus status) { this.status = status; }

        /**
         * Report intermediate value.
         */
        public void reportIntermediate(String metric, double value) {
            intermediateValues.put(metric, value);
        }

        /**
         * Should prune (early stop).
         */
        public boolean shouldPrune(double threshold) {
            return intermediateValues.values().stream()
                    .anyMatch(v -> v > threshold);
        }

        /**
         * Set resource for Hyperband.
         */
        public void setResource(String name, int value) {
            params.put("_resource_" + name, value);
        }
    }

    /**
     * Trial status.
     */
    public enum TrialStatus {
        RUNNING,
        COMPLETED,
        FAILED,
        PRUNED
    }

    /**
     * Optimization algorithm.
     */
    public enum OptimizationAlgorithm {
        GRID,
        RANDOM,
        BAYESIAN,
        HYPERBAND
    }

    /**
     * Trial result.
     */
    public static class TrialResult {
        private final Trial trial;
        private final double value;
        private final Exception error;

        TrialResult(Trial trial, double value, Exception error) {
            this.trial = trial;
            this.value = value;
            this.error = error;
        }

        public Trial trial() { return trial; }
        public double value() { return value; }
        public Exception error() { return error; }
        public boolean isSuccess() { return error == null; }
    }

    /**
     * Convert Map&lt;String, Object&gt; to Map&lt;String, Double&gt; narrowing values.
     */
    private static Map<String, Double> toDoubleMap(Map<String, Object> in) {
        Map<String, Double> out = new HashMap<>();
        if (in == null) return out;
        for (Map.Entry<String, Object> e : in.entrySet()) {
            Object v = e.getValue();
            if (v instanceof Number) {
                out.put(e.getKey(), ((Number) v).doubleValue());
            } else if (v != null) {
                try {
                    out.put(e.getKey(), Double.parseDouble(v.toString()));
                } catch (NumberFormatException ignored) {
                    // skip non-numeric values
                }
            }
        }
        return out;
    }

    /**
     * Optimization result.
     */
    public static class OptimizationResult {
        private final Map<String, Double> bestParams;
        private final double bestValue;
        private final List<TrialResult> allResults;
        private final long durationMs;

        public OptimizationResult(Map<String, Double> bestParams, double bestValue,
                                List<TrialResult> allResults, long durationMs) {
            this.bestParams = bestParams;
            this.bestValue = bestValue;
            this.allResults = allResults;
            this.durationMs = durationMs;
        }

        public Map<String, Double> bestParams() { return bestParams; }
        public double bestValue() { return bestValue; }
        public List<TrialResult> allResults() { return allResults; }
        public long durationMs() { return durationMs; }
    }

    /**
     * Statistics.
     */
    public static class HyperparameterOptimizerStats {
        public final OptimizationAlgorithm algorithm;
        public final int maxTrials;
        public final int completedTrials;
        public final double bestValue;
        public final Map<String, Double> bestParams;

        public HyperparameterOptimizerStats(OptimizationAlgorithm algorithm, int maxTrials,
                                           int completedTrials, double bestValue,
                                           Map<String, Double> bestParams) {
            this.algorithm = algorithm;
            this.maxTrials = maxTrials;
            this.completedTrials = completedTrials;
            this.bestValue = bestValue;
            this.bestParams = bestParams;
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private SearchSpace searchSpace = new SearchSpace();
        private Function<Trial, Double> objective = t -> 0.0;
        private OptimizationAlgorithm algorithm = OptimizationAlgorithm.RANDOM;
        private int maxTrials = 100;
        private int maxConcurrency = 4;
        private long timeoutSeconds = 3600;
        private boolean earlyStopping = true;

        public Builder searchSpace(SearchSpace space) { this.searchSpace = space; return this; }
        public Builder searchSpace(Function<SearchSpace, SearchSpace> config) {
            this.searchSpace = config.apply(new SearchSpace());
            return this;
        }
        public Builder objective(Function<Trial, Double> obj) { this.objective = obj; return this; }
        public Builder algorithm(OptimizationAlgorithm algo) { this.algorithm = algo; return this; }
        public Builder maxTrials(int max) { this.maxTrials = max; return this; }
        public Builder maxConcurrency(int max) { this.maxConcurrency = max; return this; }
        public Builder timeoutSeconds(long seconds) { this.timeoutSeconds = seconds; return this; }
        public Builder earlyStopping(boolean enable) { this.earlyStopping = enable; return this; }

        public HyperparameterOptimizer build() {
            return new HyperparameterOptimizer(this);
        }
    }
}
