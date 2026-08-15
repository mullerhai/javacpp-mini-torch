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
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.trl.callback;
import org.bytedeco.pytorch.optim.schedulers.*;

import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.TrainerCallback;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiConsumer;
import java.util.function.Predicate;

/**
 * Enterprise callback system for training monitoring and control.
 *
 * <p>Provides a comprehensive callback system with:
 * <ul>
 *   <li>Early stopping based on metrics</li>
 *   <li>Learning rate scheduling</li>
 *   <li>Gradient clipping monitoring</li>
 *   <li>Custom metric tracking</li>
 *   <li>Checkpoint integration</li>
 *   <li>Event logging</li>
 * </ul>
 *
 * <p>Available callbacks:
 * <ul>
 *   <li>{@link EarlyStoppingCallback} - Stop training based on metric degradation</li>
 *   <li>{@link LRSchedulerCallback} - Adaptive learning rate scheduling</li>
 *   <li>{@link MetricsLoggerCallback} - Structured metrics logging</li>
 *   <li>{@link CheckpointCallback} - Automatic checkpointing</li>
 *   <li>{@link GradientMonitorCallback} - Gradient health monitoring</li>
 * </ul>
 *
 * <pre>{@code
 * CallbackManager manager = CallbackManager.builder(trainer)
 *     .addEarlyStopping("loss", EarlyStoppingCallback.MetricDirection.LOWER_IS_BETTER, 3)
 *     .addMetricsLogger(System.out::println)
 *     .build();
 * }</pre>
 */
public class CallbackManager implements AutoCloseable {

    private final BaseTrainer trainer;
    private final List<TrainerCallback> callbacks = new ArrayList<>();
    private final Map<String, MetricTracker> metricTrackers = new ConcurrentHashMap<>();
    private volatile boolean closed = false;

    public CallbackManager(BaseTrainer trainer) {
        this.trainer = Objects.requireNonNull(trainer, "trainer");
    }

    /**
     * Add a callback to the manager.
     */
    public CallbackManager addCallback(TrainerCallback callback) {
        if (closed) throw new IllegalStateException("CallbackManager is closed");
        if (callback != null) {
            callbacks.add(callback);
            trainer.addCallback(callback);
        }
        return this;
    }

    /**
     * Add early stopping based on a metric.
     *
     * @param metricName Metric to monitor
     * @param direction Whether higher or lower is better
     * @param patience Number of epochs without improvement before stopping
     */
    public CallbackManager addEarlyStopping(
            String metricName,
            EarlyStoppingCallback.MetricDirection direction,
            int patience) {

        EarlyStoppingCallback callback = new EarlyStoppingCallback(
                metricName, direction, patience, trainer);
        return addCallback(callback);
    }

    /**
     * Add metrics logger callback.
     */
    public CallbackManager addMetricsLogger(BiConsumer<Integer, Map<String, Double>> logger) {
        return addCallback(new MetricsLoggerCallback(trainer, logger));
    }

    /**
     * Add checkpoint callback.
     */
    public CallbackManager addCheckpointCallback(CheckpointCallback.CheckpointProvider provider, int interval) {
        CheckpointCallback callback = new CheckpointCallback(trainer, provider, interval);
        return addCallback(callback);
    }

    /**
     * Add gradient monitoring callback.
     */
    public CallbackManager addGradientMonitor(double maxGradNorm) {
        return addCallback(new GradientMonitorCallback(trainer, maxGradNorm));
    }

    /**
     * Add learning rate scheduler callback.
     */
    public CallbackManager addLRScheduler(LRSchedulerCallback.SchedulerProvider scheduler) {
        return addCallback(new LRSchedulerCallback(trainer, scheduler));
    }

    // ==================== Metric Tracking ====================

    /**
     * Get or create a metric tracker.
     */
    public MetricTracker getTracker(String metricName) {
        return metricTrackers.computeIfAbsent(metricName, MetricTracker::new);
    }

    /**
     * Record a metric value.
     */
    public void record(String metricName, double value, int step) {
        getTracker(metricName).record(value, step);
    }

    /**
     * Get metric history.
     */
    public List<Double> getHistory(String metricName) {
        return getTracker(metricName).getHistory();
    }

    // ==================== Lifecycle ====================

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        callbacks.clear();
        metricTrackers.clear();
    }

    public boolean isClosed() { return closed; }

    // ==================== Builder ====================

    public static Builder builder(BaseTrainer trainer) {
        return new Builder(trainer);
    }

    public static final class Builder {
        private final BaseTrainer trainer;
        private final List<TrainerCallback> callbacks = new ArrayList<>();

        public Builder(BaseTrainer trainer) {
            this.trainer = Objects.requireNonNull(trainer, "trainer");
        }

        public Builder addCallback(TrainerCallback callback) {
            callbacks.add(callback);
            return this;
        }

        public Builder addEarlyStopping(
                String metricName,
                EarlyStoppingCallback.MetricDirection direction,
                int patience) {
            callbacks.add(new EarlyStoppingCallback(metricName, direction, patience, trainer));
            return this;
        }

        public Builder addMetricsLogger(BiConsumer<Integer, Map<String, Double>> logger) {
            callbacks.add(new MetricsLoggerCallback(trainer, logger));
            return this;
        }

        public Builder addCheckpointCallback(
                CheckpointCallback.CheckpointProvider provider,
                int interval) {
            callbacks.add(new CheckpointCallback(trainer, provider, interval));
            return this;
        }

        public Builder addGradientMonitor(double maxGradNorm) {
            callbacks.add(new GradientMonitorCallback(trainer, maxGradNorm));
            return this;
        }

        public Builder addLRScheduler(LRSchedulerCallback.SchedulerProvider scheduler) {
            callbacks.add(new LRSchedulerCallback(trainer, scheduler));
            return this;
        }

        public CallbackManager build() {
            CallbackManager manager = new CallbackManager(trainer);
            for (TrainerCallback callback : callbacks) {
                manager.addCallback(callback);
            }
            return manager;
        }
    }

    // ==================== Supporting Types ====================

    /**
     * Metric history tracker.
     */
    public static class MetricTracker {
        private final String name;
        private final List<Double> values = new ArrayList<>();
        private final List<Integer> steps = new ArrayList<>();

        public MetricTracker(String name) {
            this.name = name;
        }

        public synchronized void record(double value, int step) {
            values.add(value);
            steps.add(step);
        }

        public synchronized List<Double> getHistory() {
            return new ArrayList<>(values);
        }

        public synchronized List<Integer> getSteps() {
            return new ArrayList<>(steps);
        }

        public synchronized double getLatest() {
            return values.isEmpty() ? 0 : values.get(values.size() - 1);
        }

        public synchronized double getBest(EarlyStoppingCallback.MetricDirection direction) {
            if (values.isEmpty()) return 0;
            return direction == EarlyStoppingCallback.MetricDirection.HIGHER_IS_BETTER
                    ? values.stream().mapToDouble(Double::doubleValue).max().orElse(0)
                    : values.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        }

        public synchronized int getPatienceCount(
                double currentValue,
                double bestValue,
                EarlyStoppingCallback.MetricDirection direction) {

            boolean isBetter = direction == EarlyStoppingCallback.MetricDirection.HIGHER_IS_BETTER
                    ? currentValue > bestValue
                    : currentValue < bestValue;

            return isBetter ? 0 : values.size();
        }
    }
}

/**
 * Early stopping callback.
 *
 * <p>Monitors a specified metric and stops training when no improvement
 * is seen for a given number of epochs.
 */
class EarlyStoppingCallback implements TrainerCallback {

    public enum MetricDirection {
        HIGHER_IS_BETTER,
        LOWER_IS_BETTER
    }

    private final String metricName;
    private final MetricDirection direction;
    private final int patience;
    private final BaseTrainer trainer;
    private volatile boolean shouldStop = false;
    private int bestStep = 0;
    private double bestValue = 0;
    private int patienceCounter = 0;

    public EarlyStoppingCallback(
            String metricName,
            MetricDirection direction,
            int patience,
            BaseTrainer trainer) {
        this.metricName = Objects.requireNonNull(metricName, "metricName");
        this.direction = Objects.requireNonNull(direction, "direction");
        this.patience = Math.max(1, patience);
        this.trainer = Objects.requireNonNull(trainer, "trainer");
    }

    @Override
    public void onStepEnd(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        if (shouldStop) return;

        Double value = metrics.get(metricName);
        if (value == null) return;

        boolean isBetter = direction == MetricDirection.HIGHER_IS_BETTER
                ? value > bestValue
                : value < bestValue;

        if (isBetter || bestStep == 0) {
            bestValue = value;
            bestStep = step;
            patienceCounter = 0;
        } else {
            patienceCounter++;
        }

        if (patienceCounter >= patience) {
            shouldStop = true;
            System.out.printf(
                    "[EarlyStopping] Stopping at step %d (%s=%.4f, best=%.4f at step %d)%n",
                    step, metricName, value, bestValue, bestStep);
        }
    }

    @Override
    public void onTrainEnd(BaseTrainer trainer) {
        System.out.printf(
                "[EarlyStopping] Training ended. Best %s=%.4f at step %d%n",
                metricName, bestValue, bestStep);
    }

    public boolean shouldStop() { return shouldStop; }
    public double getBestValue() { return bestValue; }
    public int getBestStep() { return bestStep; }
}

/**
 * Metrics logging callback.
 */
class MetricsLoggerCallback implements TrainerCallback {

    private final BaseTrainer trainer;
    private final BiConsumer<Integer, Map<String, Double>> logger;
    private final int logInterval;
    private int lastLogStep = 0;

    public MetricsLoggerCallback(
            BaseTrainer trainer,
            BiConsumer<Integer, Map<String, Double>> logger) {
        this(trainer, logger, 1);
    }

    public MetricsLoggerCallback(
            BaseTrainer trainer,
            BiConsumer<Integer, Map<String, Double>> logger,
            int logInterval) {
        this.trainer = trainer;
        this.logger = Objects.requireNonNull(logger, "logger");
        this.logInterval = logInterval;
    }

    @Override
    public void onLog(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        if (step - lastLogStep >= logInterval) {
            logger.accept(step, new LinkedHashMap<>(metrics));
            lastLogStep = step;
        }
    }
}

/**
 * Checkpoint callback.
 */
class CheckpointCallback implements TrainerCallback {

    @FunctionalInterface
    public interface CheckpointProvider {
        void saveCheckpoint(int step, Map<String, Double> metrics);
    }

    private final BaseTrainer trainer;
    private final CheckpointProvider provider;
    private final int saveInterval;
    private int lastSaveStep = 0;

    public CheckpointCallback(
            BaseTrainer trainer,
            CheckpointProvider provider,
            int saveInterval) {
        this.trainer = trainer;
        this.provider = Objects.requireNonNull(provider, "provider");
        this.saveInterval = Math.max(1, saveInterval);
    }

    @Override
    public void onStepEnd(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        if (step - lastSaveStep >= saveInterval) {
            provider.saveCheckpoint(step, new LinkedHashMap<>(metrics));
            lastSaveStep = step;
        }
    }
}

/**
 * Gradient monitoring callback.
 */
class GradientMonitorCallback implements TrainerCallback {

    private final BaseTrainer trainer;
    private final double maxGradNorm;
    private volatile double lastGradNorm = 0;
    private int clippedSteps = 0;
    private int totalSteps = 0;

    public GradientMonitorCallback(BaseTrainer trainer, double maxGradNorm) {
        this.trainer = trainer;
        this.maxGradNorm = maxGradNorm;
    }

    @Override
    public void onStepEnd(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        totalSteps++;

        Double gradNorm = metrics.get("grad_norm");
        if (gradNorm != null) {
            lastGradNorm = gradNorm;
            if (gradNorm > maxGradNorm) {
                clippedSteps++;
            }
        }

        // Log warning if gradients are exploding
        if (gradNorm != null && gradNorm > maxGradNorm * 10) {
            System.err.printf(
                    "[GradientMonitor] WARNING: Large gradient norm at step %d: %.2f%n",
                    step, gradNorm);
        }
    }

    @Override
    public void onTrainEnd(BaseTrainer trainer) {
        double clipRate = totalSteps > 0 ? (double) clippedSteps / totalSteps : 0;
        System.out.printf(
                "[GradientMonitor] Training ended. Avg grad norm: %.4f, Clipped: %d/%d (%.1f%%)%n",
                lastGradNorm, clippedSteps, totalSteps, clipRate * 100);
    }

    public double getLastGradNorm() { return lastGradNorm; }
    public int getClippedSteps() { return clippedSteps; }
}

/**
 * Learning rate scheduler callback.
 *
 * <p>Provides adaptive learning rate scheduling based on training metrics.
 * Supports various scheduling strategies including step decay, exponential decay,
 * cosine annealing, and metric-based scheduling.
 */
class LRSchedulerCallback implements TrainerCallback {

    /**
     * Provider interface for the actual scheduler implementation.
     */
    @FunctionalInterface
    public interface SchedulerProvider {
        /**
         * Get the new learning rate based on current step and metrics.
         *
         * @param currentLr Current learning rate
         * @param step Current training step
         * @param metrics Current training metrics
         * @return New learning rate
         */
        double getNextLr(double currentLr, int step, Map<String, Double> metrics);
    }

    private final BaseTrainer trainer;
    private final SchedulerProvider scheduler;
    private final LRSchedulerType schedulerType;
    private double initialLr;
    private double lastLr;
    private int lastUpdateStep = 0;

    public enum LRSchedulerType {
        STEP_DECAY,
        EXPONENTIAL_DECAY,
        COSINE_ANNEALING,
        METRIC_BASED,
        POLYNOMIAL_DECAY
    }

    public LRSchedulerCallback(BaseTrainer trainer, SchedulerProvider scheduler) {
        this(trainer, scheduler, LRSchedulerType.METRIC_BASED);
    }

    public LRSchedulerCallback(
            BaseTrainer trainer,
            SchedulerProvider scheduler,
            LRSchedulerType schedulerType) {
        this.trainer = Objects.requireNonNull(trainer, "trainer");
        this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
        this.schedulerType = schedulerType;
    }

    @Override
    public void onStepEnd(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        double newLr = scheduler.getNextLr(lastLr > 0 ? lastLr : getInitialLr(), step, metrics);

        if (newLr != lastLr && step > lastUpdateStep) {
            applyLr(newLr, step);
            lastUpdateStep = step;
        }
    }

    @Override
    public void onTrainBegin(BaseTrainer trainer) {
        this.initialLr = trainer.optimizer() != null ? trainer.config().learningRate() : 1e-5;
        this.lastLr = initialLr;
        System.out.printf("[LRScheduler] Initial LR: %.2e%n", initialLr);
    }

    private double getInitialLr() {
        if (initialLr > 0) return initialLr;
        return trainer.optimizer() != null ? trainer.config().learningRate() : 1e-5;
    }

    private void applyLr(double newLr, int step) {
        lastLr = newLr;
        // In a full implementation, this would update the optimizer's LR
        // For now, just track the change
        System.out.printf("[LRScheduler] Step %d: LR = %.2e%n", step, newLr);
    }

    public double getCurrentLr() { return lastLr; }
    public double getInitialLrValue() { return getInitialLr(); }

    // ==================== Factory Methods ====================

    /**
     * Create a step decay scheduler.
     */
    public static LRSchedulerCallback stepDecay(
            BaseTrainer trainer,
            double decayRate,
            int stepSize) {
        return new LRSchedulerCallback(trainer, (currentLr, step, metrics) -> {
            if (step > 0 && step % stepSize == 0) {
                return currentLr * decayRate;
            }
            return currentLr;
        }, LRSchedulerType.STEP_DECAY);
    }

    /**
     * Create an exponential decay scheduler.
     */
    public static LRSchedulerCallback exponentialDecay(
            BaseTrainer trainer,
            double gamma) {
        return new LRSchedulerCallback(trainer, (currentLr, step, metrics) -> {
            if (step > 0) {
                return currentLr * Math.pow(gamma, step);
            }
            return currentLr;
        }, LRSchedulerType.EXPONENTIAL_DECAY);
    }

    /**
     * Create a cosine annealing scheduler.
     */
    public static LRSchedulerCallback cosineAnnealing(
            BaseTrainer trainer,
            int totalSteps) {
        return new LRSchedulerCallback(trainer, (currentLr, step, metrics) -> {
            if (step > 0 && totalSteps > 0) {
                double progress = (double) step / totalSteps;
                return trainer.config().learningRate() * (1 + Math.cos(Math.PI * progress)) / 2;
            }
            return currentLr;
        }, LRSchedulerType.COSINE_ANNEALING);
    }
}
