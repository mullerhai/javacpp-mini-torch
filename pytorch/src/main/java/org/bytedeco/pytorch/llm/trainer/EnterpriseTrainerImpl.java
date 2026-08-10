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
package org.bytedeco.pytorch.llm.trainer;
import org.bytedeco.pytorch.optim.schedulers.*;

import org.bytedeco.pytorch.amp.GradScaler;
import org.bytedeco.pytorch.amp.config.AmpConfig;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.optim.lr_scheduler.LRScheduler;
import org.bytedeco.pytorch.schedulers.WarmupScheduler;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Enterprise trainer implementation.
 */
public class EnterpriseTrainerImpl implements EnterpriseTrainer {

    private volatile boolean closed;
    private final ReentrantLock stateLock = new ReentrantLock();

    // Components
    private final Module model;
    private final Optimizer optimizer;
    private final TrainerConfig config;
    private final List<TrainingCallback> callbacks;
    private final GradScaler gradScaler;

    // State
    private final AtomicReference<TrainingState> state = new AtomicReference<>(TrainingState.IDLE);
    private final AtomicInteger globalStep = new AtomicInteger(0);
    private final AtomicInteger currentEpoch = new AtomicInteger(0);
    private final AtomicBoolean paused = new AtomicBoolean(false);
    private final AtomicBoolean stopped = new AtomicBoolean(false);
    private final AtomicReference<Throwable> lastError = new AtomicReference<>(null);

    // Metrics
    private final ConcurrentLinkedQueue<TrainingMetrics> metricsHistory = new ConcurrentLinkedQueue<>();
    private final AtomicLong totalTokensProcessed = new AtomicLong(0);
    private final AtomicLong totalSamplesProcessed = new AtomicLong(0);
    private volatile TrainingMetrics lastMetrics;

    // Thread pool for async operations
    private final ExecutorService executor = Executors.newCachedThreadPool();

    public EnterpriseTrainerImpl(Module model, Optimizer optimizer, TrainerConfig config,
                               List<TrainingCallback> callbacks) {
        this.model = Objects.requireNonNull(model, "model");
        this.optimizer = optimizer;
        this.config = config != null ? config : TrainerConfig.defaults();
        this.callbacks = callbacks != null ? new CopyOnWriteArrayList<>(callbacks) : new CopyOnWriteArrayList<>();

        // Initialize AMP if enabled
        if (this.config.useAmp()) {
            AmpConfig ampConfig = AmpConfig.builder()
                    .enabled(true)
                    .initScale(this.config.ampInitScale())
                    .growthFactor(this.config.ampGrowthFactor())
                    .backoffFactor(this.config.ampBackoffFactor())
                    .build();
            this.gradScaler = GradScaler.builder()
                    .initScale(ampConfig.initScale())
                    .growthFactor(ampConfig.growthFactor())
                    .backoffFactor(ampConfig.backoffFactor())
                    .build();
        } else {
            this.gradScaler = null;
        }
    }

    @Override
    public void initialize() {
        stateLock.lock();
        try {
            state.set(TrainingState.IDLE);
            globalStep.set(0);
            currentEpoch.set(0);
            stopped.set(false);
            paused.set(false);
            lastError.set(null);

            // Fire initialize callbacks
            for (TrainingCallback cb : callbacks) {
                try { cb.onTrainBegin(this); } catch (Exception e) {
                    System.err.println("Callback error: " + e.getMessage());
                }
            }
        } finally {
            stateLock.unlock();
        }
    }

    @Override
    public void train() {
        stateLock.lock();
        try {
            if (state.get() == TrainingState.TRAINING) {
                return;  // Already training
            }
            state.set(TrainingState.TRAINING);
            stopped.set(false);
            paused.set(false);
        } finally {
            stateLock.unlock();
        }

        try {
            int epochs = config.numTrain_epochs();
            int stepsPerEpoch = config.maxSteps() / epochs;

            for (int epoch = currentEpoch.get(); epoch < epochs && !stopped.get(); epoch++) {
                currentEpoch.set(epoch);

                // Fire epoch callbacks
                for (TrainingCallback cb : callbacks) {
                    try { cb.onEpochBegin(this, epoch); } catch (Exception e) {
                        System.err.println("Callback error: " + e.getMessage());
                    }
                }

                for (int step = 0; step < stepsPerEpoch && !stopped.get(); step++) {
                    // Handle pause
                    while (paused.get() && !stopped.get()) {
                        Thread.sleep(100);
                    }

                    if (stopped.get()) break;

                    int currentStep = globalStep.get();
                    long stepStart = System.currentTimeMillis();

                    // Fire step begin callbacks
                    for (TrainingCallback cb : callbacks) {
                        try { cb.onStepBegin(this, currentStep); } catch (Exception e) {
                            System.err.println("Callback error: " + e.getMessage());
                        }
                    }

                    // Execute training step
                    TrainingMetrics metrics = executeStep(currentStep, stepStart);

                    // Store metrics
                    metricsHistory.add(metrics);
                    lastMetrics = metrics;
                    globalStep.incrementAndGet();

                    // Fire step end callbacks
                    for (TrainingCallback cb : callbacks) {
                        try { cb.onStepEnd(this, currentStep, metrics); } catch (Exception e) {
                            System.err.println("Callback error: " + e.getMessage());
                        }
                    }
                }

                // Fire epoch end callbacks
                for (TrainingCallback cb : callbacks) {
                    try { cb.onEpochEnd(this, epoch, lastMetrics); } catch (Exception e) {
                        System.err.println("Callback error: " + e.getMessage());
                    }
                }
            }

            state.set(TrainingState.COMPLETED);

        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            state.set(TrainingState.KILLED);
        } catch (Throwable e) {
            lastError.set(e);
            state.set(TrainingState.FAILED);
            for (TrainingCallback cb : callbacks) {
                try { cb.onError(this, e); } catch (Exception ignored) {}
            }
        } finally {
            for (TrainingCallback cb : callbacks) {
                try { cb.onTrainEnd(this); } catch (Exception ignored) {}
            }
        }
    }

    private TrainingMetrics executeStep(int step, long stepStart) {
        try {
            // Forward pass
            Tensor input = createDummyInput();
            Tensor loss = modelForward(input);

            // Scale loss for AMP
            if (gradScaler != null) {
                loss = gradScaler.scale(loss);
            }

            // Backward pass
            loss.backward();

            // Unscale gradients
            if (gradScaler != null) {
                // Create dummy params for unscaling
                gradScaler.update();
            }

            // Optimizer step
            if (optimizer != null) {
                optimizer.step();
                optimizer.zero_grad();
            }

            // Calculate metrics
            long stepTime = System.currentTimeMillis() - stepStart;
            double currentLoss = loss.item_float();

            return new TrainingMetrics(
                    step,
                    currentEpoch.get(),
                    (double) step / config.maxSteps(),
                    currentLoss,
                    getLearningRate(),
                    getGradNorm(),
                    getGpuMemoryMB(),
                    calculateThroughput(stepTime),
                    stepTime,
                    System.currentTimeMillis()
            );

        } catch (Exception e) {
            System.err.println("Step error: " + e.getMessage());
            return new TrainingMetrics(
                    step, currentEpoch.get(), 0, Double.NaN,
                    getLearningRate(), 0, getGpuMemoryMB(), 0, 0, System.currentTimeMillis()
            );
        }
    }

    private Tensor modelForward(Tensor input) {
        // Simplified - actual implementation would use real data
        return torch.randn(new long[]{1, 10});
    }

    private Tensor createDummyInput() {
        return torch.randn(new long[]{config.trainBatchSize(), config.maxSeqLength()});
    }

    private double getLearningRate() {
        if (optimizer == null) return 0;
        return 0.001;  // Placeholder
    }

    private double getGradNorm() {
        return 0.5;  // Placeholder
    }

    private double getGpuMemoryMB() {
        try {
            if (torch.cuda_is_available()) {
                // Get CUDA memory
                return 0;  // Placeholder
            }
        } catch (Exception ignored) {}
        return 0;
    }

    private double calculateThroughput(long stepTimeMs) {
        if (stepTimeMs <= 0) return 0;
        int batchTokens = config.trainBatchSize() * config.maxSeqLength();
        return batchTokens / (stepTimeMs / 1000.0);
    }

    @Override
    public void eval(List<String> metrics) {
        if (state.get() == TrainingState.TRAINING) {
            // Run eval in separate thread
            executor.submit(() -> runEval(metrics));
        } else {
            runEval(metrics);
        }
    }

    private void runEval(List<String> metrics) {
        // Simplified eval implementation
        System.out.println("Running evaluation...");
    }

    @Override
    public org.bytedeco.pytorch.Tensor predict(org.bytedeco.pytorch.Tensor input) {
        model.eval();
        try {
            return model.forward(input);
        } finally {
            if (state.get() == TrainingState.TRAINING) {
                model.train();
            }
        }
    }

    @Override
    public List<org.bytedeco.pytorch.Tensor> predictBatch(List<org.bytedeco.pytorch.Tensor> inputs) {
        List<org.bytedeco.pytorch.Tensor> results = new ArrayList<>();
        for (org.bytedeco.pytorch.Tensor input : inputs) {
            results.add(predict(input));
        }
        return results;
    }

    @Override
    public void pause() {
        paused.set(true);
        state.set(TrainingState.PAUSED);
        for (TrainingCallback cb : callbacks) {
            try { cb.onPause(this); } catch (Exception ignored) {}
        }
    }

    @Override
    public void resume() {
        paused.set(false);
        state.set(TrainingState.TRAINING);
        for (TrainingCallback cb : callbacks) {
            try { cb.onResume(this); } catch (Exception ignored) {}
        }
    }

    @Override
    public void stop() {
        stopped.set(true);
        state.set(TrainingState.STOPPING);
        for (TrainingCallback cb : callbacks) {
            try { cb.onStop(this); } catch (Exception ignored) {}
        }
    }

    @Override
    public void kill() {
        stopped.set(true);
        state.set(TrainingState.KILLED);
        executor.shutdownNow();
    }

    @Override
    public boolean isStopped() { return stopped.get(); }
    @Override
    public boolean isPaused() { return paused.get(); }

    @Override
    public TrainingState getState() { return state.get(); }

    @Override
    public int getGlobalStep() { return globalStep.get(); }

    @Override
    public int getCurrentEpoch() { return currentEpoch.get(); }

    @Override
    public int getTotalSteps() { return config.maxSteps(); }

    @Override
    public double getProgress() {
        int total = config.maxSteps();
        return total > 0 ? (double) globalStep.get() / total : 0;
    }

    @Override
    public Module getModel() { return model; }

    @Override
    public Optimizer getOptimizer() { return optimizer; }

    @Override
    public void saveCheckpoint(String path) {
        try {
            model.save(path);
            for (TrainingCallback cb : callbacks) {
                try { cb.onCheckpointSave(this, path); } catch (Exception ignored) {}
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to save checkpoint: " + path, e);
        }
    }

    @Override
    public void loadCheckpoint(String path) {
        try {
            model.load(path);
            for (TrainingCallback cb : callbacks) {
                try { cb.onCheckpointLoad(this, path); } catch (Exception ignored) {}
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to load checkpoint: " + path, e);
        }
    }

    @Override
    public TrainingMetrics getMetrics() {
        return lastMetrics;
    }

    @Override
    public List<TrainingMetrics> getHistory() {
        return new ArrayList<>(metricsHistory);
    }

    @Override
    public TrainingMetrics getMetricsAtStep(int step) {
        return metricsHistory.stream()
                .filter(m -> m.globalStep() == step)
                .findFirst()
                .orElse(null);
    }

    @Override
    public void addCallback(TrainingCallback callback) {
        if (callback != null) {
            callbacks.add(callback);
        }
    }

    @Override
    public void removeCallback(TrainingCallback callback) {
        callbacks.remove(callback);
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        stop();
        executor.shutdown();

        if (gradScaler != null) {
            try { gradScaler.close(); } catch (Exception ignored) {}
        }

        System.out.printf(
                "[EnterpriseTrainer] Closed: steps=%d, epochs=%d, state=%s%n",
                globalStep.get(), currentEpoch.get(), state.get());
    }

    public boolean isClosed() { return closed; }
}
