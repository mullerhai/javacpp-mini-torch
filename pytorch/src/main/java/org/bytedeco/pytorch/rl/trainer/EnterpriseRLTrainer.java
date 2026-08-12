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
package org.bytedeco.pytorch.rl.trainer;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer.TrainingCallback;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer.TrainingMetrics;
import org.bytedeco.pytorch.llm.trainer.ORPOTrainer;
import org.bytedeco.pytorch.llm.trainer.SimPOTrainer;
import org.bytedeco.pytorch.llm.trainer.RLOOTrainer;
import org.bytedeco.pytorch.llm.trainer.MultiModalTrainer;
import org.bytedeco.pytorch.llm.monitoring.MetricsCollector;

import java.util.concurrent.atomic.AtomicReference;

/**
 * Enterprise RL Trainer integrating ORPO, SimPO, RLOO with existing RL modules.
 *
 * <p>Reference: TRL (Transformers Reinforcement Learning)
 *
 * <pre>{@code
 * EnterpriseRLTrainer trainer = EnterpriseRLTrainer.builder()
 *     .model(model)
 *     .algorithm(EnterpriseRLTrainer.Algorithm.ORPO)
 *     .beta(0.1)
 *     .metricsCollector(collector)
 *     .build();
 *
 * trainer.train();
 * }</pre>
 */
public class EnterpriseRLTrainer implements EnterpriseTrainer {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Algorithm type
    private final Algorithm algorithm;

    // Trainers
    private ORPOTrainer orpoTrainer;
    private SimPOTrainer simpoTrainer;
    private RLOOTrainer rlooTrainer;

    // Delegate
    private final EnterpriseTrainer delegate;

    // Metrics
    private final MetricsCollector metrics;

    // State
    private final AtomicReference<TrainingState> state = new AtomicReference<>(TrainingState.IDLE);

    /**
     * Supported RL algorithms.
     */
    public enum Algorithm {
        ORPO,
        SIMPO,
        RLOO,
        DPO,       // Legacy
        GRPO,      // Legacy
        PPO        // Legacy
    }

    public static Builder builder() {
        return new Builder();
    }

    private EnterpriseRLTrainer(Builder builder) {
        this.algorithm = builder.algorithm;
        this.metrics = builder.metrics;
        this.orpoTrainer = builder.orpoTrainer;
        this.simpoTrainer = builder.simpoTrainer;
        this.rlooTrainer = builder.rlooTrainer;
        this.delegate = EnterpriseTrainer.builder()
                .model(builder.model)
                .config(builder.config)
                .build();

        // Initialize trainers based on algorithm
        initializeTrainer(builder);

        // Add metrics callback
        if (metrics != null) {
            delegate.addCallback(new MetricsCallback());
        }
    }

    private void initializeTrainer(Builder builder) {
        switch (algorithm) {
            case ORPO:
                if (orpoTrainer == null) {
                    orpoTrainer = ORPOTrainer.builder()
                            .model(builder.model)
                            .beta(builder.beta)
                            .lambda(builder.lambda)
                            .build();
                }
                break;
            case SIMPO:
                if (simpoTrainer == null) {
                    simpoTrainer = SimPOTrainer.builder()
                            .model(builder.model)
                            .beta(builder.beta)
                            .gamma(builder.gamma)
                            .build();
                }
                break;
            case RLOO:
                if (rlooTrainer == null) {
                    rlooTrainer = RLOOTrainer.builder()
                            .model(builder.model)
                            .baselineCoeff(builder.baselineCoeff)
                            .learningRate(builder.learningRate)
                            .build();
                }
                break;
        }
    }

    @Override
    public void initialize() {
        delegate.initialize();
        state.set(TrainingState.IDLE);
    }

    @Override
    public void train() {
        state.set(TrainingState.TRAINING);
        try {
            delegate.train();
            state.set(TrainingState.COMPLETED);
        } catch (Exception e) {
            state.set(TrainingState.FAILED);
            throw e;
        }
    }

    @Override
    public void eval(java.util.List<String> evalMetrics) {
        delegate.eval(evalMetrics);
    }

    @Override
    public org.bytedeco.pytorch.Tensor predict(org.bytedeco.pytorch.Tensor input) {
        return delegate.predict(input);
    }

    @Override
    public java.util.List<org.bytedeco.pytorch.Tensor> predictBatch(java.util.List<org.bytedeco.pytorch.Tensor> inputs) {
        return delegate.predictBatch(inputs);
    }

    @Override
    public void pause() {
        delegate.pause();
        state.set(TrainingState.PAUSED);
    }

    @Override
    public void resume() {
        delegate.resume();
        state.set(TrainingState.TRAINING);
    }

    @Override
    public void stop() {
        delegate.stop();
        state.set(TrainingState.STOPPING);
    }

    @Override
    public void kill() {
        delegate.kill();
        state.set(TrainingState.KILLED);
    }

    @Override
    public boolean isStopped() { return delegate.isStopped(); }

    @Override
    public boolean isPaused() { return delegate.isPaused(); }

    @Override
    public TrainingState getState() { return state.get(); }

    @Override
    public int getGlobalStep() { return delegate.getGlobalStep(); }

    @Override
    public int getCurrentEpoch() { return delegate.getCurrentEpoch(); }

    @Override
    public int getTotalSteps() { return delegate.getTotalSteps(); }

    @Override
    public double getProgress() { return delegate.getProgress(); }

    @Override
    public Module getModel() { return delegate.getModel(); }

    @Override
    public org.bytedeco.pytorch.optim.Optimizer getOptimizer() { return delegate.getOptimizer(); }

    @Override
    public void saveCheckpoint(String path) { delegate.saveCheckpoint(path); }

    @Override
    public void loadCheckpoint(String path) { delegate.loadCheckpoint(path); }

    @Override
    public TrainingMetrics getMetrics() { return delegate.getMetrics(); }

    @Override
    public java.util.List<TrainingMetrics> getHistory() { return delegate.getHistory(); }

    @Override
    public TrainingMetrics getMetricsAtStep(int step) { return delegate.getMetricsAtStep(step); }

    @Override
    public void addCallback(TrainingCallback callback) { delegate.addCallback(callback); }

    @Override
    public void removeCallback(TrainingCallback callback) { delegate.removeCallback(callback); }

    @Override
    public void close() throws Exception {
        if (closed) return;
        closed = true;

        if (orpoTrainer != null) orpoTrainer.close();
        if (simpoTrainer != null) simpoTrainer.close();
        if (rlooTrainer != null) rlooTrainer.close();
        delegate.close();

        System.out.println("[EnterpriseRLTrainer] Closed: algorithm=" + algorithm);
    }

    public boolean isClosed() { return closed; }

    /**
     * Get current algorithm.
     */
    public Algorithm getAlgorithm() { return algorithm; }

    /**
     * Get ORPO trainer.
     */
    public ORPOTrainer getOrpoTrainer() { return orpoTrainer; }

    /**
     * Get SimPO trainer.
     */
    public SimPOTrainer getSimpoTrainer() { return simpoTrainer; }

    /**
     * Get RLOO trainer.
     */
    public RLOOTrainer getRlooTrainer() { return rlooTrainer; }

    /**
     * Metrics callback.
     */
    private class MetricsCallback implements TrainingCallback {
        @Override
        public void onStepEnd(EnterpriseTrainer trainer, int step, TrainingMetrics metrics) {
            if (EnterpriseRLTrainer.this.metrics != null) {
                MetricsCollector mc = EnterpriseRLTrainer.this.metrics;
                mc.incrementCounter("rl.steps");
                mc.gauge("rl.loss").set(metrics.loss());
                mc.gauge("rl.algorithm").set(algorithm.ordinal());
            }
        }

        @Override
        public void onEpochEnd(EnterpriseTrainer trainer, int epoch, TrainingMetrics metrics) {
            System.out.printf("[EnterpriseRLTrainer] Epoch %d: algorithm=%s, loss=%.4f%n",
                    epoch, algorithm, metrics.loss());
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private Module model;
        private Algorithm algorithm = Algorithm.ORPO;
        private org.bytedeco.pytorch.llm.trainer.TrainerConfig config;
        private MetricsCollector metrics;

        // ORPO params
        private float beta = 0.1f;
        private float lambda = 1.0f;
        private ORPOTrainer orpoTrainer;

        // SimPO params
        private float gamma = 0.5f;
        private SimPOTrainer simpoTrainer;

        // RLOO params
        private float baselineCoeff = 0.99f;
        private float learningRate = 1e-5f;
        private RLOOTrainer rlooTrainer;

        public Builder model(Module model) { this.model = model; return this; }
        public Builder algorithm(Algorithm algorithm) { this.algorithm = algorithm; return this; }
        public Builder config(org.bytedeco.pytorch.llm.trainer.TrainerConfig config) { this.config = config; return this; }
        public Builder metrics(MetricsCollector metrics) { this.metrics = metrics; return this; }

        // ORPO
        public Builder beta(float beta) { this.beta = beta; return this; }
        public Builder lambda(float lambda) { this.lambda = lambda; return this; }
        public Builder orpoTrainer(ORPOTrainer trainer) { this.orpoTrainer = trainer; return this; }

        // SimPO
        public Builder gamma(float gamma) { this.gamma = gamma; return this; }
        public Builder simpoTrainer(SimPOTrainer trainer) { this.simpoTrainer = trainer; return this; }

        // RLOO
        public Builder baselineCoeff(float coeff) { this.baselineCoeff = coeff; return this; }
        public Builder learningRate(float lr) { this.learningRate = lr; return this; }
        public Builder rlooTrainer(RLOOTrainer trainer) { this.rlooTrainer = trainer; return this; }

        public EnterpriseRLTrainer build() {
            return new EnterpriseRLTrainer(this);
        }
    }
}
