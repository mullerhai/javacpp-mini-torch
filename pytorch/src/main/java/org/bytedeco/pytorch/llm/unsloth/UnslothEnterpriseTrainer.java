/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, any later version (collectively, the "License");
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
package org.bytedeco.pytorch.llm.unsloth;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer.TrainingCallback;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer.TrainingMetrics;
import org.bytedeco.pytorch.llm.trainer.TrainerConfig;
import org.bytedeco.pytorch.llm.monitoring.MetricsCollector;

import java.util.concurrent.atomic.AtomicReference;

/**
 * Unsloth integration with EnterpriseTrainer.
 *
 * <p>Bridges Unsloth's fast model optimization with enterprise training features.
 *
 * <pre>{@code
 * UnslothEnterpriseTrainer trainer = UnslothEnterpriseTrainer.builder()
 *     .model(fastModel)
 *     .optimizer(optimizer)
 *     .trainerConfig(config)
 *     .metricsCollector(collector)
 *     .build();
 *
 * trainer.train();
 * }</pre>
 */
public class UnslothEnterpriseTrainer implements EnterpriseTrainer {

    private volatile boolean closed;

    // Components
    private final FastLanguageModel fastModel;
    private final EnterpriseTrainer delegate;
    private final MetricsCollector metrics;

    // State
    private final AtomicReference<TrainingState> state = new AtomicReference<>(TrainingState.IDLE);

    public static Builder builder() {
        return new Builder();
    }

    private UnslothEnterpriseTrainer(Builder builder) {
        this.fastModel = builder.fastModel;

        // Create delegate with config
        TrainerConfig config = builder.config != null ? builder.config : TrainerConfig.defaults();
        this.delegate = EnterpriseTrainer.builder()
                .model(fastModel.model())
                .optimizer(builder.optimizer)
                .config(config)
                .build();

        // Initialize metrics if provided
        this.metrics = builder.metrics;
        if (metrics != null) {
            this.delegate.addCallback(new MetricsCallback());
        }
    }

    @Override
    public void initialize() {
        fastModel.forTraining();
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
    public void eval(java.util.List<String> metrics) {
        fastModel.model().eval();
        delegate.eval(metrics);
        fastModel.forTraining();
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
    public Optimizer getOptimizer() { return delegate.getOptimizer(); }

    @Override
    public void saveCheckpoint(String path) {
        delegate.saveCheckpoint(path);
    }

    @Override
    public void loadCheckpoint(String path) {
        delegate.loadCheckpoint(path);
    }

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
    public void close() {
        if (closed) return;
        closed = true;
        delegate.close();
        System.out.println("[UnslothEnterpriseTrainer] Closed");
    }

    public boolean isClosed() { return closed; }

    /**
     * Get FastLanguageModel.
     */
    public FastLanguageModel getFastModel() {
        return fastModel;
    }

    /**
     * Metrics callback for Unsloth.
     */
    private class MetricsCallback implements TrainingCallback {
        @Override
        public void onStepEnd(EnterpriseTrainer trainer, int step, TrainingMetrics metrics) {
            if (MetricsCollector.this.metrics != null) {
                MetricsCollector.this.metrics.incrementCounter("training.steps");
                MetricsCollector.this.metrics.gauge("training.loss").set(metrics.loss());
                MetricsCollector.this.metrics.gauge("training.learning_rate").set(metrics.learningRate());
                MetricsCollector.this.metrics.gauge("training.grad_norm").set(metrics.gradNorm());
            }
        }

        @Override
        public void onEpochEnd(EnterpriseTrainer trainer, int epoch, TrainingMetrics metrics) {
            System.out.printf("[Unsloth] Epoch %d complete: loss=%.4f%n", epoch, metrics.loss());
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private FastLanguageModel fastModel;
        private Optimizer optimizer;
        private TrainerConfig config;
        private MetricsCollector metrics;

        public Builder fastModel(FastLanguageModel model) { this.fastModel = model; return this; }
        public Builder optimizer(Optimizer optimizer) { this.optimizer = optimizer; return this; }
        public Builder trainerConfig(TrainerConfig config) { this.config = config; return this; }
        public Builder config(TrainerConfig config) { this.config = config; return this; }
        public Builder metrics(MetricsCollector metrics) { this.metrics = metrics; return this; }

        public UnslothEnterpriseTrainer build() {
            return new UnslothEnterpriseTrainer(this);
        }
    }
}
