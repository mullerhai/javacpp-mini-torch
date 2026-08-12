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
package org.bytedeco.pytorch.llm.llamafactory;

import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer.TrainingCallback;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer.TrainingMetrics;
import org.bytedeco.pytorch.llm.trainer.TrainerConfig;
import org.bytedeco.pytorch.llm.trainer.ORPOTrainer;
import org.bytedeco.pytorch.llm.trainer.SimPOTrainer;
import org.bytedeco.pytorch.llm.trainer.RLOOTrainer;
import org.bytedeco.pytorch.llm.monitoring.MetricsCollector;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * LlamaFactory integration with EnterpriseTrainer.
 *
 * <p>Bridges LlamaFactory's training pipeline with enterprise features:
 * <ul>
 *   <li>Multiple training stages (SFT, RLHF, DPO, ORPO, etc.)</li>
 *   <li>PEFT methods (LoRA, QLoRA, etc.)</li>
 *   <li>Enterprise monitoring and metrics</li>
 * </ul>
 *
 * <pre>{@code
 * LlamaFactoryEnterpriseTrainer trainer = LlamaFactoryEnterpriseTrainer.builder()
 *     .factoryArgs(args)
 *     .trainerConfig(config)
 *     .metricsCollector(collector)
 *     .build();
 *
 * trainer.train();
 * }</pre>
 */
public class LlamaFactoryEnterpriseTrainer implements EnterpriseTrainer {

    private volatile boolean closed;

    // Components
    private final FactoryArgs factoryArgs;
    private final EnterpriseTrainer delegate;
    private final TrainerConfig config;
    private final MetricsCollector metrics;

    // State
    private final AtomicReference<TrainingState> state = new AtomicReference<>(TrainingState.IDLE);

    // Trainer instances for different stages
    private ORPOTrainer orpoTrainer;
    private SimPOTrainer simpoTrainer;
    private RLOOTrainer rlooTrainer;

    public static Builder builder() {
        return new Builder();
    }

    private LlamaFactoryEnterpriseTrainer(Builder builder) {
        this.factoryArgs = builder.factoryArgs;
        this.config = builder.config != null ? builder.config : TrainerConfig.defaults();
        this.metrics = builder.metrics;

        // Create delegate trainer
        this.delegate = EnterpriseTrainer.builder()
                .config(this.config)
                .build();

        // Add metrics callback if provided
        if (metrics != null) {
            this.delegate.addCallback(new FactoryMetricsCallback());
        }

        // Initialize specialized trainers based on stage
        initializeTrainers();
    }

    private void initializeTrainers() {
        String stage = factoryArgs.finetuning().stage().wireName();

        switch (stage.toLowerCase()) {
            case "orpo":
                orpoTrainer = ORPOTrainer.builder()
                        .model(delegate.getModel())
                        .build();
                break;
            case "simpo":
                simpoTrainer = SimPOTrainer.builder()
                        .model(delegate.getModel())
                        .build();
                break;
            case "rloo":
            case "reinforce":
                rlooTrainer = RLOOTrainer.builder()
                        .model(delegate.getModel())
                        .build();
                break;
            default:
                // Use standard SFT
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
    public void close() throws Exception {
        if (closed) return;
        closed = true;

        if (orpoTrainer != null) orpoTrainer.close();
        if (simpoTrainer != null) simpoTrainer.close();
        if (rlooTrainer != null) rlooTrainer.close();
        delegate.close();

        System.out.println("[LlamaFactoryEnterpriseTrainer] Closed");
    }

    public boolean isClosed() { return closed; }

    /**
     * Get ORPO trainer (if configured).
     */
    public ORPOTrainer getOrpoTrainer() {
        return orpoTrainer;
    }

    /**
     * Get SimPO trainer (if configured).
     */
    public SimPOTrainer getSimpoTrainer() {
        return simpoTrainer;
    }

    /**
     * Get RLOO trainer (if configured).
     */
    public RLOOTrainer getRlooTrainer() {
        return rlooTrainer;
    }

    /**
     * Metrics callback for LlamaFactory.
     */
    private class FactoryMetricsCallback implements TrainingCallback {
        @Override
        public void onStepEnd(EnterpriseTrainer trainer, int step, TrainingMetrics metrics) {
            if (LlamaFactoryEnterpriseTrainer.this.metrics != null) {
                MetricsCollector mc = LlamaFactoryEnterpriseTrainer.this.metrics;
                mc.incrementCounter("training.steps");
                mc.gauge("training.loss").set(metrics.loss());
                mc.gauge("training.learning_rate").set(metrics.learningRate());
                mc.gauge("training.grad_norm").set(metrics.gradNorm());
                mc.gauge("training.throughput").set(metrics.throughputTokensPerSec());
            }
        }

        @Override
        public void onEpochEnd(EnterpriseTrainer trainer, int epoch, TrainingMetrics metrics) {
            System.out.printf("[LlamaFactory] Epoch %d: loss=%.4f, perplexity=%.2f%n",
                    epoch, metrics.loss(), Math.exp(metrics.loss()));
        }

        @Override
        public void onCheckpointSave(EnterpriseTrainer trainer, String path) {
            System.out.println("[LlamaFactory] Checkpoint saved: " + path);
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private FactoryArgs factoryArgs;
        private TrainerConfig config;
        private MetricsCollector metrics;

        public Builder factoryArgs(FactoryArgs args) { this.factoryArgs = args; return this; }
        public Builder config(TrainerConfig config) { this.config = config; return this; }
        public Builder trainerConfig(TrainerConfig config) { this.config = config; return this; }
        public Builder metrics(MetricsCollector metrics) { this.metrics = metrics; return this; }

        public LlamaFactoryEnterpriseTrainer build() {
            return new LlamaFactoryEnterpriseTrainer(this);
        }
    }
}
