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
package org.bytedeco.pytorch.llm.deepspeed;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer;
import org.bytedeco.pytorch.llm.trainer.TrainerConfig;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer.TrainingCallback;
import org.bytedeco.pytorch.llm.trainer.EnterpriseTrainer.TrainingMetrics;

import java.util.concurrent.atomic.AtomicReference;

/**
 * DeepSpeed integration with EnterpriseTrainer.
 *
 * <p>Bridges DeepSpeed ZeRO optimization with the enterprise training interface.
 *
 * <pre>{@code
 * DeepSpeedEnterpriseTrainer trainer = DeepSpeedEnterpriseTrainer.builder()
 *     .module(model)
 *     .optimizer(optimizer)
 *     .deepSpeedConfig(dsConfig)
 *     .trainerConfig(trainerConfig)
 *     .build();
 *
 * trainer.train();
 * }</pre>
 */
public class DeepSpeedEnterpriseTrainer implements EnterpriseTrainer {

    private volatile boolean closed;

    // Components
    private final DeepSpeedEngine engine;
    private final EnterpriseTrainer delegate;

    // DeepSpeed-specific config
    private final DeepSpeedConfig dsConfig;

    // State
    private final AtomicReference<TrainingState> state = new AtomicReference<>(TrainingState.IDLE);
    private int globalStep = 0;

    public static Builder builder() {
        return new Builder();
    }

    private DeepSpeedEnterpriseTrainer(Builder builder) {
        this.dsConfig = builder.dsConfig != null ? builder.dsConfig : DeepSpeedConfig.defaults();

        // Create DeepSpeed engine
        this.engine = new DeepSpeedEngine(
                builder.module,
                builder.optimizer,
                this.dsConfig,
                null  // Process group
        );

        // Create delegate trainer
        this.delegate = EnterpriseTrainer.builder()
                .model(builder.module)
                .optimizer(builder.optimizer)
                .config(builder.trainerConfig)
                .build();

        // Add DeepSpeed-specific callbacks
        if (builder.enableDsCallbacks) {
            this.delegate.addCallback(new DeepSpeedCallback());
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
    public void eval(java.util.List<String> metrics) {
        delegate.eval(metrics);
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
    public int getGlobalStep() {
        return Math.max(globalStep, delegate.getGlobalStep());
    }

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
        try {
            engine.saveCheckpoint(java.nio.file.Paths.get(path));
        } catch (Exception e) {
            System.err.println("DeepSpeed save_checkpoint failed: " + e.getMessage());
        }
        delegate.saveCheckpoint(path);
    }

    @Override
    public void loadCheckpoint(String path) {
        try {
            engine.loadCheckpoint(java.nio.file.Paths.get(path));
        } catch (Exception e) {
            System.err.println("DeepSpeed load_checkpoint failed: " + e.getMessage());
        }
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
        delegate.close();
        engine.close();
    }

    public boolean isClosed() { return closed; }

    /**
     * Get DeepSpeed engine.
     */
    public DeepSpeedEngine getEngine() {
        return engine;
    }

    /**
     * Get zero stage.
     */
    public int getZeroStage() {
        return engine.zeroStage();
    }

    /**
     * Check if parameters are partitioned.
     */
    public boolean isPartitioned() {
        return !engine.isGathered();
    }

    /**
     * DeepSpeed-specific callback.
     */
    private class DeepSpeedCallback implements TrainingCallback {
        @Override
        public void onStepEnd(EnterpriseTrainer trainer, int step, TrainingMetrics metrics) {
            globalStep = step;

            // Log DeepSpeed-specific metrics
            if (step % 100 == 0) {
                System.out.printf("[DeepSpeed] Step %d: loss=%.4f, gradNorm=%.4f, zeroStage=%d%n",
                        step, metrics.loss(), metrics.gradNorm(), engine.zeroStage());
            }
        }

        @Override
        public void onCheckpointSave(EnterpriseTrainer trainer, String path) {
            try {
                engine.saveCheckpoint(java.nio.file.Paths.get(path));
            } catch (Exception e) {
                System.err.println("DeepSpeed save_checkpoint failed: " + e.getMessage());
            }
            System.out.println("[DeepSpeed] Checkpoint saved: " + path);
        }

        @Override
        public void onCheckpointLoad(EnterpriseTrainer trainer, String path) {
            try {
                engine.loadCheckpoint(java.nio.file.Paths.get(path));
            } catch (Exception e) {
                System.err.println("DeepSpeed load_checkpoint failed: " + e.getMessage());
            }
            System.out.println("[DeepSpeed] Checkpoint loaded: " + path);
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private Module module;
        private Optimizer optimizer;
        private DeepSpeedConfig dsConfig;
        private TrainerConfig trainerConfig;
        private boolean enableDsCallbacks = true;

        public Builder module(Module module) { this.module = module; return this; }
        public Builder optimizer(Optimizer optimizer) { this.optimizer = optimizer; return this; }
        public Builder deepSpeedConfig(DeepSpeedConfig dsConfig) { this.dsConfig = dsConfig; return this; }
        public Builder trainerConfig(TrainerConfig config) { this.trainerConfig = config; return this; }
        public Builder enableDsCallbacks(boolean enable) { this.enableDsCallbacks = enable; return this; }

        public DeepSpeedEnterpriseTrainer build() {
            return new DeepSpeedEnterpriseTrainer(this);
        }
    }
}
