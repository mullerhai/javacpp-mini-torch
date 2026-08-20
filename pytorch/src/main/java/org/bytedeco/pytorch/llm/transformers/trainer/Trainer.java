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
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.transformers.trainer;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.datasets.HfDataset;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;

/**
 * HuggingFace-style training loop for PyTorch models.
 *
 * <p>Mirrors {@code transformers.Trainer}. Configure with {@link TrainingArguments},
 * register callbacks, and call {@link #train()}.
 *
 * <pre>{@code
 * Trainer trainer = new Trainer(model, args, trainDataset, evalDataset, tokenizer,
 *     List.of(new PrinterCallback(), new EarlyStoppingCallback()));
 * trainer.train();
 * trainer.save_model(Path.of("output/checkpoint-1000"));
 * }</pre>
 */
public class Trainer {

    protected final Module model;
    protected final TrainingArguments args;
    protected final HfDataset train;
    protected final HfDataset eval;
    protected final FastTokenizer tokenizer;
    protected final List<TrainerCallback> callbacks;
    protected final ComputeMetrics computeMetrics;
    protected TrainerState state;

    /**
     * Functional interface for computing evaluation metrics.
     */
    @FunctionalInterface
    public interface ComputeMetrics {
        Map<String, Double> apply(TrainerPredictionOutput predictions);
    }

    public Trainer(Module model, TrainingArguments args, HfDataset train, HfDataset eval,
                   FastTokenizer tokenizer, List<TrainerCallback> callbacks,
                   ComputeMetrics computeMetrics) {
        this.model = Objects.requireNonNull(model, "model");
        this.args = Objects.requireNonNull(args, "args");
        this.train = train;
        this.eval = eval;
        this.tokenizer = tokenizer;
        this.callbacks = callbacks == null ? new ArrayList<>() : new ArrayList<>(callbacks);
        this.computeMetrics = computeMetrics;
        this.state = new TrainerState();
    }

    public Trainer(Module model, TrainingArguments args, HfDataset train, HfDataset eval,
                   FastTokenizer tokenizer, List<TrainerCallback> callbacks) {
        this(model, args, train, eval, tokenizer, callbacks, null);
    }

    public Trainer(Module model, TrainingArguments args, HfDataset train, HfDataset eval,
                   FastTokenizer tokenizer) {
        this(model, args, train, eval, tokenizer, new ArrayList<>(), null);
    }

    /**
     * Run the full training loop.
     */
    public void train() {
        if (!state.isLocalProcessZero()) return;

        args.validate();

        // Fire init callbacks
        TrainerControl control = new TrainerControl();
        for (TrainerCallback cb : callbacks) {
            control = cb.onInit(args, state, control);
        }
        control = fireOnTrainBegin(control);

        System.out.println("*** Start training ***");
        System.out.println("  num_examples_train = " + (train == null ? 0 : train.size()));
        System.out.println("  num_examples_eval  = " + (eval == null ? 0 : eval.size()));
        System.out.println("  num_train_epochs  = " + args.numTrainEpochs());
        System.out.println("  per_device_train_batch_size = " + args.perDeviceTrainBatchSize());
        System.out.println("  learning_rate     = " + args.learningRate());

        // Stub: real implementation would iterate over epochs, steps, batches,
        // accumulate gradients, call callbacks, evaluate, checkpoint, etc.
        throw new UnsupportedOperationException(
                "Trainer.train() is a stub — real training loop wiring (optimizer, scheduler, " +
                "DataLoader, gradient accumulation) is not yet implemented. " +
                "Use PTrainer sub-class for custom training logic.");
    }

    /**
     * Run evaluation on the eval dataset.
     *
     * @return metrics map
     */
    public Map<String, Double> evaluate() {
        if (!state.isLocalProcessZero()) return Map.of();
        if (eval == null) {
            System.out.println("*** No eval dataset — skipping evaluation ***");
            return Map.of();
        }
        TrainerControl control = new TrainerControl();
        System.out.println("*** Evaluate — running evaluation loop ***");
        // Stub: real implementation would run prediction_step over the eval dataloader
        Map<String, Double> metrics = new HashMap<>();
        metrics.put("eval_loss", Double.NaN);
        metrics.put("eval_runtime", 0.0);
        control = fireOnEvaluate(control, metrics);
        return metrics;
    }

    /**
     * Run predictions on a dataset.
     *
     * @param dataset dataset to predict on
     * @return prediction output
     */
    public TrainerPredictionOutput predict(HfDataset dataset) {
        if (!state.isLocalProcessZero()) return null;
        System.out.println("*** Predict — running prediction loop ***");
        // Stub: real implementation would call prediction_step for each batch
        throw new UnsupportedOperationException(
                "Trainer.predict() is a stub — implement prediction_step in a sub-class " +
                "or use Seq2SeqTrainer for generation-based prediction.");
    }

    /**
     * Save the model to a directory.
     *
     * @param outputDir directory to save to
     */
    public void save_model(Path outputDir) {
        if (!state.isLocalProcessZero()) return;
        System.out.println("*** Saving model to " + outputDir + " ***");
        // Stub: real implementation would serialise model state_dict + config
        throw new UnsupportedOperationException(
                "Trainer.save_model() is a stub — implement model serialization (state_dict, " +
                "safetensors, config.json) for your use case.");
    }

    /**
     * Log accumulated metrics.
     */
    public void log_metrics() {
        if (!state.isLocalProcessZero()) return;
        System.out.println("*** log_metrics — log history ***");
        for (Map<String, Object> entry : state.logHistory()) {
            System.out.println("  step=" + entry);
        }
    }

    /**
     * Push the model to the HuggingFace Hub.
     */
    public void push_to_hub() {
        if (!state.isLocalProcessZero()) return;
        if (!args.pushToHub()) {
            System.out.println("push_to_hub is disabled in TrainingArguments");
            return;
        }
        // Stub: real implementation would use HfApi.upload_folder or similar
        throw new UnsupportedOperationException(
                "Trainer.push_to_hub() is a stub — implement Hub upload using " +
                "org.bytedeco.pytorch.llm.hub.HfApi.");
    }

    public TrainerState state() { return state; }
    public TrainingArguments args() { return args; }
    public Module model() { return model; }
    public FastTokenizer tokenizer() { return tokenizer; }
    public List<TrainerCallback> callbacks() { return List.copyOf(callbacks); }

    public void addCallback(TrainerCallback cb) { callbacks.add(cb); }
    public void removeCallback(TrainerCallback cb) { callbacks.remove(cb); }

    // -------------------------------------------------------------------------
    // Internal callback helpers
    // -------------------------------------------------------------------------

    protected TrainerControl fireOnTrainBegin(TrainerControl control) {
        for (TrainerCallback cb : callbacks) {
            control = cb.onTrainBegin(args, state, control);
        }
        return control;
    }

    protected TrainerControl fireOnEpochBegin(int epoch, TrainerControl control) {
        TrainerControl c = control;
        for (TrainerCallback cb : callbacks) {
            c = cb.onEpochBegin(args, state, c);
        }
        return c;
    }

    protected TrainerControl fireOnEpochEnd(TrainerControl control) {
        TrainerControl c = control;
        for (TrainerCallback cb : callbacks) {
            c = cb.onEpochEnd(args, state, c);
        }
        return c;
    }

    protected TrainerControl fireOnStepBegin(TrainerControl control) {
        TrainerControl c = control;
        for (TrainerCallback cb : callbacks) {
            c = cb.onStepBegin(args, state, c);
        }
        return c;
    }

    protected TrainerControl fireOnStepEnd(TrainerControl control) {
        TrainerControl c = control;
        for (TrainerCallback cb : callbacks) {
            c = cb.onStepEnd(args, state, c);
        }
        return c;
    }

    protected TrainerControl fireOnEvaluate(TrainerControl control, Map<String, Double> metrics) {
        TrainerControl c = control;
        for (TrainerCallback cb : callbacks) {
            c = cb.onEvaluate(args, state, c, metrics);
        }
        return c;
    }

    protected TrainerControl fireOnSave(TrainerControl control) {
        TrainerControl c = control;
        for (TrainerCallback cb : callbacks) {
            c = cb.onSave(args, state, c);
        }
        return c;
    }

    protected TrainerControl fireOnLog(TrainerControl control, Map<String, Object> logs) {
        TrainerControl c = control;
        for (TrainerCallback cb : callbacks) {
            c = cb.onLog(args, state, c, logs);
        }
        return c;
    }
}
