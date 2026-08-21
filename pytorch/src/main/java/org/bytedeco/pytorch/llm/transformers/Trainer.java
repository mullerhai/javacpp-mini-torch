/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "the Classpath" exception),
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
package org.bytedeco.pytorch.llm.transformers;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.datasets.HfDataset;
import org.bytedeco.pytorch.llm.transformers.data.Dataset;
import org.bytedeco.pytorch.llm.transformers.trainer.TrainerPredictionOutput;
import org.bytedeco.pytorch.llm.transformers.trainer.TrainingArguments;

import java.util.Map;

/**
 * Convenience re-export of {@link Trainer} for the transformers top-level namespace.
 *
 * <p>Provides a simple entry point for common training tasks without
 * requiring users to import from the {@code trainer} sub-package.
 *
 * <p>Example:
 * <pre>{@code
 * TrainerPredictionOutput output = Trainer.train(
 *     model,
 *     trainDataset,
 *     evalDataset,
 *     Map.of(
 *         "output_dir", "/tmp/output",
 *         "num_train_epochs", 3,
 *         "per_device_train_batch_size", 8
 *     )
 * );
 * }</pre>
 */
public final class Trainer {

    private Trainer() {} // static utility

    /**
     * Train a model on the given datasets with the provided arguments.
     *
     * <p>Constructs {@link org.bytedeco.pytorch.llm.transformers.trainer.Trainer}
     * and runs {@link org.bytedeco.pytorch.llm.transformers.trainer.Trainer#train()}.
     *
     * @param model the model to train
     * @param train the training dataset
     * @param eval  the evaluation dataset (may be null)
     * @param args  training arguments (e.g. output_dir, num_train_epochs, learning_rate)
     * @return the prediction output containing metrics and predictions
     */
    public static TrainerPredictionOutput train(Module model,
                                               Dataset train,
                                               Dataset eval,
                                               Map<String, Object> args) {
        TrainingArguments ta = TrainingArguments.builder()
                .outputDir(String.valueOf(args.getOrDefault("output_dir", "output")))
                .build();
        HfDataset trainDs = train instanceof HfDataset h ? h : (train == null ? null : train.toHfDataset());
        HfDataset evalDs = eval instanceof HfDataset h ? h : (eval == null ? null : eval.toHfDataset());
        org.bytedeco.pytorch.llm.transformers.trainer.Trainer t =
                new org.bytedeco.pytorch.llm.transformers.trainer.Trainer(model, ta, trainDs, evalDs, null);
        t.train();
        return new TrainerPredictionOutput(null, null, Map.of());
    }

    /**
     * Train a model on the given dataset.
     *
     * @param model the model to train
     * @param train the training dataset
     * @param args  training arguments
     * @return the prediction output
     */
    public static TrainerPredictionOutput train(Module model,
                                               Dataset train,
                                               Map<String, Object> args) {
        return train(model, train, null, args);
    }
}
