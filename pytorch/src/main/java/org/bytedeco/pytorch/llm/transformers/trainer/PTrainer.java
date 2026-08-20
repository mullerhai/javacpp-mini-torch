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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

import java.util.List;
import java.util.Map;

/**
 * Abstract base class for trainers, providing the {@link #prediction_step} hook.
 *
 * <p>Sub-classes (e.g. {@link Trainer} and {@link Seq2SeqTrainer}) override
 * {@link #prediction_step} to implement model-specific inference logic.
 */
public abstract class PTrainer {

    /**
     * Result of a single prediction / evaluation step.
     */
    public record PredictionOutput(
            Tensor predictions,
            Tensor labelIds,
            Map<String, Double> metrics
    ) {}

    /**
     * Run a single forward pass to produce predictions.
     *
     * <p>The default stub throws {@link UnsupportedOperationException}.
     * Sub-classes implement real model evaluation.
     *
     * @param model               the model to evaluate
     * @param inputs              batch features as a map of column name → tensor
     * @param predictionLossOnly  if true, only return the loss tensor
     * @param ignoreKeys          parameter names to ignore (may be null)
     * @return prediction output containing logits and optionally label ids
     */
    public PredictionOutput prediction_step(
            Module model,
            Map<String, Object> inputs,
            boolean predictionLossOnly,
            List<String> ignoreKeys) {
        throw new UnsupportedOperationException(
                "prediction_step must be implemented by a sub-class of PTrainer");
    }

    /**
     * Compute a scalar metric from model predictions and labels.
     *
     * <p>Override in sub-classes for domain-specific evaluation.
     *
     * @param preds predicted tensors
     * @param labels ground-truth tensors
     * @return metric name → value map
     */
    public Map<String, Double> computeMetrics(Tensor preds, Tensor labels) {
        return Map.of();
    }
}
