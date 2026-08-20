/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "Classpath" exception),
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

import java.util.Map;

/**
 * Early-stopping callback mirroring HF's {@code EarlyStoppingCallback}.
 *
 * <p>Stops training when a monitored metric stops improving for
 * {@code earlyStoppingPatience} consecutive evaluations.
 */
public final class EarlyStoppingCallback implements TrainerCallback {

    private final int patience;
    private final String metricToMonitor;
    private final boolean greaterIsBetter;
    private int patienceCounter;
    private Double bestValue;

    /**
     * @param patience          number of evaluations without improvement before stopping
     * @param metricToMonitor   metric key to watch (e.g. {@code "eval_loss"})
     * @param greaterIsBetter   whether higher metric values are better
     */
    public EarlyStoppingCallback(int patience, String metricToMonitor, boolean greaterIsBetter) {
        this.patience = patience;
        this.metricToMonitor = metricToMonitor;
        this.greaterIsBetter = greaterIsBetter;
        this.patienceCounter = 0;
        this.bestValue = null;
    }

    public EarlyStoppingCallback() {
        this(3, "eval_loss", false);
    }

    @Override
    public TrainerControl onEvaluate(TrainingArguments args, TrainerState state,
                                    TrainerControl control, Map<String, Double> metrics) {
        if (!state.isLocalProcessZero()) return control;

        Double current = metrics.get(metricToMonitor);
        if (current == null) return control;

        boolean improved;
        if (bestValue == null) {
            improved = true;
        } else if (greaterIsBetter) {
            improved = current > bestValue;
        } else {
            improved = current < bestValue;
        }

        if (improved) {
            bestValue = current;
            patienceCounter = 0;
        } else {
            patienceCounter++;
            if (patienceCounter >= patience) {
                System.out.println("[EarlyStopping] No improvement for " + patienceCounter
                        + " evaluations — stopping training at step " + state.globalStep());
                control.setShouldTrainingStop(true);
            }
        }
        return control;
    }

    @Override
    public TrainerControl onTrainBegin(TrainingArguments args, TrainerState state, TrainerControl control) {
        patienceCounter = 0;
        bestValue = null;
        return control;
    }
}
