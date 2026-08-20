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

import java.util.Map;

/**
 * Default callback implementing the HF-style default flow control:
 * sets {@code shouldEvaluate = true} at epoch end and {@code shouldSave = true}
 * at step end when the respective strategy matches.
 */
public final class DefaultFlowCallback implements TrainerCallback {

    @Override
    public TrainerControl onEpochEnd(TrainingArguments args, TrainerState state, TrainerControl control) {
        if ("epoch".equalsIgnoreCase(args.evalStrategy()) && state.isLocalProcessZero()) {
            control.setShouldEvaluate(true);
        }
        if ("epoch".equalsIgnoreCase(args.saveStrategy()) && state.isLocalProcessZero()) {
            control.setShouldSave(true);
        }
        return control;
    }

    @Override
    public TrainerControl onStepEnd(TrainingArguments args, TrainerState state, TrainerControl control) {
        if ("steps".equalsIgnoreCase(args.evalStrategy())
                && args.evalSteps() > 0
                && state.globalStep() % args.evalSteps() == 0
                && state.isLocalProcessZero()) {
            control.setShouldEvaluate(true);
        }
        if ("steps".equalsIgnoreCase(args.saveStrategy())
                && args.saveSteps() > 0
                && state.globalStep() % args.saveSteps() == 0
                && state.isLocalProcessZero()) {
            control.setShouldSave(true);
        }
        return control;
    }

    @Override
    public TrainerControl onEvaluate(TrainingArguments args, TrainerState state,
                                    TrainerControl control, Map<String, Double> metrics) {
        control.setShouldEvaluate(false);
        return control;
    }

    @Override
    public TrainerControl onSave(TrainingArguments args, TrainerState state, TrainerControl control) {
        control.setShouldSave(false);
        return control;
    }
}
