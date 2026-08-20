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

/**
 * Progress-tracking callback that prints a simple ASCII progress bar
 * (e.g. {@code [####----] 40%}) during training.
 */
public final class ProgressCallback implements TrainerCallback {

    private int lastPercent = -1;

    @Override
    public TrainerControl onStepBegin(TrainingArguments args, TrainerState state, TrainerControl control) {
        if (!state.isLocalProcessZero()) return control;

        int totalSteps = args.maxSteps() > 0 ? args.maxSteps() : 100;
        int current = (int) state.globalStep();
        int percent = Math.min(100, (current * 100) / totalSteps);

        if (percent != lastPercent) {
            int bars = percent / 5;
            int dashes = 20 - bars;
            StringBuilder bar = new StringBuilder("[");
            for (int i = 0; i < bars; i++) bar.append('#');
            for (int i = 0; i < dashes; i++) bar.append('-');
            bar.append("] ").append(percent).append("% step=").append(current);
            System.out.println(bar);
            lastPercent = percent;
        }
        return control;
    }

    @Override
    public TrainerControl onTrainBegin(TrainingArguments args, TrainerState state, TrainerControl control) {
        lastPercent = -1;
        return control;
    }
}
