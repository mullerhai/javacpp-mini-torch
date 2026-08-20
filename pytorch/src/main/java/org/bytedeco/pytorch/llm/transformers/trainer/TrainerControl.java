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
 * Control flags set by {@link TrainerCallback} handlers to steer the training loop.
 *
 * <p>Mirrors HF's {@code TrainerControl}. All flags default to {@code false}
 * except {@code shouldLog} which is {@code true}.
 */
public final class TrainerControl {

    private boolean shouldEvaluate;
    private boolean shouldSave;
    private boolean shouldTrainingStop;
    private boolean shouldLog;

    public TrainerControl() {
        this.shouldEvaluate = false;
        this.shouldSave = false;
        this.shouldTrainingStop = false;
        this.shouldLog = true;
    }

    public TrainerControl(boolean shouldEvaluate, boolean shouldSave,
                         boolean shouldTrainingStop, boolean shouldLog) {
        this.shouldEvaluate = shouldEvaluate;
        this.shouldSave = shouldSave;
        this.shouldTrainingStop = shouldTrainingStop;
        this.shouldLog = shouldLog;
    }

    public boolean shouldEvaluate() { return shouldEvaluate; }
    public boolean shouldSave() { return shouldSave; }
    public boolean shouldTrainingStop() { return shouldTrainingStop; }
    public boolean shouldLog() { return shouldLog; }

    public void setShouldEvaluate(boolean v) { this.shouldEvaluate = v; }
    public void setShouldSave(boolean v) { this.shouldSave = v; }
    public void setShouldTrainingStop(boolean v) { this.shouldTrainingStop = v; }
    public void setShouldLog(boolean v) { this.shouldLog = v; }

    public TrainerControl withEvaluate(boolean v) {
        TrainerControl c = new TrainerControl(shouldEvaluate, shouldSave, shouldTrainingStop, shouldLog);
        c.setShouldEvaluate(v);
        return c;
    }

    public TrainerControl withSave(boolean v) {
        TrainerControl c = new TrainerControl(shouldEvaluate, shouldSave, shouldTrainingStop, shouldLog);
        c.setShouldSave(v);
        return c;
    }

    public TrainerControl withTrainingStop(boolean v) {
        TrainerControl c = new TrainerControl(shouldEvaluate, shouldSave, shouldTrainingStop, shouldLog);
        c.setShouldTrainingStop(v);
        return c;
    }

    public TrainerControl withLog(boolean v) {
        TrainerControl c = new TrainerControl(shouldEvaluate, shouldSave, shouldTrainingStop, shouldLog);
        c.setShouldLog(v);
        return c;
    }
}
