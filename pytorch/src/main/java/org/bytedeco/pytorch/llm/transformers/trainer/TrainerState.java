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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Training state tracked throughout a {@link Trainer} run.
 *
 * <p>Mirrors HF's {@code TrainerState} JSON blob: global step, epoch, log history,
 * best metric, and distributed-process flags.
 */
public final class TrainerState {

    private long globalStep;
    private double epoch;
    private int maxSteps;
    private final List<Map<String, Object>> logHistory;
    private Double bestMetric;
    private boolean isLocalProcessZero;
    private boolean isWorldProcessZero;

    public TrainerState() {
        this.globalStep = 0;
        this.epoch = 0.0;
        this.maxSteps = -1;
        this.logHistory = new ArrayList<>();
        this.bestMetric = null;
        this.isLocalProcessZero = true;
        this.isWorldProcessZero = true;
    }

    public TrainerState(long globalStep, double epoch, int maxSteps) {
        this.globalStep = globalStep;
        this.epoch = epoch;
        this.maxSteps = maxSteps;
        this.logHistory = new ArrayList<>();
        this.bestMetric = null;
        this.isLocalProcessZero = true;
        this.isWorldProcessZero = true;
    }

    public long globalStep() { return globalStep; }
    public double epoch() { return epoch; }
    public int maxSteps() { return maxSteps; }
    public List<Map<String, Object>> logHistory() { return logHistory; }
    public Double bestMetric() { return bestMetric; }
    public boolean isLocalProcessZero() { return isLocalProcessZero; }
    public boolean isWorldProcessZero() { return isWorldProcessZero; }

    public void setGlobalStep(long step) { this.globalStep = step; }
    public void setEpoch(double epoch) { this.epoch = epoch; }
    public void setMaxSteps(int steps) { this.maxSteps = steps; }
    public void setBestMetric(double m) { this.bestMetric = m; }
    public void setLocalProcessZero(boolean v) { this.isLocalProcessZero = v; }
    public void setWorldProcessZero(boolean v) { this.isWorldProcessZero = v; }

    public void log(Map<String, Object> metrics) {
        logHistory.add(Map.copyOf(metrics));
    }

    public void incrementGlobalStep() { this.globalStep++; }

    public void resetLogHistory() { this.logHistory.clear(); }
}
