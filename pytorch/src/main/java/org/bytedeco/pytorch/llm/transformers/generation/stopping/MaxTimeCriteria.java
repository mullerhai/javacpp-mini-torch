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
package org.bytedeco.pytorch.llm.transformers.generation.stopping;

import org.bytedeco.pytorch.Tensor;

/**
 * Stop after a wall-clock time budget elapses (HF {@code MaxTimeCriteria}).
 *
 * <p>Generator must invoke {@link #start()} when generation begins. We use
 * {@link System#nanoTime()} for monotonic timing.
 */
public final class MaxTimeCriteria extends StoppingCriteria {

    private final double maxSeconds;
    private long startNanos = -1L;

    public MaxTimeCriteria(double maxSeconds) {
        if (maxSeconds <= 0) throw new IllegalArgumentException("maxSeconds must be > 0");
        this.maxSeconds = maxSeconds;
    }

    public double maxSeconds() { return maxSeconds; }

    public void start() { this.startNanos = System.nanoTime(); }

    @Override
    public boolean call(Tensor inputIds, Tensor scores) {
        if (startNanos < 0) return false;
        double elapsed = (System.nanoTime() - startNanos) / 1e9;
        return elapsed >= maxSeconds;
    }
}