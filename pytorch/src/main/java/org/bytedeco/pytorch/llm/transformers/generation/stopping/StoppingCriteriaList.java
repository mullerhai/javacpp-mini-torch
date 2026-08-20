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

import java.util.ArrayList;
import java.util.List;

/**
 * HF-style stopping criteria container.
 *
 * <p>Mirrors {@code transformers/generation/stopping_criteria.py:StoppingCriteriaList}.
 * Each criterion receives the current input ids and the latest scores; it
 * returns {@code true} to halt generation.
 *
 * <p>Provided criteria:
 * <ul>
 *   <li>{@link MaxLengthCriteria}</li>
 *   <li>{@link MaxNewTokensCriteria}</li>
 *   <li>{@link MaxTimeCriteria}</li>
 *   <li>{@link EosTokenCriteria}</li>
 *   <li>{@link StopSequenceCriteria}</li>
 * </ul>
 */
public class StoppingCriteriaList extends ArrayList<StoppingCriteria> {

    private static final long serialVersionUID = 1L;

    public StoppingCriteriaList() {}

    public StoppingCriteriaList(Iterable<StoppingCriteria> seed) {
        for (StoppingCriteria c : seed) add(c);
    }

    public boolean call(Tensor inputIds, Tensor scores) {
        for (StoppingCriteria c : this) {
            if (c.call(inputIds, scores)) return true;
        }
        return false;
    }

    /**
     * @return maximum {@link StoppingCriteria#getMaxLength()} across all
     *         criteria, or {@link Integer#MAX_VALUE} when none.
     */
    public int maxLength() {
        int max = Integer.MAX_VALUE;
        for (StoppingCriteria c : this) {
            int m = c.getMaxLength();
            if (m < max) max = m;
        }
        return max;
    }
}
