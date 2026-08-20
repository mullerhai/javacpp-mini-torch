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
package org.bytedeco.pytorch.llm.trl.data;

import org.bytedeco.pytorch.Tensor;

import java.util.List;
import java.util.Map;

/**
 * TRL-side collator producing {@code Map<String, Tensor>} batches for
 * {@link org.bytedeco.pytorch.llm.trl.trainer.BaseTrainer.BatchSupplier}.
 *
 * <p>Distinct from {@code transformers.trainer.data_collator.DataCollator}
 * which returns {@code List<Map>} of padded feature rows.
 */
@FunctionalInterface
public interface TrlDataCollator {

    /**
     * Collate a list of dataset rows into a single batched tensor map
     * ({@code input_ids}, {@code attention_mask}, {@code labels}).
     */
    Map<String, Tensor> collate(List<Map<String, Object>> features);
}
