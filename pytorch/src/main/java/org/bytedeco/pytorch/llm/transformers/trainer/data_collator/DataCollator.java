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
package org.bytedeco.pytorch.llm.transformers.trainer.data_collator;

import java.util.List;
import java.util.Map;

/**
 * Interface for dynamic batching / collation of feature maps.
 *
 * <p>Mirrors HF's {@code DataCollator}. Implement {@link #collate_batch}
 * to stack variable-length sequences into fixed-size tensors with padding.
 */
public interface DataCollator {

    /**
     * Collate a list of feature maps into a single batched map.
     *
     * @param features list of feature maps from a dataset
     * @return batched map with tensors ready for model forward
     */
    List<Map<String, Object>> collate_batch(List<Map<String, Object>> features);
}
