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
package org.bytedeco.pytorch.llm.transformers.evaluation.evaluate;

import java.util.List;
import java.util.Map;

/**
 * Interface for evaluation metrics.
 *
 * <p>Reference: HuggingFace {@code evaluate} library metric interface.
 */
public interface Metric {

    /**
     * Compute the metric from predictions and references.
     *
     * @param predictions list of prediction maps (e.g. {@code {"pred": value}})
     * @param references  list of reference maps (e.g. {@code {"ref": value}})
     * @return map of metric name to value
     */
    Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references);

    /**
     * Short description of what this metric measures.
     */
    String description();

    /**
     * Description of expected inputs (predictions/references format).
     */
    String inputsDescription();
}
