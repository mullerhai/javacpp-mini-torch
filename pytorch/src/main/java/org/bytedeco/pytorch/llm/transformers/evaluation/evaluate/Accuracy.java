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
 * Accuracy metric (top-1).
 *
 * <p>Computes exact-match accuracy between predictions and references.
 * Expected keys: {@code "pred"} and {@code "ref"} in each map.
 */
public final class Accuracy implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        if (predictions == null || references == null || predictions.size() != references.size()) {
            throw new IllegalArgumentException("predictions and references must be non-null and have equal size");
        }
        int correct = 0;
        for (int i = 0; i < predictions.size(); i++) {
            Object pred = predictions.get(i).get("pred");
            Object ref = references.get(i).get("ref");
            if (pred != null && pred.equals(ref)) {
                correct++;
            }
        }
        double acc = predictions.isEmpty() ? 0.0 : (double) correct / predictions.size();
        return Map.of("accuracy", acc);
    }

    @Override
    public String description() {
        return "Top-1 classification accuracy.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' keys.";
    }
}
