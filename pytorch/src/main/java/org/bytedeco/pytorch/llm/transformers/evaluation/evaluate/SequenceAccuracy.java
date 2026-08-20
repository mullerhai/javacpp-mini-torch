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

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Sequence-level accuracy.
 *
 * <p>Full string/exact match accuracy (unlike token-level).
 * Expected keys: {@code "pred"} and {@code "ref"} containing sequences
 * (strings, int[], or lists).
 */
public final class SequenceAccuracy implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        if (predictions == null || references == null || predictions.size() != references.size()) {
            return Map.of("sequence_accuracy", 0.0);
        }

        int correct = 0;
        for (int i = 0; i < predictions.size(); i++) {
            if (equals(predictions.get(i).get("pred"), references.get(i).get("ref"))) {
                correct++;
            }
        }

        double acc = predictions.isEmpty() ? 0.0 : (double) correct / predictions.size();
        return Map.of("sequence_accuracy", acc);
    }

    private static boolean equals(Object a, Object b) {
        if (a == null && b == null) return true;
        if (a == null || b == null) return false;

        if (a instanceof String s && b instanceof String t) {
            return s.equals(t);
        }
        if (a instanceof int[] arrA && b instanceof int[] arrB) {
            if (arrA.length != arrB.length) return false;
            for (int i = 0; i < arrA.length; i++) if (arrA[i] != arrB[i]) return false;
            return true;
        }
        if (a instanceof List<?> listA && b instanceof List<?> listB) {
            if (listA.size() != listB.size()) return false;
            for (int i = 0; i < listA.size(); i++) {
                if (!equals(listA.get(i), listB.get(i))) return false;
            }
            return true;
        }
        return a.equals(b);
    }

    @Override
    public String description() {
        return "Full sequence exact-match accuracy.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' keys containing strings, int[], or Lists.";
    }
}
