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
 * SQuAD evaluation metrics (EM and F1).
 *
 * <p>Expected keys: {@code "pred"} (answer string) and {@code "ref"} (answer string
 * or Map with "text" key). Computes exact match and token-level F1.
 */
public final class Squad implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        int em = 0, f1Sum = 0;

        for (int i = 0; i < predictions.size(); i++) {
            String pred = normalize(getText(predictions.get(i).get("pred")));
            String ref = normalize(getText(references.get(i).get("ref")));

            if (pred.equals(ref)) {
                em++;
                f1Sum += 100;
            } else {
                double f1 = computeF1(pred, ref);
                f1Sum += f1;
            }
        }

        int n = Math.max(predictions.size(), 1);
        Map<String, Double> result = new HashMap<>();
        result.put("exact_match", (double) em / n);
        result.put("f1", (double) f1Sum / n);
        return result;
    }

    private static String getText(Object obj) {
        if (obj instanceof String s) return s;
        if (obj instanceof Map<?, ?> m && m.containsKey("text")) {
            Object t = m.get("text");
            return t != null ? t.toString() : "";
        }
        return obj != null ? obj.toString() : "";
    }

    private static String normalize(String s) {
        return s.trim().toLowerCase().replaceAll("\\s+", " ");
    }

    private static double computeF1(String pred, String ref) {
        String[] pTokens = pred.split("\\s+");
        String[] rTokens = ref.split("\\s+");

        java.util.Set<String> pSet = new java.util.HashSet<>();
        java.util.Set<String> rSet = new java.util.HashSet<>();
        for (String t : pTokens) if (!t.isBlank()) pSet.add(t);
        for (String t : rTokens) if (!t.isBlank()) rSet.add(t);

        int match = 0;
        for (String t : pSet) if (rSet.contains(t)) match++;

        double p = pSet.isEmpty() ? 0 : (double) match / pSet.size();
        double r = rSet.isEmpty() ? 0 : (double) match / rSet.size();
        return (p + r) > 0 ? 2 * p * r / (p + r) * 100 : 0;
    }

    @Override
    public String description() {
        return "SQuAD EM (exact match) and token-level F1 score for question answering.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' keys containing answer text.";
    }
}
