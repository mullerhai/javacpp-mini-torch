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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * F1 score metric with per-label and averaged F1.
 *
 * <p>F1 = 2 * P * R / (P + R). Falls back to 0 when denominator is 0.
 * Expected keys: {@code "pred"} and {@code "ref"} in each map.
 */
public final class F1 implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        Map<String, Double> result = new HashMap<>();

        Map<Object, Integer> tp = new HashMap<>();
        Map<Object, Integer> fp = new HashMap<>();
        Map<Object, Integer> fn = new HashMap<>();

        for (int i = 0; i < predictions.size(); i++) {
            Object pred = predictions.get(i).get("pred");
            Object ref = references.get(i).get("ref");
            if (pred.equals(ref)) {
                tp.merge(pred, 1, Integer::sum);
            } else {
                fp.merge(pred, 1, Integer::sum);
                fn.merge(ref, 1, Integer::sum);
            }
        }

        List<Double> f1s = new ArrayList<>();
        for (Object label : tp.keySet()) {
            int truePos = tp.getOrDefault(label, 0);
            int falsePos = fp.getOrDefault(label, 0);
            int falseNeg = fn.getOrDefault(label, 0);
            double p = truePos + falsePos > 0 ? (double) truePos / (truePos + falsePos) : 0.0;
            double r = truePos + falseNeg > 0 ? (double) truePos / (truePos + falseNeg) : 0.0;
            double f = (p + r) > 0 ? 2 * p * r / (p + r) : 0.0;
            result.put("f1:" + label, f);
            f1s.add(f);
        }

        int totalTp = tp.values().stream().mapToInt(Integer::intValue).sum();
        int totalFp = fp.values().stream().mapToInt(Integer::intValue).sum();
        int totalFn = fn.values().stream().mapToInt(Integer::intValue).sum();
        double pMicro = totalTp + totalFp > 0 ? (double) totalTp / (totalTp + totalFp) : 0.0;
        double rMicro = totalTp + totalFn > 0 ? (double) totalTp / (totalTp + totalFn) : 0.0;
        result.put("f1_micro", (pMicro + rMicro) > 0 ? 2 * pMicro * rMicro / (pMicro + rMicro) : 0.0);
        result.put("f1_macro", f1s.stream().mapToDouble(Double::doubleValue).average().orElse(0.0));

        return result;
    }

    @Override
    public String description() {
        return "F1 score: harmonic mean of precision and recall.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' keys.";
    }
}
