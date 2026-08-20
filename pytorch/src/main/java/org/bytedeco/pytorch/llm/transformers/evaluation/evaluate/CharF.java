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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Character n-gram F-score (chrF).
 *
 * <p>Combines character n-gram precision and recall (default n=6).
 * Expected keys: {@code "pred"} and {@code "ref"} containing text strings.
 */
public final class CharF implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        final int n = 6;
        double totalClip = 0, totalCand = 0, totalRef = 0;

        for (int i = 0; i < predictions.size(); i++) {
            String pred = String.valueOf(predictions.get(i).get("pred"));
            String ref = String.valueOf(references.get(i).get("ref"));

            Map<String, Integer> cngramPred = charNgrams(pred, n);
            Map<String, Integer> cngramRef = charNgrams(ref, n);

            int candLen = cngramPred.values().stream().mapToInt(Integer::intValue).sum();
            int refLen = cngramRef.values().stream().mapToInt(Integer::intValue).sum();

            int clipped = 0;
            for (Map.Entry<String, Integer> e : cngramPred.entrySet()) {
                int rc = cngramRef.getOrDefault(e.getKey(), 0);
                clipped += Math.min(e.getValue(), rc);
            }

            totalClip += clipped;
            totalCand += candLen;
            totalRef += refLen;
        }

        double p = totalCand > 0 ? totalClip / totalCand : 0;
        double r = totalRef > 0 ? totalClip / totalRef : 0;
        double chrf = (p + r) > 0 ? 2 * p * r / (p + r) : 0;

        return Map.of("chrf", chrf);
    }

    private static Map<String, Integer> charNgrams(String text, int n) {
        Map<String, Integer> map = new HashMap<>();
        String t = text.replaceAll("\\s+", "").toLowerCase();
        for (int i = 0; i <= t.length() - n; i++) {
            String ng = t.substring(i, i + n);
            map.merge(ng, 1, Integer::sum);
        }
        return map;
    }

    @Override
    public String description() {
        return "chrF: character n-gram F-score (default n=6).";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' keys containing text strings.";
    }
}
