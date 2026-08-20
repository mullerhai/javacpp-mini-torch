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
 * METEOR score (Metric for Evaluation of Translation with Explicit ORdering).
 *
 * <p>Combines unigram precision and recall with a fragmentation penalty based on
 * chunks (consecutive matching unigrams).
 * Expected keys: {@code "pred"} and {@code "ref"} containing text strings.
 */
public final class Meteor implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        double sumMeteor = 0;

        for (int i = 0; i < predictions.size(); i++) {
            String pred = String.valueOf(predictions.get(i).get("pred"));
            String ref = String.valueOf(references.get(i).get("ref"));

            String[] pWords = pred.toLowerCase().trim().split("\\s+");
            String[] rWords = ref.toLowerCase().trim().split("\\s+");

            Set<String> unigramsRef = new HashSet<>();
            for (String w : rWords) if (!w.isBlank()) unigramsRef.add(w);

            int matches = 0;
            for (String w : pWords) if (!w.isBlank() && unigramsRef.contains(w)) matches++;

            double precision = pWords.length > 0 ? (double) matches / pWords.length : 0;
            double recall = rWords.length > 0 ? (double) matches / rWords.length : 0;
            double fMean = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;

            int chunks = countChunks(pWords, rWords);
            int uniqMatches = matches;
            double fragPenalty = chunks > 0 && uniqMatches > 0
                    ? 0.5 * Math.pow(chunks / (double) uniqMatches, 3)
                    : 0;

            double meteor = fMean * (1.0 - fragPenalty);
            sumMeteor += meteor;
        }

        int n = Math.max(predictions.size(), 1);
        return Map.of("meteor", sumMeteor / n);
    }

    private static int countChunks(String[] pred, String[] ref) {
        Set<String> matchedRef = new HashSet<>();
        int chunks = 0;
        int prevMatchIdx = -1;

        for (String w : pred) {
            if (w.isBlank()) continue;
            for (int j = 0; j < ref.length; j++) {
                if (!ref[j].isBlank() && w.equals(ref[j]) && !matchedRef.contains(j + ":" + ref[j])) {
                    matchedRef.add(j + ":" + ref[j]);
                    if (prevMatchIdx != j - 1) chunks++;
                    prevMatchIdx = j;
                    break;
                }
            }
        }
        return chunks;
    }

    @Override
    public String description() {
        return "METEOR: unigram precision/recall with fragmentation penalty.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' keys containing text strings.";
    }
}
