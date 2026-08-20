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
 * SacreBLEU: standardized BLEU score (tokenization-independent).
 *
 * <p>This is a simplified pure-Java implementation using whitespace tokenization.
 * For production use, prefer the native sacrebleu library.
 * Expected keys: {@code "pred"} and {@code "ref"} containing text strings.
 */
public final class Sacrebleu implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        // SacreBLEU uses standardized tokenization; we fall back to whitespace split
        // and reuse the Bleu computation with a modified reference length heuristic.
        if (predictions == null || predictions.isEmpty()) {
            return Map.of("sacrebleu", 0.0);
        }

        int totalCandLen = 0;
        int totalRefLen = 0;
        int n = predictions.size();

        for (int i = 0; i < n; i++) {
            String pred = String.valueOf(predictions.get(i).get("pred"));
            String ref = String.valueOf(references.get(i).get("ref"));

            String[] pWords = pred.trim().split("\\s+");
            String[] rWords = ref.trim().split("\\s+");

            totalCandLen += pWords.length;
            // Use shortest reference length for sacrebleu compatibility
            totalRefLen += rWords.length;
        }

        // Simplified: reuse Bleu-like precision
        int[][] clippedCounts = new int[4][n];
        int[][] candCounts = new int[4][n];

        for (int i = 0; i < n; i++) {
            String pred = String.valueOf(predictions.get(i).get("pred"));
            String ref = String.valueOf(references.get(i).get("ref"));

            String[] pWords = pred.toLowerCase().split("\\s+");
            String[] rWords = ref.toLowerCase().split("\\s+");

            for (int k = 1; k <= 4; k++) {
                Map<String, Integer> candNgram = ngrams(pWords, k);
                Map<String, Integer> refNgram = ngrams(rWords, k);

                candCounts[k - 1][i] = Math.max(pWords.length - k + 1, 0);

                int clipped = 0;
                for (Map.Entry<String, Integer> e : candNgram.entrySet()) {
                    int c = e.getValue();
                    int rc = refNgram.getOrDefault(e.getKey(), 0);
                    clipped += Math.min(c, rc);
                }
                clippedCounts[k - 1][i] = clipped;
            }
        }

        double precisionSum = 0;
        for (int k = 0; k < 4; k++) {
            int totalClipped = 0;
            int totalCand = 0;
            for (int i = 0; i < n; i++) {
                totalClipped += clippedCounts[k][i];
                totalCand += candCounts[k][i];
            }
            double p = totalCand > 0 ? (double) totalClipped / totalCand : 0;
            precisionSum += (k == 0 ? 1.0 : 0.25) * Math.log(p + 1e-15);
        }

        double bp = totalCandLen < totalRefLen
                ? Math.exp(1.0 - (double) totalRefLen / Math.max(totalCandLen, 1))
                : 1.0;

        double sacrebleu = bp * Math.exp(precisionSum / 4.0);

        return Map.of("sacrebleu", sacrebleu);
    }

    private static Map<String, Integer> ngrams(String[] words, int n) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i <= words.length - n; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < n; j++) {
                if (j > 0) sb.append(' ');
                sb.append(words[i + j]);
            }
            map.merge(sb.toString(), 1, Integer::sum);
        }
        return map;
    }

    @Override
    public String description() {
        return "SacreBLEU: standardized, tokenization-independent BLEU score.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' keys containing text strings.";
    }
}
