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
 * ROUGE score (Recall-Oriented Understudy for Gisting Evaluation).
 *
 * <p>Returns ROUGE-1, ROUGE-2, and ROUGE-L scores.
 * Expected keys: {@code "pred"} and {@code "ref"} containing text strings.
 */
public final class Rouge implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        double sumR1 = 0, sumR2 = 0, sumRL = 0;

        for (int i = 0; i < predictions.size(); i++) {
            String pred = String.valueOf(predictions.get(i).get("pred"));
            String ref = String.valueOf(references.get(i).get("ref"));

            Set<String> unigramsPred = unigrams(pred);
            Set<String> unigramsRef = unigrams(ref);
            Set<String> bigramsPred = bigrams(pred);
            Set<String> bigramsRef = bigrams(ref);

            sumR1 += recall(unigramsPred, unigramsRef);
            sumR2 += recall(bigramsPred, bigramsRef);
            sumRL += lcsRecall(pred, ref);
        }

        int n = Math.max(predictions.size(), 1);
        Map<String, Double> result = new HashMap<>();
        result.put("rouge1", sumR1 / n);
        result.put("rouge2", sumR2 / n);
        result.put("rougeL", sumRL / n);
        return result;
    }

    private static Set<String> unigrams(String text) {
        Set<String> set = new HashSet<>();
        for (String w : text.toLowerCase().split("\\s+")) {
            if (!w.isBlank()) set.add(w);
        }
        return set;
    }

    private static Set<String> bigrams(String text) {
        Set<String> set = new HashSet<>();
        String[] words = text.toLowerCase().split("\\s+");
        for (int i = 0; i < words.length - 1; i++) {
            if (!words[i].isBlank() && !words[i + 1].isBlank()) {
                set.add(words[i] + " " + words[i + 1]);
            }
        }
        return set;
    }

    private static double recall(Set<String> cand, Set<String> ref) {
        if (cand.isEmpty()) return 0.0;
        int overlap = 0;
        for (String w : cand) {
            if (ref.contains(w)) overlap++;
        }
        return (double) overlap / ref.size();
    }

    private static double lcsRecall(String pred, String ref) {
        String[] pWords = pred.toLowerCase().split("\\s+");
        String[] rWords = ref.toLowerCase().split("\\s+");
        int m = pWords.length;
        int n = rWords.length;
        if (m == 0 || n == 0) return 0.0;

        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (pWords[i - 1].equals(rWords[j - 1])) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        int lcsLen = dp[m][n];
        return (double) lcsLen / n;
    }

    @Override
    public String description() {
        return "ROUGE: n-gram and longest-common-subsequence overlap between predictions and references.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' keys containing text strings.";
    }
}
