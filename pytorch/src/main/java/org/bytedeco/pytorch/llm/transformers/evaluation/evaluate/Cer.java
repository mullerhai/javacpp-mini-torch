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
 * Character Error Rate (CER).
 *
 * <p>Edit distance at the character level: (S + D + I) / N.
 * Expected keys: {@code "pred"} and {@code "ref"} containing text strings.
 */
public final class Cer implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        int totalErrors = 0;
        int totalChars = 0;

        for (int i = 0; i < predictions.size(); i++) {
            String pred = String.valueOf(predictions.get(i).get("pred"));
            String ref = String.valueOf(references.get(i).get("ref"));

            char[] p = pred.toCharArray();
            char[] r = ref.toCharArray();
            int m = p.length, n = r.length;
            int[][] dp = new int[m + 1][n + 1];

            for (int a = 0; a <= m; a++) dp[a][0] = a;
            for (int b = 0; b <= n; b++) dp[0][b] = b;

            for (int a = 1; a <= m; a++) {
                for (int b = 1; b <= n; b++) {
                    if (p[a - 1] == r[b - 1]) {
                        dp[a][b] = dp[a - 1][b - 1];
                    } else {
                        dp[a][b] = 1 + Math.min(dp[a - 1][b], Math.min(dp[a][b - 1], dp[a - 1][b - 1]));
                    }
                }
            }

            totalErrors += dp[m][n];
            totalChars += n;
        }

        double cer = totalChars > 0 ? (double) totalErrors / totalChars : 0.0;
        return Map.of("cer", cer);
    }

    @Override
    public String description() {
        return "Character Error Rate: fraction of characters incorrectly predicted.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' keys containing text strings.";
    }
}
