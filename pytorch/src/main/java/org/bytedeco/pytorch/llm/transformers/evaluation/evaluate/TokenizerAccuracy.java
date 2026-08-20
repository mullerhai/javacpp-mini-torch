/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "Classpath" exception),
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
 * Tokenizer accuracy: fraction of correctly tokenized sequences.
 *
 * <p>Compares tokenized outputs against reference tokenizations.
 * Expected keys: {@code "pred_ids"} (int[] or List<Integer>) and
 * {@code "ref_ids"} (int[] or List<Integer>).
 */
public final class TokenizerAccuracy implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        if (predictions == null || references == null || predictions.size() != references.size()) {
            return Map.of("tokenizer_accuracy", 0.0);
        }

        int totalCorrect = 0;
        int totalTokens = 0;

        for (int i = 0; i < predictions.size(); i++) {
            int[] predIds = toIntArray(predictions.get(i).get("pred_ids"));
            int[] refIds = toIntArray(references.get(i).get("ref_ids"));

            int minLen = Math.min(predIds.length, refIds.length);
            int correct = 0;
            for (int j = 0; j < minLen; j++) {
                if (predIds[j] == refIds[j]) correct++;
            }
            totalCorrect += correct;
            totalTokens += refIds.length;
        }

        double acc = totalTokens > 0 ? (double) totalCorrect / totalTokens : 0.0;
        return Map.of("tokenizer_accuracy", acc);
    }

    private static int[] toIntArray(Object obj) {
        if (obj instanceof int[] arr) return arr;
        if (obj instanceof List<?> list) {
            int[] arr = new int[list.size()];
            for (int i = 0; i < list.size(); i++) {
                arr[i] = ((Number) list.get(i)).intValue();
            }
            return arr;
        }
        return new int[0];
    }

    @Override
    public String description() {
        return "Fraction of correctly tokenized token IDs against reference tokenization.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred_ids' and 'ref_ids' containing int[] or List<Integer>.";
    }
}
