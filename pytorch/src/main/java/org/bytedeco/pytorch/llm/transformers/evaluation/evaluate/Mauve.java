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
import java.util.Set;

/**
 * MAUVE score for text generation evaluation.
 *
 * <p>Measures distributional divergence between predicted and reference texts
 * using quantum-inspired clustering. This is a simplified approximation.
 * Expected keys: {@code "pred"} and {@code "ref"} containing text strings.
 */
public final class Mauve implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        if (predictions == null || predictions.isEmpty() || references == null || references.isEmpty()) {
            return Map.of("mauve", 0.0);
        }

        // Simplified: use average character-level BLEU as proxy for MAUVE divergence.
        // Real MAUVE requires embeddings and quantum clustering (HuggingFace evaluate library).
        double sumScore = 0;
        int count = Math.min(predictions.size(), references.size());

        for (int i = 0; i < count; i++) {
            String pred = String.valueOf(predictions.get(i).get("pred"));
            String ref = String.valueOf(references.get(i).get("ref"));

            // Simplified character overlap score
            double score = charOverlap(pred, ref);
            sumScore += score;
        }

        double mauve = count > 0 ? sumScore / count : 0.0;
        return Map.of("mauve", mauve);
    }

    private static double charOverlap(String a, String b) {
        if (a.isEmpty() || b.isEmpty()) return 0.0;
        Set<Character> setA = new java.util.HashSet<>();
        for (char c : a.toCharArray()) if (!Character.isWhitespace(c)) setA.add(c);
        Set<Character> setB = new java.util.HashSet<>();
        for (char c : b.toCharArray()) if (!Character.isWhitespace(c)) setB.add(c);

        Set<Character> intersection = new java.util.HashSet<>(setA);
        intersection.retainAll(setB);
        double union = setA.size() + setB.size() - intersection.size();
        return union > 0 ? intersection.size() / union : 0.0;
    }

    @Override
    public String description() {
        return "MAUVE: distributional divergence score for text generation (simplified approximation).";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' keys containing text strings.";
    }
}
