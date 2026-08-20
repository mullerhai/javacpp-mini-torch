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
 * Perplexity metric.
 *
 * <p>Computes token-level perplexity from log-probabilities.
 * Expected keys: {@code "log_probs"} (List<Double>) in predictions and
 * {@code "count"} (int token count) in references.
 */
public final class Perplexity implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        if (predictions == null || predictions.isEmpty()) {
            return Map.of("perplexity", 0.0);
        }

        double totalLogProb = 0.0;
        int totalTokens = 0;

        for (int i = 0; i < predictions.size(); i++) {
            Object lpObj = predictions.get(i).get("log_probs");
            Object cntObj = references.get(i).get("count");

            if (lpObj instanceof List<?> lpList) {
                for (Object v : lpList) {
                    if (v instanceof Number n) {
                        totalLogProb += n.doubleValue();
                    }
                }
            }
            if (cntObj instanceof Number cnt) {
                totalTokens += cnt.intValue();
            }
        }

        double avgLogProb = totalTokens > 0 ? totalLogProb / totalTokens : 0.0;
        double ppl = Math.exp(-avgLogProb);

        return Map.of("perplexity", ppl);
    }

    @Override
    public String description() {
        return "Perplexity: exponent of the average negative log-likelihood.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'log_probs' (List<Double>) and 'count' (int) keys.";
    }
}
