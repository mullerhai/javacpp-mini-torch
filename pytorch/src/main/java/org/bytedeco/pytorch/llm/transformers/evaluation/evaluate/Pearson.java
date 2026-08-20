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
 * Pearson correlation coefficient.
 *
 * <p>Expected keys: {@code "pred"} and {@code "ref"} with numeric values.
 */
public final class Pearson implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        if (predictions == null || references == null || predictions.isEmpty()) {
            return Map.of("pearson", 0.0);
        }

        int n = predictions.size();
        double[] pred = new double[n];
        double[] ref = new double[n];

        for (int i = 0; i < n; i++) {
            pred[i] = ((Number) predictions.get(i).get("pred")).doubleValue();
            ref[i] = ((Number) references.get(i).get("ref")).doubleValue();
        }

        double meanPred = mean(pred);
        double meanRef = mean(ref);
        double cov = 0.0;
        double varPred = 0.0;
        double varRef = 0.0;

        for (int i = 0; i < n; i++) {
            double dp = pred[i] - meanPred;
            double dr = ref[i] - meanRef;
            cov += dp * dr;
            varPred += dp * dp;
            varRef += dr * dr;
        }

        double denominator = Math.sqrt(varPred * varRef);
        double pearson = denominator > 0 ? cov / denominator : 0.0;

        return Map.of("pearson", pearson);
    }

    private static double mean(double[] arr) {
        double sum = 0.0;
        for (double v : arr) sum += v;
        return sum / arr.length;
    }

    @Override
    public String description() {
        return "Pearson correlation coefficient between predictions and references.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' keys containing numeric values.";
    }
}
