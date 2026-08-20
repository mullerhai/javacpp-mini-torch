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
 * Time series forecasting metrics.
 *
 * <p>Computes MSE, MAE, and MAPE over time series predictions.
 * Expected keys: {@code "pred"} (List<Double> or double[]) and
 * {@code "ref"} (List<Double> or double[]).
 */
public final class TimeSeriesMetric implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        double sumSqErr = 0, sumAbsErr = 0, sumAbsPctErr = 0;
        int count = 0;

        for (int i = 0; i < predictions.size(); i++) {
            double[] pred = toDoubleArray(predictions.get(i).get("pred"));
            double[] ref = toDoubleArray(references.get(i).get("ref"));

            int minLen = Math.min(pred.length, ref.length);
            for (int j = 0; j < minLen; j++) {
                double e = pred[j] - ref[j];
                sumSqErr += e * e;
                sumAbsErr += Math.abs(e);
                if (ref[j] != 0) {
                    sumAbsPctErr += Math.abs(e / ref[j]);
                }
                count++;
            }
        }

        int n = Math.max(count, 1);
        Map<String, Double> result = new HashMap<>();
        result.put("mse", sumSqErr / n);
        result.put("mae", sumAbsErr / n);
        result.put("mape", sumAbsPctErr / n);
        return result;
    }

    private static double[] toDoubleArray(Object obj) {
        if (obj instanceof double[] arr) return arr;
        if (obj instanceof List<?> list) {
            double[] arr = new double[list.size()];
            for (int i = 0; i < list.size(); i++) {
                arr[i] = ((Number) list.get(i)).doubleValue();
            }
            return arr;
        }
        return new double[0];
    }

    @Override
    public String description() {
        return "Time series metrics: MSE, MAE, and MAPE.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' containing double[] or List<Double>.";
    }
}
