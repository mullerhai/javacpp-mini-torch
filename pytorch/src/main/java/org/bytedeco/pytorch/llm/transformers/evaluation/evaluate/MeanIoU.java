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
 * Mean Intersection over Union (mIoU) for semantic segmentation.
 *
 * <p>Computes per-class IoU and averages across classes.
 * Expected keys: {@code "pred"} (int[][] or flattened int[]) and
 * {@code "ref"} (int[][] or flattened int[]) — label maps of equal shape.
 */
public final class MeanIoU implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        Map<Integer, Integer> intersection = new HashMap<>();
        Map<Integer, Integer> predCount = new HashMap<>();
        Map<Integer, Integer> refCount = new HashMap<>();

        for (int i = 0; i < predictions.size(); i++) {
            Object pObj = predictions.get(i).get("pred");
            Object rObj = references.get(i).get("ref");

            int[] pred = toIntArray(pObj);
            int[] ref = toIntArray(rObj);

            if (pred.length != ref.length) continue;

            for (int j = 0; j < pred.length; j++) {
                int p = pred[j];
                int r = ref[j];
                predCount.merge(p, 1, Integer::sum);
                refCount.merge(r, 1, Integer::sum);
                if (p == r) {
                    intersection.merge(p, 1, Integer::sum);
                }
            }
        }

        // Compute IoU per class
        Map<String, Double> result = new HashMap<>();
        java.util.Set<Integer> allClasses = new java.util.HashSet<>();
        allClasses.addAll(predCount.keySet());
        allClasses.addAll(refCount.keySet());

        double iouSum = 0;
        int count = 0;
        for (int cls : allClasses) {
            int inter = intersection.getOrDefault(cls, 0);
            int predC = predCount.getOrDefault(cls, 0);
            int refC = refCount.getOrDefault(cls, 0);
            int union = predC + refC - inter;
            if (union > 0) {
                double iou = (double) inter / union;
                result.put("iou:cls_" + cls, iou);
                iouSum += iou;
                count++;
            }
        }

        result.put("mean_iou", count > 0 ? iouSum / count : 0.0);
        return result;
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
        return "Mean Intersection over Union for semantic segmentation.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred' and 'ref' keys containing label map arrays.";
    }
}
