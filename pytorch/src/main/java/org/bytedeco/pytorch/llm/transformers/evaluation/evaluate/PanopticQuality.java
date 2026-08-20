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
 * Panoptic Quality (PQ) for panoptic segmentation.
 *
 * <p>PQ = (sum IoU per matched segment) / (TP + 0.5*FP + 0.5*FN).
 * Expected keys: {@code "pred_segments"} (List<Map<String,Object>>) and
 * {@code "ref_segments"} (List<Map<String,Object>>) with "id" and "category_id".
 */
public final class PanopticQuality implements Metric {

    @Override
    public Map<String, Double> compute(List<Map<String, Object>> predictions, List<Map<String, Object>> references) {
        // Build lookup maps by segment id
        Map<Integer, Map<String, Object>> predSeg = new HashMap<>();
        Map<Integer, Map<String, Object>> refSeg = new HashMap<>();

        Object pObj = predictions.isEmpty() ? null : predictions.get(0).get("pred_segments");
        Object rObj = references.isEmpty() ? null : references.get(0).get("ref_segments");

        if (pObj instanceof List<?> pList) {
            for (Object item : pList) {
                if (item instanceof Map<?, ?> m) {
                    Object id = m.get("id");
                    if (id != null) predSeg.put(((Number) id).intValue(), castMap(m));
                }
            }
        }

        if (rObj instanceof List<?> rList) {
            for (Object item : rList) {
                if (item instanceof Map<?, ?> m) {
                    Object id = m.get("id");
                    if (id != null) refSeg.put(((Number) id).intValue(), castMap(m));
                }
            }
        }

        double sumIou = 0;
        int tp = 0, fp = 0, fn = 0;

        for (Map.Entry<Integer, Map<String, Object>> e : predSeg.entrySet()) {
            int id = e.getKey();
            Map<String, Object> pSeg = e.getValue();
            Integer catId = pSeg.get("category_id") instanceof Number n ? n.intValue() : null;

            if (refSeg.containsKey(id)) {
                Map<String, Object> rSeg = refSeg.get(id);
                Integer refCatId = rSeg.get("category_id") instanceof Number n ? n.intValue() : null;
                if (catId != null && catId.equals(refCatId)) {
                    double iou = rSeg.get("iou") instanceof Number i ? i.doubleValue() : 1.0;
                    sumIou += iou;
                    tp++;
                } else {
                    fp++;
                }
            } else {
                fp++;
            }
        }

        for (Integer id : refSeg.keySet()) {
            if (!predSeg.containsKey(id)) fn++;
        }

        double pq = (tp + 0.5 * fp + 0.5 * fn) > 0
                ? sumIou / (tp + 0.5 * fp + 0.5 * fn)
                : 0.0;

        return Map.of("panoptic_quality", pq);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> castMap(Map<?, ?> m) {
        Map<String, Object> result = new HashMap<>();
        for (Map.Entry<?, ?> e : m.entrySet()) {
            result.put(String.valueOf(e.getKey()), e.getValue());
        }
        return result;
    }

    @Override
    public String description() {
        return "Panoptic Quality: segmentation quality metric combining recognition and segmentation.";
    }

    @Override
    public String inputsDescription() {
        return "Maps with 'pred_segments' and 'ref_segments' containing segment lists.";
    }
}
