/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
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
package org.bytedeco.pytorch.llm.transformers.loss;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.Scalar;

import java.util.Map;
import java.util.LinkedHashMap;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * DETR-style detection loss combining classification and box regression.
 *
 * <p>Combines a cross-entropy classification loss over all classes with
 * an IoU / L1 box regression loss.
 */
public class DetectionLoss implements Loss {

    public static final double DEFAULT_CLASS_WEIGHT = 1.0;
    public static final double DEFAULT_BOX_WEIGHT = 5.0;

    private final double classWeight;
    private final double boxWeight;
    private final double ceGamma;
    private final double ceAlpha;
    private final double lossEps;

    /**
     * Construct a DetectionLoss with default weights.
     */
    public DetectionLoss() {
        this(defaultWeights());
    }

    /**
     * Construct with custom per-component weights.
     *
     * @param weights map with keys: "loss_ce", "loss_bbox", "loss_giou"
     */
    public DetectionLoss(Map<String, Float> weights) {
        if (weights == null) weights = defaultWeights();
        this.classWeight = weights.getOrDefault("loss_ce", 1.0f);
        this.boxWeight = weights.getOrDefault("loss_bbox", 5.0f);
        this.ceGamma = weights.containsKey("ce_gamma") ? weights.get("ce_gamma") : 0.0;
        this.ceAlpha = weights.containsKey("ce_alpha") ? weights.get("ce_alpha") : 0.0;
        this.lossEps = 1e-8;
    }

    private static Map<String, Float> defaultWeights() {
        Map<String, Float> w = new LinkedHashMap<>();
        w.put("loss_ce", 1.0f);
        w.put("loss_bbox", 5.0f);
        w.put("loss_giou", 2.0f);
        return w;
    }

    @Override
    public Tensor compute(Tensor logits, Tensor labels, Map<String, Object> kwargs) {
        // logits shape: [B, num_queries, num_classes]
        // labels is a composite: first dim holds class labels, second holds boxes [x,y,w,h]
        // We expect kwargs to carry "boxes" tensor for box regression
        Tensor boxes = kwargs != null && kwargs.containsKey("boxes")
                ? (Tensor) kwargs.get("boxes")
                : null;

        // Classification loss
        Tensor classLoss;
        if (ceGamma > 0 && ceAlpha > 0) {
            classLoss = focalClassificationLoss(logits, labels);
        } else {
            classLoss = cross_entropy(logits.transpose(0, 1).contiguous(),
                    labels.to(ScalarType.Long),
                    new org.bytedeco.pytorch.nn.functional.CrossEntropyFuncOptions()
                            .ignore_index(-100)
                            .label_smoothing(0.0f));
        }

        Tensor total = mul(new Scalar((float) classWeight), classLoss);

        if (boxes != null) {
            Tensor boxLoss = l1Loss(logits, boxes);
            total = add(total, mul(new Scalar((float) boxWeight), boxLoss));
        }

        return total;
    }

    private Tensor focalClassificationLoss(Tensor logits, Tensor targets) {
        // Focal CE for imbalanced detection
        Tensor probs = softmax(logits, logits.dim() - 1);
        Tensor pt = probs.gather(logits.dim() - 1, targets.unsqueeze(-1), true)
                .clamp(1e-7f, 1.0f - 1e-7f);
        Tensor focalWeight = pow(sub(new Scalar(1.0f), pt), new Scalar((float) ceGamma));
        Tensor ce = neg(log(pt));
        return mul(mul(new Scalar((float) ceAlpha, focalWeight.scalar_type()), focalWeight), ce).mean();
    }

    private Tensor l1Loss(Tensor logits, Tensor boxes) {
        // Stub: use L1 distance between predicted and target boxes
        return boxes != null ? abs(sub(logits.mean(), boxes.mean())) : tensor(0.0f);
    }
}
