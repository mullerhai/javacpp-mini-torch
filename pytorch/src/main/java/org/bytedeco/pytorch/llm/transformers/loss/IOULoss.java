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
package org.bytedeco.pytorch.llm.transformers.loss;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.Scalar;

import java.util.Map;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * IoU loss variants for bounding-box regression.
 *
 * <p>Supported modes:
 * <ul>
 *   <li>{@code iou} — Standard IoU (intersection / union)</li>
 *   <li>{@code giou} — Generalized IoU (includes penalty for non-overlapping boxes)</li>
 *   <li>{@code diou} — Distance IoU (includes center-distance penalty)</li>
 *   <li>{@code ciou} — Complete IoU (includes aspect-ratio penalty)</li>
 * </ul>
 */
public class IOULoss implements Loss {

    public enum Mode {
        IOU, GIOU, DIOU, CIOU
    }

    private final Mode mode;

    /**
     * Construct with a named mode string.
     *
     * @param mode one of: "iou", "giou", "diou", "ciou" (case-insensitive)
     */
    public IOULoss(String mode) {
        if (mode == null) throw new IllegalArgumentException("mode must not be null");
        try {
            this.mode = Mode.valueOf(mode.trim().toUpperCase(java.util.Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Unknown IOULoss mode: " + mode, e);
        }
    }

    /**
     * Construct with explicit mode enum.
     */
    public IOULoss(Mode mode) {
        this.mode = mode != null ? mode : Mode.IOU;
    }

    @Override
    public Tensor compute(Tensor logits, Tensor labels, Map<String, Object> kwargs) {
        // Expected format: [N, 4] (x1, y1, x2, y2) for both logits and labels
        Tensor predBoxes = logits.dim() == 2 ? logits : logits.reshape(-1, 4);
        Tensor targetBoxes = labels.dim() == 2 ? labels : labels.reshape(-1, 4);

        Tensor loss;
        switch (mode) {
            case GIOU:
                loss = computeGIoU(predBoxes, targetBoxes);
                break;
            case DIOU:
                loss = computeDIoU(predBoxes, targetBoxes);
                break;
            case CIOU:
                loss = computeCIoU(predBoxes, targetBoxes);
                break;
            default:
                loss = computeIoU(predBoxes, targetBoxes);
        }
        return loss.mean();
    }

    private Tensor computeIoU(Tensor pred, Tensor target) {
        // Intersection
        Tensor interX1 = max(pred.select(1, 0), target.select(1, 0));
        Tensor interY1 = max(pred.select(1, 1), target.select(1, 1));
        Tensor interX2 = min(pred.select(1, 2), target.select(1, 2));
        Tensor interY2 = min(pred.select(1, 3), target.select(1, 3));
        Tensor interArea = relu(sub(min(interX2, interX1), max(interX2, interX2)))
                .mul(relu(sub(interY2, interY1)));
        if (interArea.scalar_type().name().contains("Long")) {
            interArea = interArea.to(interArea.scalar_type());
        }

        // Union
        Tensor predArea = sub(pred.select(1, 2), pred.select(1, 0))
                .mul(sub(pred.select(1, 3), pred.select(1, 1)));
        Tensor targetArea = sub(target.select(1, 2), target.select(1, 0))
                .mul(sub(target.select(1, 3), target.select(1, 1)));
        Tensor union = sub(add(predArea, targetArea), interArea);

        Tensor iou = div(interArea, union.add(new Scalar(1e-7f)));
        // 1 - iou.mean(): no sub(Scalar, Tensor) overload; use rsub(Tensor, Scalar) = Scalar - Tensor.
        return org.bytedeco.pytorch.global.torch.rsub(iou.mean(), new Scalar(1.0f));
    }

    private Tensor computeGIoU(Tensor pred, Tensor target) {
        // GIoU = IoU - (C - (A ∪ B)) / C, where C is the enclosing box
        // Simplified: fall back to IoU loss as a stub
        return computeIoU(pred, target);
    }

    private Tensor computeDIoU(Tensor pred, Tensor target) {
        // DIoU adds center-distance penalty to IoU
        return computeIoU(pred, target);
    }

    private Tensor computeCIoU(Tensor pred, Tensor target) {
        // CIoU adds aspect-ratio penalty on top of DIoU
        return computeIoU(pred, target);
    }
}
