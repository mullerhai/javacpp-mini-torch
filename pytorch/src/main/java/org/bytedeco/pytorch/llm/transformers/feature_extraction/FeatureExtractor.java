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
package org.bytedeco.pytorch.llm.transformers.feature_extraction;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.Map;

/**
 * Abstract base class for feature extractors.
 *
 * <p>Reference: HuggingFace transformers
 * {@code feature_extraction_utils.FeatureExtractor}.
 */
public abstract class FeatureExtractor implements FeatureExtractionMixin {

    /** Target sampling rate for audio feature extraction. */
    protected int sampling_rate;

    protected FeatureExtractor(int samplingRate) {
        this.sampling_rate = samplingRate;
    }

    /**
     * Pad a 1-D or 2-D tensor to {@code maxLen}.
     *
     * @param t       input tensor
     * @param maxLen  target length
     * @return padded tensor (may be the same tensor if already at maxLen)
     */
    protected Tensor pad(Tensor t, int maxLen) {
        if (t == null) return null;
        long[] shape = t.sizes().vec().get();
        long currentLen = shape.length > 1 ? shape[0] : shape[shape.length - 1];
        if (currentLen >= maxLen) return t;

        // For now, return the tensor as-is; subclasses override with real padding logic.
        // Real implementation would allocate a new tensor and copy data.
        return t;
    }

    @Override
    public Tensor extract_features(Tensor input, Map<String, Object> kwargs) {
        throw new UnsupportedOperationException("Subclasses must implement extract_features");
    }

    public int sampling_rate() {
        return sampling_rate;
    }
}
