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

import java.util.Map;

/**
 * Abstract sequence feature extractor for 1-D sequence models.
 *
 * <p>Reference: HuggingFace transformers
 * {@code feature_extraction_utils.SequenceFeatureExtractor}.
 */
public abstract class SequenceFeatureExtractor extends FeatureExtractor {

    /** Whether to truncate sequences exceeding max_length. */
    protected boolean truncation;

    /** Padding strategy: "max_length", "longest", or "do_not_pad". */
    protected String padding;

    protected SequenceFeatureExtractor(int samplingRate, boolean truncation, String padding) {
        super(samplingRate);
        this.truncation = truncation;
        this.padding = padding != null ? padding : "longest";
    }

    public boolean truncation() { return truncation; }
    public String padding() { return padding; }
}
