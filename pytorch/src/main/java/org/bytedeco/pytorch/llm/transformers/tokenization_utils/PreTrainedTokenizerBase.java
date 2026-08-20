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
package org.bytedeco.pytorch.llm.transformers.tokenization_utils;

import java.util.List;

/**
 * Base class for pretrained tokenizers.
 *
 * <p>Defines common padding/truncation configuration.
 * Reference: HuggingFace transformers
 * {@code tokenization_utils_base.PreTrainedTokenizerBase}.
 */
public abstract class PreTrainedTokenizerBase {

    /** Padding side: "left" or "right". */
    protected String padding_side = "right";

    /** Truncation side: "left" or "right". */
    protected String truncation_side = "right";

    /** Maximum sequence length accepted by this tokenizer. */
    protected long model_max_length = 512;

    protected PreTrainedTokenizerBase() {}

    public String padding_side() { return padding_side; }
    public String truncation_side() { return truncation_side; }
    public long model_max_length() { return model_max_length; }

    protected void padding_side(String v) { this.padding_side = v; }
    protected void truncation_side(String v) { this.truncation_side = v; }
    protected void model_max_length(long v) { this.model_max_length = v; }
}
