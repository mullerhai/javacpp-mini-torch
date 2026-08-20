/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "Classpath" exception),
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
package org.bytedeco.pytorch.llm.transformers.processing_utils;

import java.util.Map;

/**
 * Container for text and image processing keyword arguments.
 *
 * <p>Reference: HuggingFace transformers
 * {@code processing_utils.ProcessingKwargs}.
 */
public class ProcessingKwargs {

    /** Keyword arguments for text processing. */
    protected Map<String, Object> text_kwargs;

    /** Keyword arguments for image processing. */
    protected Map<String, Object> images_kwargs;

    public ProcessingKwargs() {}

    public ProcessingKwargs(Map<String, Object> text_kwargs, Map<String, Object> images_kwargs) {
        this.text_kwargs = text_kwargs;
        this.images_kwargs = images_kwargs;
    }

    public Map<String, Object> text_kwargs() { return text_kwargs; }
    public Map<String, Object> images_kwargs() { return images_kwargs; }

    public ProcessingKwargs text_kwargs(Map<String, Object> text_kwargs) {
        this.text_kwargs = text_kwargs;
        return this;
    }

    public ProcessingKwargs images_kwargs(Map<String, Object> images_kwargs) {
        this.images_kwargs = images_kwargs;
        return this;
    }
}
