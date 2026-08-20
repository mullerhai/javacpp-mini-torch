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
package org.bytedeco.pytorch.llm.transformers.tokenization_utils;

/**
 * Truncation strategy for sequence truncation.
 *
 * <p>Reference: HuggingFace transformers
 * {@code tokenization_utils_base.TruncationStrategy}.
 */
public enum TruncationStrategy {

    /** Truncate longest sequence first (PyTorch default). */
    LONGEST_FIRST(0),

    /** Only truncate the first sequence in a pair. */
    ONLY_FIRST(1),

    /** Only truncate the second sequence in a pair. */
    ONLY_SECOND(2),

    /** Do not truncate. */
    DO_NOT_TRUNCATE(3);

    private final int value;

    TruncationStrategy(int value) {
        this.value = value;
    }

    public int value() {
        return value;
    }

    public static TruncationStrategy from(int v) {
        for (TruncationStrategy s : values()) {
            if (s.value == v) return s;
        }
        return DO_NOT_TRUNCATE;
    }
}
