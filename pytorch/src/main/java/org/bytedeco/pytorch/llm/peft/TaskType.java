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
package org.bytedeco.pytorch.llm.peft;

/**
 * HuggingFace PEFT {@code peft.utils.peft_types.TaskType}.
 *
 * <p>The tutorial writes {@code task_type="CAUSAL_LM"} as a string; both the
 * enum and the string form are accepted by {@link PeftConfig.Builder}.
 */
public enum TaskType {
    CAUSAL_LM,
    SEQ_2_SEQ_LM,
    SEQ_CLS,
    TOKEN_CLS,
    QUESTION_ANS,
    FEATURE_EXTRACTION;

    public static TaskType fromString(String raw) {
        if (raw == null || raw.isBlank()) return CAUSAL_LM;
        String n = raw.trim().toUpperCase().replace('-', '_');
        for (TaskType t : values()) {
            if (t.name().equals(n)) return t;
        }
        throw new IllegalArgumentException("Unknown PEFT TaskType: " + raw);
    }
}
