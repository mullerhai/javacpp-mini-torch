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
package org.bytedeco.pytorch.llm.transformers.trainer;

import java.util.Map;

/**
 * Simple callback that prints {@code [step N] loss=...} to stdout on each log event.
 */
public final class PrinterCallback implements TrainerCallback {

    @Override
    public TrainerControl onLog(TrainingArguments args, TrainerState state,
                               TrainerControl control, Map<String, Object> logs) {
        if (!state.isLocalProcessZero()) return control;
        StringBuilder sb = new StringBuilder("[step ").append(state.globalStep()).append("]");
        for (var e : logs.entrySet()) {
            sb.append(" ").append(e.getKey()).append("=").append(e.getValue());
        }
        System.out.println(sb);
        return control;
    }
}
