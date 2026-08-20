/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed undered the Apache License, Version 2.0, or (at your option)
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
 * Java analog of HuggingFace {@code peft.utils.warning.PeftWarning(UserWarning)}.
 *
 * <p>Extends {@link RuntimeException} so it can be caught and inspected by callers, but its
 * presence in business code is non-fatal. Code that needs to silence a warning should
 * wrap the offending block in {@code try/catch (PeftWarning warn) { ... }}.
 */
public class PeftWarning extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public PeftWarning(String message) {
        super(message);
    }

    public PeftWarning(String message, Throwable cause) {
        super(message, cause);
    }

    public PeftWarning(Throwable cause) {
        super(cause);
    }
}