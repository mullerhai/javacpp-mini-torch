/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.yaml;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Represents a validation error with location information.
 */
public final class ValidationError {
    private final String path;
    private final String message;
    private final String code;

    public ValidationError(String path, String message) {
        this(path, message, null);
    }

    public ValidationError(String path, String message, String code) {
        this.path = path;
        this.message = message;
        this.code = code;
    }

    public String path() { return path; }
    public String message() { return message; }
    public String code() { return code; }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        if (path != null && !path.isEmpty()) {
            sb.append("at ").append(path).append(": ");
        }
        if (code != null) sb.append("[").append(code).append("] ");
        sb.append(message);
        return sb.toString();
    }

    public static String formatAll(List<ValidationError> errors) {
        if (errors == null || errors.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < errors.size(); i++) {
            if (i > 0) sb.append('\n');
            sb.append("- ").append(errors.get(i).toString());
        }
        return sb.toString();
    }
}
