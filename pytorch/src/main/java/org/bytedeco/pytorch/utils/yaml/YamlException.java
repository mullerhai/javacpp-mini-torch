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

/**
 * Exception thrown for YAML processing errors.
 */
public class YamlException extends RuntimeException {

    private final String path;
    private final int line;
    private final int column;

    public YamlException(String message) {
        super(message);
        this.path = null;
        this.line = -1;
        this.column = -1;
    }

    public YamlException(String message, Throwable cause) {
        super(message, cause);
        this.path = null;
        this.line = -1;
        this.column = -1;
    }

    public YamlException(String message, String path, int line, int column) {
        super(formatMessage(message, path, line, column));
        this.path = path;
        this.line = line;
        this.column = column;
    }

    private static String formatMessage(String message, String path, int line, int column) {
        StringBuilder sb = new StringBuilder(message);
        if (path != null) sb.append(" (path: ").append(path).append(")");
        if (line > 0) sb.append(" at line ").append(line);
        if (column > 0) sb.append(", column ").append(column);
        return sb.toString();
    }

    public String getPath() { return path; }
    public int getLine() { return line; }
    public int getColumn() { return column; }
}
