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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * YAML patch operations (RFC 6902 style).
 */
public final class YamlPatch {

    private final List<Operation> operations = new ArrayList<>();

    public YamlPatch() {}

    YamlPatch(List<Operation> operations) {
        this.operations.addAll(operations);
    }

    public void add(Operation op) {
        operations.add(op);
    }

    public List<Operation> operations() {
        return new ArrayList<>(operations);
    }

    public boolean isEmpty() { return operations.isEmpty(); }
    public int size() { return operations.size(); }

    public String toYaml() {
        List<Map<String, Object>> ops = new ArrayList<>();
        for (Operation op : operations) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("op", op.op);
            m.put("path", op.path);
            if (op.value != null) m.put("value", op.value);
            ops.add(m);
        }
        return Yaml.dump(ops);
    }

    public static class Operation {
        public final String op;
        public final String path;
        public final Object value;

        public Operation(String op, String path, Object value) {
            this.op = op;
            this.path = path;
            this.value = value;
        }

        public Operation(String op, String path) {
            this(op, path, null);
        }

        @Override
        public String toString() {
            return "Operation{op=" + op + ", path=" + path + ", value=" + value + "}";
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof Operation)) return false;
            Operation that = (Operation) o;
            return Objects.equals(op, that.op) && Objects.equals(path, that.path) && Objects.equals(value, that.value);
        }

        @Override
        public int hashCode() {
            return Objects.hash(op, path, value);
        }
    }

    @Override
    public String toString() {
        return "YamlPatch{operations=" + operations + "}";
    }
}
