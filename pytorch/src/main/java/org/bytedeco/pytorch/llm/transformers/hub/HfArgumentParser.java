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
package org.bytedeco.pytorch.llm.transformers.hub;

import java.util.Arrays;
import java.util.List;

/**
 * Helper to parse arbitrary HF training CLI args (mirrors
 * {@code transformers.HfArgumentParser}). Accepts a list of {@code --key=value}
 * strings and produces a typed map. Mimics the minimum surface needed by
 * tutorials.
 */
public final class HfArgumentParser {

    private HfArgumentParser() {}

    public static java.util.Map<String, String> parse(String[] argv) {
        java.util.LinkedHashMap<String, String> out = new java.util.LinkedHashMap<>();
        for (String a : argv) {
            if (a == null) continue;
            if (!a.startsWith("--")) continue;
            String kv = a.substring(2);
            int eq = kv.indexOf('=');
            if (eq >= 0) {
                out.put(kv.substring(0, eq), kv.substring(eq + 1));
            } else {
                out.put(kv, "true");
            }
        }
        return out;
    }

    public static java.util.Map<String, String> parse(List<String> argv) {
        return parse(argv.toArray(new String[0]));
    }

    public static java.util.Map<String, String> parseCli(String[] argv) {
        return parse(Arrays.copyOfRange(argv, 1, argv.length));
    }
}