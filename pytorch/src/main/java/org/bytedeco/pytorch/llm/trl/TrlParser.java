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
package org.bytedeco.pytorch.llm.trl;

import org.bytedeco.pytorch.llm.trl.config.SFTConfig;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * HuggingFace TRL {@code TrlParser} — Java stand-in.
 *
 * <p>Python uses {@code HfArgumentParser} over dataclasses. Java has no
 * dataclass CLI; this parser fills {@link ScriptArguments}, {@link SFTConfig}
 * and {@link ModelConfig} from a {@code --key value} argv or a map.
 * Tutorials typically call the builders directly.
 */
public final class TrlParser {

    public record Parsed(ScriptArguments script, SFTConfig training, ModelConfig model) {}

    private TrlParser() {}

    public static Parsed parse(String[] args) {
        Map<String, String> m = new LinkedHashMap<>();
        if (args != null) {
            for (int i = 0; i < args.length; i++) {
                String a = args[i];
                if (a != null && a.startsWith("--")) {
                    String key = a.substring(2).replace('-', '_');
                    String val = (i + 1 < args.length && !args[i + 1].startsWith("--")) ? args[++i] : "true";
                    m.put(key, val);
                }
            }
        }
        return parse(m);
    }

    public static Parsed parse(Map<String, String> m) {
        ScriptArguments.Builder sb = ScriptArguments.builder();
        ModelConfig.Builder mb = ModelConfig.builder();
        SFTConfig.Builder tb = SFTConfig.builder();
        if (m != null) {
            put(m, "dataset_name", sb::dataset_name);
            put(m, "dataset_config", sb::dataset_config);
            put(m, "dataset_train_split", sb::dataset_train_split);
            put(m, "dataset_test_split", sb::dataset_test_split);
            put(m, "model_name_or_path", mb::model_name_or_path);
            put(m, "model_revision", mb::model_revision);
            put(m, "torch_dtype", mb::torch_dtype);
            put(m, "output_dir", tb::output_dir);
        }
        return new Parsed(sb.build(), tb.build(), mb.build());
    }

    private static void put(Map<String, String> m, String key, java.util.function.Consumer<String> c) {
        if (m.containsKey(key)) c.accept(m.get(key));
    }
}
