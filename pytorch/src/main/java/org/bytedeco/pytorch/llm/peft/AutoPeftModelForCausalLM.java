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

import org.bytedeco.pytorch.llm.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.nn.Module;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * HuggingFace PEFT {@code AutoPeftModelForCausalLM.from_pretrained(adapter_path)}.
 *
 * <p>Reads {@code adapter_config.json} for {@code base_model_name_or_path},
 * loads the base causal LM, then attaches the adapter via
 * {@link PeftModel#fromPretrained(Module, File)}.
 */
public final class AutoPeftModelForCausalLM {

    private AutoPeftModelForCausalLM() {}

    public static PeftModel fromPretrained(String adapterPath) throws IOException {
        return from_pretrained(adapterPath);
    }

    public static PeftModel from_pretrained(String adapterPath) throws IOException {
        Objects.requireNonNull(adapterPath, "adapterPath");
        Path dir = Path.of(adapterPath);
        String base = readBaseModelName(dir);
        if (base == null || base.isBlank()) {
            throw new IOException("adapter_config.json missing base_model_name_or_path in " + dir);
        }
        AutoModelForCausalLM.Bundle bundle;
        Path local = Path.of(base);
        if (Files.isDirectory(local)) {
            bundle = AutoModelForCausalLM.fromLocal(local);
        } else {
            bundle = AutoModelForCausalLM.fromPretrainedDefault(base);
        }
        return PeftModel.fromPretrained(bundle.model(), dir.toFile());
    }

    public static PeftModel fromPretrained(Module baseModel, String adapterPath) throws IOException {
        return PeftModel.fromPretrained(baseModel, adapterPath);
    }

    static String readBaseModelName(Path dir) throws IOException {
        Path cfg = dir.resolve("adapter_config.json");
        if (!Files.isRegularFile(cfg)) return null;
        String json = Files.readString(cfg, StandardCharsets.UTF_8);
        String key = "\"base_model_name_or_path\"";
        int i = json.indexOf(key);
        if (i < 0) return null;
        int colon = json.indexOf(':', i + key.length());
        if (colon < 0) return null;
        int q1 = json.indexOf('"', colon + 1);
        if (q1 < 0) return null;
        int q2 = json.indexOf('"', q1 + 1);
        if (q2 < 0) return null;
        return json.substring(q1 + 1, q2);
    }
}
