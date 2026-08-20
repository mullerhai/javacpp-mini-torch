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

import org.bytedeco.pytorch.nn.Module;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * Static facade mirroring HuggingFace {@code peft.__init__} top-level re-exports.
 *
 * <p>All entry points forward to the {@code functional} / {@code mapping_func} / {@code utils}
 * implementations in their own packages; this class exists for ergonomic static imports
 * of the form {@code import static org.bytedeco.pytorch.llm.peft.Peft.*;}.
 */
public final class Peft {

    private Peft() {}

    /** Java port of HuggingFace PEFT version (mirrors peft 0.20.0). */
    public static final String VERSION = "0.20.0-java";

    /** Equivalent to {@code peft.get_peft_model(model, peft_config, adapter_name="default")}. */
    public static PeftModel getPeftModel(Module model, PeftConfig peftConfig) {
        return PeftModel.getPeftModel(model, peftConfig);
    }

    public static PeftModel getPeftModel(Module model, PeftConfig peftConfig, String adapterName) {
        return PeftModel.getPeftModel(model, peftConfig, adapterName);
    }

    /** Equivalent to {@code peft.get_peft_config(config_dict)} — returns the right concrete config. */
    public static PeftConfig getPeftConfig(Map<String, Object> configDict) {
        return PeftMethodRegistry.instance().fromDict(configDict);
    }

    /** Equivalent to {@code peft.inject_adapter_in_model(peft_config, model, adapter_name="default")}. */
    public static Module injectAdapterInModel(PeftConfig peftConfig, Module model) {
        return injectAdapterInModel(peftConfig, model, "default");
    }

    public static Module injectAdapterInModel(PeftConfig peftConfig, Module model, String adapterName) {
        try {
            return (Module) Class.forName("org.bytedeco.pytorch.llm.peft.functional.Functional")
                    .getMethod("injectAdapterInModel", PeftConfig.class, Module.class, String.class, boolean.class, Map.class)
                    .invoke(null, peftConfig, model, adapterName, false, null);
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException("injectAdapterInModel failed: " + e.getMessage(), e);
        }
    }

    /** Equivalent to {@code peft.PeftModel.from_pretrained(base_model, adapter_path)}. */
    public static PeftModel fromPretrained(Module baseModel, String adapterPath) throws IOException {
        return PeftModel.fromPretrained(baseModel, new File(adapterPath));
    }

    /** Equivalent to {@code peft.peft_model.PeftModel.save_pretrained(...)} — convenience over {@link PeftModel#savePretrained(File)}. */
    public static void savePretrained(PeftModel model, File dir) throws IOException {
        model.savePretrained(dir);
    }

    /** Equivalent to {@code peft.utils.helpers.check_if_peft_model(path)}. */
    public static boolean checkIfPeftModel(String path) {
        try {
            return (Boolean) Class.forName("org.bytedeco.pytorch.llm.peft.helpers.CheckIfPeftModel")
                    .getMethod("isPeftModel", String.class).invoke(null, path);
        } catch (ReflectiveOperationException e) {
            return false;
        }
    }

    /** Default import allow-list used by AutoPeftModel (mirrors HF {@code get_default_import_allowlist()}). */
    public static List<String> getDefaultImportAllowlist() {
        return java.util.List.of("diffusers", "lerobot", "megatron-core", "transformers");
    }
}