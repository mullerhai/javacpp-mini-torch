/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "Classpath" exception),
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
package org.bytedeco.pytorch.llm.transformers.modeling_utils;

import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.StringVector;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.transformers.loading.WeightLoader;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Base class for pretrained models.
 *
 * <p>Reference: HuggingFace transformers
 * {@code modeling_utils.PreTrainedModel}.
 */
public abstract class PreTrainedModel extends Module {

    protected PreTrainedModel() {}

    /**
     * Load a pretrained model from HuggingFace Hub or a local directory.
     *
     * @param modelId model identifier (HF repo ID or local path)
     * @param hub     HuggingFace hub client
     * @return loaded model module
     * @throws IOException on load failure
     */
    public static Module fromPretrained(String modelId, HfHub hub) throws IOException {
        Path snapshot = hub.snapshotDownload(modelId);
        Module model = null; // Subclasses create the Module instance
        if (model != null) {
            WeightLoader.LoadReport report = WeightLoader.loadAndBind(model, snapshot);
            System.out.println("Loaded " + modelId + " in " + snapshot + ": " + report);
        }
        return model;
    }

    /**
     * Get all named parameters as a map.
     *
     * <p>This is a Java-friendly convenience around {@link Module#named_parameters()}
     * which returns a native dictionary type. The map name is intentionally
     * pluralised {@code namedParameters} (camelCase) to avoid clashing with
     * the native method.
     *
     * @return unmodifiable map of parameter name to tensor
     */
    public Map<String, org.bytedeco.pytorch.Tensor> namedParameters() {
        StringTensorDict dict = named_parameters(/*recurse=*/true);
        if (dict == null || dict.is_empty()) return Collections.emptyMap();
        Map<String, org.bytedeco.pytorch.Tensor> out = new LinkedHashMap<>();
        StringVector keys = dict.keys();
        TensorVector values = dict.values();
        long n = Math.min(keys.size(), values.size());
        for (long i = 0; i < n; i++) {
            out.put(keys.get(i).getString(), values.get(i));
        }
        return Collections.unmodifiableMap(out);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }
}
