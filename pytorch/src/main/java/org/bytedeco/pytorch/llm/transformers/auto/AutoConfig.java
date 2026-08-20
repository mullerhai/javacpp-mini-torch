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
package org.bytedeco.pytorch.llm.transformers.auto;

import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

import java.io.IOException;
import java.nio.file.Path;

/**
 * HF {@code AutoConfig.from_pretrained} entry point.
 *
 * <p>Mirrors the HuggingFace Transformers {@code AutoConfig} API:
 * resolves {@code config.json} from a Hub snapshot or local directory and
 * returns a {@link PretrainedConfig} object.
 */
public final class AutoConfig {

    private AutoConfig() {}

    /**
     * Load a config from a HuggingFace Hub model id.
     *
     * @param modelId HF model id (e.g. {@code "bert-base-uncased"})
     * @param hub      Hub client
     * @return {@link PretrainedConfig}
     */
    public static PretrainedConfig fromPretrained(String modelId, HfHub hub) throws IOException {
        Path snap = hub.snapshotDownload(modelId, "main", "models", java.util.List.of("config.json"));
        return PretrainedConfig.fromDirectory(snap);
    }

    /**
     * Convenience overload that uses {@link HfHub#fromEnv()}.
     */
    public static PretrainedConfig fromPretrained(String modelId) throws IOException {
        return fromPretrained(modelId, HfHub.fromEnv());
    }

    /**
     * Load a config from a local directory containing {@code config.json}.
     *
     * @param dir local model directory
     * @return {@link PretrainedConfig}
     */
    public static PretrainedConfig fromDirectory(Path dir) throws IOException {
        return PretrainedConfig.fromDirectory(dir);
    }
}
