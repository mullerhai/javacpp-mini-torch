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
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.transformers.loading.WeightLoader;
import org.bytedeco.pytorch.llm.transformers.processor.ImageProcessor;
import org.bytedeco.pytorch.llm.transformers.processor.ImageProcessorFactory;
import org.bytedeco.pytorch.llm.transformers.modeling.BlipForImageToText;
import org.bytedeco.pytorch.nn.Module;

import java.io.IOException;
import java.nio.file.Path;

/**
 * HF {@code AutoModelForImageToText.from_pretrained} entry point.
 * Routes to Blip, GIT, InstructBlip, Llava, LlavaNext, Idefics2, PaliGemma, Kosmos2.
 */
public final class AutoModelForImageToText {

    private AutoModelForImageToText() {}

    public static final class Bundle {
        public final Module model;
        public final FastTokenizer tokenizer;
        public final ImageProcessor imageProcessor;
        public final PretrainedConfig config;
        public final Path snapshot;
        public final WeightLoader.LoadReport loadReport;
        public final String modelType;

        public Bundle(Module model, FastTokenizer tokenizer, ImageProcessor imageProcessor,
                      PretrainedConfig config, Path snapshot, WeightLoader.LoadReport loadReport,
                      String modelType) {
            this.model = model;
            this.tokenizer = tokenizer;
            this.imageProcessor = imageProcessor;
            this.config = config;
            this.snapshot = snapshot;
            this.loadReport = loadReport;
            this.modelType = modelType;
        }
    }

    public static Bundle fromPretrained(String modelId, HfHub hub) throws IOException {
        Path snap = hub.snapshotDownload(modelId, "main", "models", java.util.List.of(
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "preprocessor_config.json",
                "model.safetensors", "model.safetensors.index.json"));
        return fromDirectory(snap);
    }

    public static Bundle fromDirectory(Path dir) throws IOException {
        PretrainedConfig cfg = PretrainedConfig.fromDirectory(dir);
        String mt = cfg.modelType() == null ? "blip" : cfg.modelType().name().toLowerCase();
        FastTokenizer tok = null;
        Path tp = dir.resolve("tokenizer.json");
        if (java.nio.file.Files.isRegularFile(tp)) tok = FastTokenizer.fromFile(tp);
        ImageProcessor ip = ImageProcessorFactory.fromPretrained(dir);
        Module model = BlipForImageToText.fromConfig(cfg);
        WeightLoader.LoadReport rep = WeightLoader.loadAndBind(model, dir);
        return new Bundle(model, tok, ip, cfg, dir, rep, mt);
    }
}
