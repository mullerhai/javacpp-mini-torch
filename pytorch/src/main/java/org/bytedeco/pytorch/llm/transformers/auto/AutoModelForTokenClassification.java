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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.transformers.loading.WeightLoader;
import org.bytedeco.pytorch.llm.transformers.mapping.ModelRegistry;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * HuggingFace {@code AutoModelForTokenClassification.from_pretrained} entry point.
 *
 * <p>Wraps a backbone causal LM with a token-classification head — Linear
 * from hidden_size → {@code numLabels}, applied per token (no pooling).
 *
 * <p>Typical use cases: NER, POS tagging, slot-filling, sequence labelling.
 *
 * <pre>{@code
 * AutoModelForTokenClassification.Bundle b = AutoModelForTokenClassification.fromPretrained(
 *     "dslim/bert-base-NER", hub);
 * long[][] tags = b.predict(List.of("John works at Google in New York"));
 * }</pre>
 */
public final class AutoModelForTokenClassification {

    private AutoModelForTokenClassification() {}

    /** Configuration for the token-classification head. */
    public static final class TokenClassificationConfig {
        public final PretrainedConfig base;
        public final int numLabels;
        public final String problemType;
        public final boolean padToMultipleOf;

        public TokenClassificationConfig(PretrainedConfig base, int numLabels,
                                          String problemType, boolean padToMultipleOf) {
            this.base = base;
            this.numLabels = numLabels;
            this.problemType = problemType == null ? "single_label_classification" : problemType;
            this.padToMultipleOf = padToMultipleOf;
        }
    }

    public static final class Bundle {
        private final Module model;
        private final FastTokenizer tokenizer;
        private final PretrainedConfig config;
        private final int numLabels;
        private final Path snapshot;
        private final WeightLoader.LoadReport loadReport;

        public Bundle(Module model, FastTokenizer tokenizer, PretrainedConfig config,
                      int numLabels, Path snapshot, WeightLoader.LoadReport loadReport) {
            this.model = model;
            this.tokenizer = tokenizer;
            this.config = config;
            this.numLabels = numLabels;
            this.snapshot = snapshot;
            this.loadReport = loadReport;
        }

        public Module model() { return model; }
        public FastTokenizer tokenizer() { return tokenizer; }
        public PretrainedConfig config() { return config; }
        public int numLabels() { return numLabels; }
        public Path snapshot() { return snapshot; }
        public WeightLoader.LoadReport loadReport() { return loadReport; }

        /** Per-token argmax predictions: {@code [batch, maxLen]}. */
        public long[][] predict(java.util.List<String> texts) {
            java.util.List<int[]> batch = new java.util.ArrayList<>();
            int maxLen = 0;
            for (String t : texts) {
                int[] ids = tokenizer.encode(t, true).ids();
                batch.add(ids);
                maxLen = Math.max(maxLen, ids.length);
            }
            if (maxLen == 0) return new long[texts.size()][0];

            Tensor input = org.bytedeco.pytorch.global.torch.zeros(new long[]{texts.size(), maxLen})
                    .to(org.bytedeco.pytorch.global.torch.ScalarType.Long);
            int[] lens = new int[texts.size()];
            for (int i = 0; i < batch.size(); i++) {
                int[] ids = batch.get(i);
                lens[i] = ids.length;
                for (int j = 0; j < ids.length; j++) {
                    input.select(0, i).select(0, j).fill_(new org.bytedeco.pytorch.Scalar((long) ids[j]));
                }
            }
            Tensor logits = model.forward(input);   // [B, T, numLabels]
            long[][] out = new long[texts.size()][];
            for (int i = 0; i < texts.size(); i++) {
                int len = lens[i];
                out[i] = new long[len];
                for (int j = 0; j < len; j++) {
                    long best = 0;
                    double bestVal = Double.NEGATIVE_INFINITY;
                    for (int c = 0; c < numLabels; c++) {
                        double v = logits.select(0, i).select(1, j).select(-1, c).item().toDouble();
                        if (v > bestVal) { bestVal = v; best = c; }
                    }
                    out[i][j] = best;
                }
            }
            return out;
        }
    }

    public static Module fromConfig(TokenClassificationConfig cfg) {
        Module base = ModelRegistry.create(cfg.base);
        Module head = new TokenClassifierHead((int) cfg.base.hiddenSize(), cfg.numLabels);
        return new TokenClassificationWrapper(base, head);
    }

    public static Bundle fromPretrained(String modelId, HfHub hub) throws IOException {
        return fromPretrained(modelId, hub, LoadOptions.DEFAULT);
    }

    public static Bundle fromPretrained(String modelId, HfHub hub, LoadOptions opts) throws IOException {
        AutoModelForCausalLM.Bundle base = AutoModelForCausalLM.fromPretrained(modelId, hub, toCausalLoadOptions(opts));
        return wrapWithClassifier(base);
    }

    public static Bundle fromDirectory(Path dir) throws IOException {
        return fromDirectory(dir, LoadOptions.DEFAULT);
    }

    public static Bundle fromDirectory(Path dir, LoadOptions opts) throws IOException {
        AutoModelForCausalLM.Bundle base = AutoModelForCausalLM.fromDirectory(dir, toCausalLoadOptions(opts));
        return wrapWithClassifier(base);
    }

    private static Bundle wrapWithClassifier(AutoModelForCausalLM.Bundle base) {
        PretrainedConfig cfg = base.config();
        int numLabels = 2;
        Path cfgJson = base.snapshot().resolve("config.json");
        if (Files.isRegularFile(cfgJson)) {
            try {
                Map<String, Object> raw = Json.decodeObject(Files.readString(cfgJson, StandardCharsets.UTF_8));
                if (raw.containsKey("num_labels")) {
                    numLabels = ((Number) raw.get("num_labels")).intValue();
                }
            } catch (Exception ignored) {
                // fall back to defaults
            }
        }
        Module head = new TokenClassifierHead((int) cfg.hiddenSize(), numLabels);
        Module wrapped = new TokenClassificationWrapper(base.model(), head);
        return new Bundle(wrapped, base.tokenizer(), cfg, numLabels,
                base.snapshot(), base.loadReport());
    }

    private static AutoModelForCausalLM.LoadOptions toCausalLoadOptions(LoadOptions opts) {
        AutoModelForCausalLM.LoadOptions out = new AutoModelForCausalLM.LoadOptions();
        out.bindMode = opts.bindMode;
        out.strict = false;
        out.loadWeights = true;
        out.zeroCopyMmap = (opts.bindMode == WeightLoader.BindMode.ZERO_COPY);
        return out;
    }

    public static final class LoadOptions {
        public final WeightLoader.BindMode bindMode;
        public LoadOptions(WeightLoader.BindMode bindMode) {
            this.bindMode = bindMode;
        }
        public static final LoadOptions DEFAULT = new LoadOptions(WeightLoader.BindMode.ZERO_COPY);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    private static final class TokenClassifierHead extends Module {
        private final org.bytedeco.pytorch.nn.modules.LinearImpl classifier;
        TokenClassifierHead(int hidden, int numLabels) {
            super("classifier");
            this.classifier = register_module("classifier",
                    new org.bytedeco.pytorch.nn.modules.LinearImpl(hidden, numLabels));
        }
        @Override public Tensor forward(Tensor input) {
            return classifier.forward(input);
        }
    }

    private static final class TokenClassificationWrapper extends Module {
        private final Module base;
        private final Module head;
        TokenClassificationWrapper(Module base, Module head) {
            super("TokenClassificationWrapper");
            this.base = register_module("base", base);
            this.head = register_module("head", head);
        }
        @Override public Tensor forward(Tensor input) {
            Tensor h = base.forward(input);  // [B, T, H]
            return head.forward(h);            // [B, T, numLabels]
        }
    }
}