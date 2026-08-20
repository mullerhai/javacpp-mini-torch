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
import java.util.Objects;

/**
 * HuggingFace {@code AutoModelForSequenceClassification.from_pretrained} entry point.
 *
 * <p>Wraps a backbone causal LM (Qwen2 / Llama / Mistral / etc.) with a
 * sequence-classification head — mean-pool the per-token hidden states,
 * then project to {@code numLabels} logits.
 *
 * <p>Mirrors HF's {@code AutoModelForSequenceClassification}; routes by
 * {@code architectures} → {@code model_type} fallback. The classifier head
 * uses the same architecture as HF (mean-pool + Linear) so standard fine-tuning
 * workflows apply.
 *
 * <pre>{@code
 * AutoModelForSequenceClassification.Bundle b = AutoModelForSequenceClassification.fromPretrained(
 *     "cardiffnlp/twitter-roberta-base-sentiment-latest", hub);
 * long[] predIds = b.predict(List.of("I love this movie.", "I hate it."));
 * }</pre>
 */
public final class AutoModelForSequenceClassification {

    private AutoModelForSequenceClassification() {}

    /** Config bundle: classifier head + base LM. */
    public static final class SequenceClassificationConfig {
        public final PretrainedConfig base;
        public final int numLabels;
        public final String problemType;
        public final String classifierDropout;
        public final boolean padToMultipleOf;

        public SequenceClassificationConfig(PretrainedConfig base, int numLabels, String problemType,
                                            String classifierDropout, boolean padToMultipleOf) {
            this.base = Objects.requireNonNull(base);
            this.numLabels = numLabels;
            this.problemType = problemType == null ? "single_label_classification" : problemType;
            this.classifierDropout = classifierDropout;
            this.padToMultipleOf = padToMultipleOf;
        }
    }

    /** Loaded model + tokenizer + classification head. */
    public static final class Bundle {
        private final Module model;
        private final FastTokenizer tokenizer;
        private final PretrainedConfig config;
        private final int numLabels;
        private final String problemType;
        private final Path snapshot;
        private final WeightLoader.LoadReport loadReport;

        public Bundle(Module model, FastTokenizer tokenizer, PretrainedConfig config,
                      int numLabels, String problemType, Path snapshot,
                      WeightLoader.LoadReport loadReport) {
            this.model = model;
            this.tokenizer = tokenizer;
            this.config = config;
            this.numLabels = numLabels;
            this.problemType = problemType == null ? "single_label_classification" : problemType;
            this.snapshot = snapshot;
            this.loadReport = loadReport;
        }

        public Module model() { return model; }
        public FastTokenizer tokenizer() { return tokenizer; }
        public PretrainedConfig config() { return config; }
        public int numLabels() { return numLabels; }
        public String problemType() { return problemType; }
        public Path snapshot() { return snapshot; }
        public WeightLoader.LoadReport loadReport() { return loadReport; }

        /** Run inference on a batch of texts; returns argmax label per row. */
        public long[] predict(java.util.List<String> texts) {
            long[] out = new long[texts.size()];
            java.util.List<int[]> batch = new java.util.ArrayList<>();
            int maxLen = 0;
            for (String t : texts) {
                int[] ids = tokenizer.encode(t, true).ids();
                batch.add(ids);
                maxLen = Math.max(maxLen, ids.length);
            }
            if (maxLen == 0) return out;
            Tensor input = org.bytedeco.pytorch.global.torch.zeros(new long[]{texts.size(), maxLen})
                    .to(org.bytedeco.pytorch.global.torch.ScalarType.Long);
            for (int i = 0; i < batch.size(); i++) {
                int[] ids = batch.get(i);
                for (int j = 0; j < ids.length; j++) {
                    input.select(0, i).select(0, j).fill_(new org.bytedeco.pytorch.Scalar((long) ids[j]));
                }
            }
            Tensor logits = model.forward(input);
            long[] shape = logits.shape();
            int lastDim = (int) shape[shape.length - 1];
            for (int i = 0; i < texts.size(); i++) {
                long best = 0;
                double bestVal = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < lastDim; c++) {
                    double v = logits.select(0, i).select(-1, c).item().toDouble();
                    if (v > bestVal) { bestVal = v; best = c; }
                }
                out[i] = best;
            }
            return out;
        }

        /** Compute softmax probabilities (for thresholding, calibration). */
        public double[][] predictProba(java.util.List<String> texts) {
            java.util.List<int[]> batch = new java.util.ArrayList<>();
            int maxLen = 0;
            for (String t : texts) {
                int[] ids = tokenizer.encode(t, true).ids();
                batch.add(ids);
                maxLen = Math.max(maxLen, ids.length);
            }
            if (maxLen == 0) return new double[texts.size()][0];

            Tensor input = org.bytedeco.pytorch.global.torch.zeros(new long[]{texts.size(), maxLen})
                    .to(org.bytedeco.pytorch.global.torch.ScalarType.Long);
            for (int i = 0; i < batch.size(); i++) {
                int[] ids = batch.get(i);
                for (int j = 0; j < ids.length; j++) {
                    input.select(0, i).select(0, j).fill_(new org.bytedeco.pytorch.Scalar((long) ids[j]));
                }
            }
            Tensor logits = model.forward(input);
            Tensor probs = logits.softmax(-1);
            long[] shape = probs.shape();
            int lastDim = (int) shape[shape.length - 1];
            double[][] out = new double[texts.size()][lastDim];
            for (int i = 0; i < texts.size(); i++) {
                for (int c = 0; c < lastDim; c++) {
                    out[i][c] = probs.select(0, i).select(-1, c).item().toDouble();
                }
            }
            return out;
        }
    }

    /** Build a base causal LM and attach a sequence-classification head. */
    public static Module fromConfig(SequenceClassificationConfig cfg) {
        Module base = ModelRegistry.create(cfg.base);
        Module head = new ClassifierHead((int) cfg.base.hiddenSize(), cfg.numLabels);
        return new SequenceClassificationWrapper(base, head);
    }

    public static Bundle fromPretrained(String modelId, HfHub hub) throws IOException {
        return fromPretrained(modelId, hub, LoadOptions.DEFAULT);
    }

    public static Bundle fromPretrained(String modelId, HfHub hub, LoadOptions opts) throws IOException {
        // Delegate the actual snapshot+config+tokenizer+weights flow to AutoModelForCausalLM
        // (which we then "wrap" by appending a classifier head).
        AutoModelForCausalLM.Bundle base = AutoModelForCausalLM.fromPretrained(modelId, hub,
                toCausalLoadOptions(opts));
        return wrapWithClassifier(base, opts);
    }

    public static Bundle fromDirectory(Path dir) throws IOException {
        return fromDirectory(dir, LoadOptions.DEFAULT);
    }

    public static Bundle fromDirectory(Path dir, LoadOptions opts) throws IOException {
        AutoModelForCausalLM.Bundle base = AutoModelForCausalLM.fromDirectory(dir,
                toCausalLoadOptions(opts));
        return wrapWithClassifier(base, opts);
    }

    private static Bundle wrapWithClassifier(AutoModelForCausalLM.Bundle base, LoadOptions opts) {
        PretrainedConfig cfg = base.config();
        int numLabels = 2;
        String problemType = "single_label_classification";
        String classifierDropout = null;
        Path cfgJson = base.snapshot().resolve("config.json");
        if (Files.isRegularFile(cfgJson)) {
            try {
                Map<String, Object> raw = Json.decodeObject(Files.readString(cfgJson, StandardCharsets.UTF_8));
                if (raw.containsKey("num_labels")) {
                    numLabels = ((Number) raw.get("num_labels")).intValue();
                }
                Object pt = raw.get("problem_type");
                if (pt instanceof String) problemType = (String) pt;
                Object cd = raw.get("classifier_dropout");
                if (cd instanceof String) classifierDropout = (String) cd;
            } catch (Exception ignored) {
                // fall through to defaults
            }
        }

        SequenceClassificationConfig seqCfg = new SequenceClassificationConfig(
                cfg, numLabels, problemType, classifierDropout, false);
        Module wrapped = new SequenceClassificationWrapper(base.model(),
                new ClassifierHead((int) cfg.hiddenSize(), numLabels));
        return new Bundle(wrapped, base.tokenizer(), cfg, numLabels, problemType,
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

    /** Load options. */
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

    private static final class ClassifierHead extends Module {
        private final org.bytedeco.pytorch.nn.modules.LinearImpl classifier;
        ClassifierHead(int hidden, int numLabels) {
            super("classifier");
            this.classifier = register_module("classifier",
                    new org.bytedeco.pytorch.nn.modules.LinearImpl(hidden, numLabels));
        }
        @Override public Tensor forward(Tensor input) {
            return classifier.forward(input);
        }
    }

    private static final class SequenceClassificationWrapper extends Module {
        private final Module base;
        private final Module head;
        SequenceClassificationWrapper(Module base, Module head) {
            super("SequenceClassificationWrapper");
            this.base = register_module("base", base);
            this.head = register_module("head", head);
        }
        @Override public Tensor forward(Tensor input) {
            Tensor h = base.forward(input);
            // mean-pool over the sequence dim (position 1 for [B, T, H])
            if (h.dim() >= 3) {
                h = h.mean(1L);
            }
            return head.forward(h);
        }
    }
}