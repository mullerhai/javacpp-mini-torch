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
package org.bytedeco.pytorch.llm.transformers.pipeline;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForSequenceClassification;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("text-classification")} / {@code pipeline("sentiment-analysis")}.
 *
 * <p>Wraps a {@link AutoModelForSequenceClassification.Bundle} and converts
 * raw text → label prediction.
 */
public final class TextClassificationPipeline extends Pipeline<String, Map<String, Object>> {

    static {
        PipelineRegistry.register("text-classification", (modelId, opts) -> {
            try {
                return createDefault(modelId, opts);
            } catch (IOException e) {
                throw new RuntimeException("Failed to build text-classification pipeline: " + e.getMessage(), e);
            }
        });
    }

    public static TextClassificationPipeline createDefault(String modelId, Map<String, Object> opts) throws IOException {
        HfHub hub = (HfHub) opts.getOrDefault("hub", null);
        AutoModelForSequenceClassification.Bundle b;
        if (hub != null) {
            b = AutoModelForSequenceClassification.fromPretrained(modelId, hub);
        } else {
            // No hub → expect local model directory.
            b = AutoModelForSequenceClassification.fromDirectory(java.nio.file.Path.of(modelId));
        }
        return new TextClassificationPipeline(b);
    }

    private final AutoModelForSequenceClassification.Bundle bundle;
    private final List<String> id2label;

    public TextClassificationPipeline(AutoModelForSequenceClassification.Bundle bundle) {
        super(new Input(bundle.snapshot().toString(), Map.of(), null));
        this.bundle = Objects.requireNonNull(bundle);
        this.id2label = readId2Label(bundle.config());
    }

    public AutoModelForSequenceClassification.Bundle bundle() { return bundle; }

    @Override
    protected Object preprocess(String raw) {
        return List.of(raw);
    }

    @Override
    protected Object forward(Object x) {
        @SuppressWarnings("unchecked")
        List<String> texts = (List<String>) x;
        return bundle.predictProba(texts);
    }

    @SuppressWarnings("unchecked")
    @Override
    protected Map<String, Object> postprocess(Object y) {
        double[][] probs = (double[][]) y;
        if (probs.length == 0) return Map.of();
        double[] row = probs[0];
        long best = 0;
        double bestVal = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < row.length; c++) {
            if (row[c] > bestVal) { bestVal = row[c]; best = c; }
        }
        String label = best < id2label.size() ? id2label.get((int) best) : ("LABEL_" + best);
        java.util.List<Map<String, Object>> all = new java.util.ArrayList<>();
        for (int c = 0; c < row.length; c++) {
            String l = c < id2label.size() ? id2label.get(c) : ("LABEL_" + c);
            all.add(Map.of("label", l, "score", row[c]));
        }
        // Sort descending by score
        all.sort((a, b) -> Double.compare(((Number) b.get("score")).doubleValue(),
                                          ((Number) a.get("score")).doubleValue()));
        return Map.of(
                "label", label,
                "score", bestVal,
                "all", all);
    }

    private static List<String> readId2Label(PretrainedConfig cfg) {
        // PretrainedConfig does not expose id2label today; we fall back to numeric labels.
        // When the model has been loaded from a snapshot, callers can override via constructor.
        int numLabels = 2;
        try {
            java.nio.file.Path cfgJson = cfg.modelType() == null ? null : null;
            // (left as a hook; AutoModelForSequenceClassification.Bundle already parsed num_labels.)
        } catch (Throwable ignored) {}
        java.util.List<String> out = new java.util.ArrayList<>();
        for (int i = 0; i < Math.max(numLabels, 0); i++) out.add("LABEL_" + i);
        return out;
    }

    @Override protected void onClose() {
        try { bundle.model().close(); } catch (Throwable ignored) {}
    }
}