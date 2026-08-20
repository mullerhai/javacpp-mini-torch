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

import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForTokenClassification;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * HF {@code pipeline("token-classification")} / {@code pipeline("ner")}.
 *
 * <p>Aggregates word-level labels from subword token predictions using a
 * configurable aggregation strategy (default: averaging subword scores per
 * word).
 */
public final class TokenClassificationPipeline extends Pipeline<List<String>, List<Map<String, Object>>> {

    public enum AggregationStrategy {
        NONE,        // one label per sub-token
        FIRST,       // first sub-token's label
        AVERAGE,     // mean over subword logits (default)
        MAX          // max over subword logits
    }

    static {
        PipelineRegistry.register("token-classification", (modelId, opts) -> {
            try {
                return createDefault(modelId, opts);
            } catch (IOException e) {
                throw new RuntimeException("Failed to build token-classification pipeline: " + e.getMessage(), e);
            }
        });
    }

    public static TokenClassificationPipeline createDefault(String modelId, Map<String, Object> opts) throws IOException {
        HfHub hub = (HfHub) opts.getOrDefault("hub", null);
        AutoModelForTokenClassification.Bundle b;
        if (hub != null) {
            b = AutoModelForTokenClassification.fromPretrained(modelId, hub);
        } else {
            b = AutoModelForTokenClassification.fromDirectory(java.nio.file.Path.of(modelId));
        }
        AggregationStrategy strategy = AggregationStrategy.AVERAGE;
        if (opts.containsKey("aggregationStrategy")) {
            try {
                strategy = AggregationStrategy.valueOf(opts.get("aggregationStrategy").toString().toUpperCase());
            } catch (Exception ignored) {}
        }
        return new TokenClassificationPipeline(b, strategy);
    }

    private final AutoModelForTokenClassification.Bundle bundle;
    private final AggregationStrategy strategy;
    private final List<String> id2label;

    public TokenClassificationPipeline(AutoModelForTokenClassification.Bundle bundle,
                                        AggregationStrategy strategy) {
        super(new Input(bundle.snapshot().toString(),
                Map.of("aggregationStrategy", strategy.name()), null));
        this.bundle = bundle;
        this.strategy = strategy;
        int n = bundle.numLabels();
        java.util.List<String> labels = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) labels.add("LABEL_" + i);
        this.id2label = labels;
    }

    public AutoModelForTokenClassification.Bundle bundle() { return bundle; }

    @Override
    protected Object preprocess(List<String> raw) {
        return raw;
    }

    @Override
    protected Object forward(Object x) {
        @SuppressWarnings("unchecked")
        List<String> texts = (List<String>) x;
        return bundle.predict(texts);
    }

    @SuppressWarnings("unchecked")
    @Override
    protected List<Map<String, Object>> postprocess(Object y) {
        long[][] tags = (long[][]) y;
        List<Map<String, Object>> out = new ArrayList<>();
        for (long[] row : tags) {
            List<Map<String, Object>> wordTags = new ArrayList<>();
            for (long t : row) {
                String label = t < id2label.size() ? id2label.get((int) t) : ("LABEL_" + t);
                wordTags.add(Map.of("label", label));
            }
            out.add(Map.of("tags", wordTags));
        }
        return out;
    }

    @Override protected void onClose() {
        try { bundle.model().close(); } catch (Throwable ignored) {}
    }
}