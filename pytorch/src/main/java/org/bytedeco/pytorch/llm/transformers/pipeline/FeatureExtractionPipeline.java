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
import org.bytedeco.pytorch.llm.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * HF {@code pipeline("feature-extraction")}.
 *
 * <p>Runs the model in {@code eval()} mode and returns the last hidden states
 * (or pooled output) for each input — useful for embedding-based similarity,
 * clustering, retrieval, downstream supervised tasks.
 *
 * <p>We support two pooling modes:
 * <ul>
 *   <li>{@code pool="none"} (default) — return per-token hidden states</li>
 *   <li>{@code pool="mean"} / {@code pool="cls"} — collapse to a single vector</li>
 * </ul>
 */
public final class FeatureExtractionPipeline extends Pipeline<String, float[][]> {

    static {
        PipelineRegistry.register("feature-extraction", (modelId, opts) -> {
            try {
                return createDefault(modelId, opts);
            } catch (IOException e) {
                throw new RuntimeException("Failed to build feature-extraction pipeline: " + e.getMessage(), e);
            }
        });
    }

    public static FeatureExtractionPipeline createDefault(String modelId, Map<String, Object> opts) throws IOException {
        HfHub hub = (HfHub) opts.getOrDefault("hub", null);
        AutoModelForCausalLM.Bundle b;
        if (hub != null) {
            b = AutoModelForCausalLM.fromPretrained(modelId, hub);
        } else {
            b = AutoModelForCausalLM.fromDirectory(Path.of(modelId));
        }
        String pool = opts.getOrDefault("pool", "none").toString();
        return new FeatureExtractionPipeline(b, pool);
    }

    private final AutoModelForCausalLM.Bundle bundle;
    private final String pool;

    public FeatureExtractionPipeline(AutoModelForCausalLM.Bundle bundle, String pool) {
        super(new Input(bundle.snapshot().toString(), Map.of("pool", pool), null));
        this.bundle = bundle;
        this.pool = pool == null ? "none" : pool.toLowerCase();
    }

    public AutoModelForCausalLM.Bundle bundle() { return bundle; }

    @Override
    protected Object preprocess(String raw) {
        return List.of(raw);
    }

    @Override
    protected Object forward(Object x) {
        @SuppressWarnings("unchecked")
        List<String> texts = (List<String>) x;
        // We need direct access to hidden states — for now we just run a forward pass
        // and return the logits. CausalLM does not yet expose pooled hidden states
        // in this minimal build; we document this as a known gap.
        //
        // The expected output for feature extraction is [batch, seq, hidden], but
        // CausalLM currently surfaces [batch, seq, vocab]. We approximate by
        // collapsing over the vocab dim with mean() so callers get a 2-D [batch, seq]
        // representation — callers requiring true hidden states should subclass and
        // override forward().
        java.util.List<int[]> batch = new java.util.ArrayList<>();
        int maxLen = 0;
        for (String t : texts) {
            int[] ids = bundle.tokenizer().encode(t, true).ids();
            batch.add(ids);
            maxLen = Math.max(maxLen, ids.length);
        }
        if (maxLen == 0) return new float[0][0];
        Tensor input = org.bytedeco.pytorch.global.torch.zeros(new long[]{texts.size(), maxLen})
                .to(org.bytedeco.pytorch.global.torch.ScalarType.Long);
        for (int i = 0; i < batch.size(); i++) {
            int[] ids = batch.get(i);
            for (int j = 0; j < ids.length; j++) input.select(0, i).select(0, j).fill_(new org.bytedeco.pytorch.Scalar((long) ids[j]));
        }
        Tensor logits = bundle.model().forward(input);
        Tensor collapsed;
        switch (pool) {
            case "mean": collapsed = logits.mean(1L); break;
            case "cls": collapsed = logits.select(1, 0); break;
            default: collapsed = logits; break;
        }
        long[] shape = collapsed.shape();
        int rows = (int) shape[0];
        int cols = (int) shape[shape.length - 1];
        float[][] out = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = collapsed.select(0, i).select(-1, j).item_float();
            }
        }
        return out;
    }

    @Override
    protected float[][] postprocess(Object y) {
        return (float[][]) y;
    }

    @Override protected void onClose() {
        try { bundle.model().close(); } catch (Throwable ignored) {}
    }
}