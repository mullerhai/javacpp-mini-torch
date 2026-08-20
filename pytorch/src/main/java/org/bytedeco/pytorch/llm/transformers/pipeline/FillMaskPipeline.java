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
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForMaskedLM;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * HF {@code pipeline("fill-mask")}.
 *
 * <p>Takes a text containing the mask token ({@code [MASK]} for BERT-family,
 * {@code <mask>} for RoBERTa) and returns the top-K predictions for it.
 */
public final class FillMaskPipeline extends Pipeline<String, List<Map<String, Object>>> {

    static {
        PipelineRegistry.register("fill-mask", (modelId, opts) -> {
            try {
                return createDefault(modelId, opts);
            } catch (IOException e) {
                throw new RuntimeException("Failed to build fill-mask pipeline: " + e.getMessage(), e);
            }
        });
    }

    public static FillMaskPipeline createDefault(String modelId, Map<String, Object> opts) throws IOException {
        HfHub hub = (HfHub) opts.getOrDefault("hub", null);
        AutoModelForMaskedLM.Bundle b;
        if (hub != null) {
            b = AutoModelForMaskedLM.fromPretrained(modelId, hub);
        } else {
            b = AutoModelForMaskedLM.fromDirectory(java.nio.file.Path.of(modelId));
        }
        int topK = opts.containsKey("topK") ? ((Number) opts.get("topK")).intValue() : 5;
        return new FillMaskPipeline(b, topK);
    }

    private final AutoModelForMaskedLM.Bundle bundle;
    private final int topK;

    public FillMaskPipeline(AutoModelForMaskedLM.Bundle bundle, int topK) {
        super(new Input(bundle.snapshot().toString(), Map.of("topK", topK), null));
        this.bundle = bundle;
        this.topK = topK > 0 ? topK : 5;
    }

    @Override
    protected Object preprocess(String raw) {
        return raw;
    }

    @Override
    protected Object forward(Object x) {
        String text = (String) x;
        return bundle.predictMask(text, topK);
    }

    @SuppressWarnings("unchecked")
    @Override
    protected List<Map<String, Object>> postprocess(Object y) {
        List<AutoModelForMaskedLM.Prediction> preds = (List<AutoModelForMaskedLM.Prediction>) y;
        List<Map<String, Object>> out = new ArrayList<>();
        for (AutoModelForMaskedLM.Prediction p : preds) {
            out.add(Map.of(
                    "token", p.token,
                    "tokenId", p.tokenId,
                    "score", p.score
            ));
        }
        return out;
    }

    @Override protected void onClose() {
        try { bundle.close(); } catch (Throwable ignored) {}
    }
}