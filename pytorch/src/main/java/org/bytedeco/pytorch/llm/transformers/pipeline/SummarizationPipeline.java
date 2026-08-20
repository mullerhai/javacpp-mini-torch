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
import org.bytedeco.pytorch.llm.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("summarization")}.
 *
 * <p>Takes a long text document and returns a concise summary. The input is
 * wrapped in a summarisation prompt before being fed to the model.
 *
 * <pre>{@code
 * SummarizationPipeline pipe = SummarizationPipeline.fromPretrained(
 *     "facebook/bart-large-cnn", hub);
 * String summary = pipe.call(longArticleText);
 * }</pre>
 */
public final class SummarizationPipeline {

    private final AutoModelForCausalLM.Bundle bundle;

    public SummarizationPipeline(AutoModelForCausalLM.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static SummarizationPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new SummarizationPipeline(AutoModelForCausalLM.fromPretrained(modelId, hub));
    }

    public static SummarizationPipeline fromDirectory(Path dir) throws IOException {
        return new SummarizationPipeline(AutoModelForCausalLM.fromDirectory(dir));
    }

    public AutoModelForCausalLM.Bundle bundle() {
        return bundle;
    }

    /** Summarises the input text. */
    public String call(String text) {
        String prompt = "Summarize: " + text;
        return bundle.generate(prompt, bundle.generationConfig());
    }

    /** Summarises with explicit generation config. */
    public String call(String text, GenerationConfig gen) {
        String prompt = "Summarize: " + text;
        return bundle.generate(prompt, gen);
    }

    /** Batch summarisation — returns one summary per input document. */
    public List<String> callBatch(List<String> texts) {
        List<String> out = new java.util.ArrayList<>(texts.size());
        for (String t : texts) out.add(call(t));
        return out;
    }

    public void close() {
        try { bundle.model().close(); } catch (Throwable ignored) {}
    }
}
