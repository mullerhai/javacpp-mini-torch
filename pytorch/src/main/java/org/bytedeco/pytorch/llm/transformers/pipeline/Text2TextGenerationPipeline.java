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
 * HF {@code pipeline("text2text-generation")}.
 *
 * <p>General-purpose text-to-text pipeline that feeds the input string directly
 * to the model as a prompt. Suitable for any seq2seq-style task (translation,
 * summarisation, paraphrasing) when paired with an appropriate model.
 *
 * <pre>{@code
 * Text2TextGenerationPipeline pipe = Text2TextGenerationPipeline.fromPretrained(
 *     "google/flan-t5-base", hub);
 * String result = pipe.call("Translate to French: Hello world");
 * }</pre>
 */
public final class Text2TextGenerationPipeline {

    private final AutoModelForCausalLM.Bundle bundle;

    public Text2TextGenerationPipeline(AutoModelForCausalLM.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static Text2TextGenerationPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new Text2TextGenerationPipeline(AutoModelForCausalLM.fromPretrained(modelId, hub));
    }

    public static Text2TextGenerationPipeline fromDirectory(Path dir) throws IOException {
        return new Text2TextGenerationPipeline(AutoModelForCausalLM.fromDirectory(dir));
    }

    public AutoModelForCausalLM.Bundle bundle() {
        return bundle;
    }

    /** Feeds the text input directly as a prompt to the model. */
    public String call(String text) {
        return bundle.generate(text, bundle.generationConfig());
    }

    /** Feeds the text with explicit generation config. */
    public String call(String text, GenerationConfig gen) {
        return bundle.generate(text, gen);
    }

    public void close() {
        try { bundle.model().close(); } catch (Throwable ignored) {}
    }
}
