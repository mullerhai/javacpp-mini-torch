/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "Classpath" exception),
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
 * HF {@code pipeline("translation")}.
 *
 * <p>Translates text from a source language to a target language using a
 * sequence-to-sequence model. The source and target languages are typically
 * baked into the model name (e.g. {@code Helsinki-NLP/opus-mt-en-de}).
 *
 * <pre>{@code
 * TranslationPipeline pipe = TranslationPipeline.fromPretrained(
 *     "Helsinki-NLP/opus-mt-en-de", hub);
 * String translated = pipe.call("Hello world", "en", "de");
 * }</pre>
 */
public final class TranslationPipeline {

    private final AutoModelForCausalLM.Bundle bundle;

    public TranslationPipeline(AutoModelForCausalLM.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static TranslationPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new TranslationPipeline(AutoModelForCausalLM.fromPretrained(modelId, hub));
    }

    public static TranslationPipeline fromDirectory(Path dir) throws IOException {
        return new TranslationPipeline(AutoModelForCausalLM.fromDirectory(dir));
    }

    public AutoModelForCausalLM.Bundle bundle() {
        return bundle;
    }

    /** Translates the source text. Language codes may be ignored if the model is language-specific. */
    public String call(String text, String srcLang, String tgtLang) {
        // Format the prompt with language context so the model knows the translation direction.
        String prompt = String.format("Translate from %s to %s: %s", srcLang, tgtLang, text);
        return bundle.generate(prompt, bundle.generationConfig());
    }

    /** Translates with an explicit generation config (max_new_tokens, temperature, etc.). */
    public String call(String text, String srcLang, String tgtLang, GenerationConfig gen) {
        String prompt = String.format("Translate from %s to %s: %s", srcLang, tgtLang, text);
        return bundle.generate(prompt, gen);
    }

    public void close() {
        try { bundle.model().close(); } catch (Throwable ignored) {}
    }
}
