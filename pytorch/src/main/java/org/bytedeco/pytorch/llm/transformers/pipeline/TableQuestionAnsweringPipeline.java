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
 * HF {@code pipeline("table-question-answering")}.
 *
 * <p>Takes a tabular dataset (column names → list of values) and a natural-language
 * query; returns a free-form answer. Driven by a T5-style or TAPAS seq2seq model.
 *
 * <pre>{@code
 * TableQuestionAnsweringPipeline pipe = TableQuestionAnsweringPipeline.fromPretrained(
 *     "google/tapas-base-finetuned-wtq", hub);
 * Map<String, List<String>> table = Map.of(
 *     "Country", List.of("France", "Germany", "Italy"),
 *     "Population", List.of("67M", "83M", "60M"));
 * String answer = pipe.call(table, "Which country has the largest population?");
 * }</pre>
 */
public final class TableQuestionAnsweringPipeline {

    private final AutoModelForCausalLM.Bundle bundle;

    public TableQuestionAnsweringPipeline(AutoModelForCausalLM.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static TableQuestionAnsweringPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new TableQuestionAnsweringPipeline(AutoModelForCausalLM.fromPretrained(modelId, hub));
    }

    public static TableQuestionAnsweringPipeline fromDirectory(Path dir) throws IOException {
        return new TableQuestionAnsweringPipeline(AutoModelForCausalLM.fromDirectory(dir));
    }

    public AutoModelForCausalLM.Bundle bundle() {
        return bundle;
    }

    /** Returns the answer string for the given table and query. */
    public String call(Map<String, List<String>> table, String query) {
        // TODO: serialise the table into the seq2seq input format expected by the
        // TAPAS / T5 model, run forward, and return the decoded answer string.
        return "";
    }

    public void close() {
        try { bundle.model().close(); } catch (Throwable ignored) {}
    }
}
