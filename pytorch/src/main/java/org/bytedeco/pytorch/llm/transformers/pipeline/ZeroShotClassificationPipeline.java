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
import org.bytedeco.pytorch.llm.transformers.auto.AutoModelForSequenceClassification;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("zero-shot-classification")}.
 *
 * <p>Runs NLI-style inference using a pre-trained sequence classifier by
 * forming hypothesis-template prompts and scoring against candidate labels.
 * Input: {@code (text, candidate_labels, hypothesis_template)}.
 *
 * <pre>{@code
 * ZeroShotClassificationPipeline pipe = ZeroShotClassificationPipeline.fromPretrained(
 *     "facebook/bart-large-mnli", hub);
 * List<Map<String, Object>> out = pipe.call("I love this movie",
 *         List.of("positive", "negative"),
 *         "This example is {}.");
 * }</pre>
 */
public final class ZeroShotClassificationPipeline {

    private final AutoModelForSequenceClassification.Bundle bundle;

    public ZeroShotClassificationPipeline(AutoModelForSequenceClassification.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static ZeroShotClassificationPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new ZeroShotClassificationPipeline(AutoModelForSequenceClassification.fromPretrained(modelId, hub));
    }

    public static ZeroShotClassificationPipeline fromDirectory(Path dir) throws IOException {
        return new ZeroShotClassificationPipeline(AutoModelForSequenceClassification.fromDirectory(dir));
    }

    public AutoModelForSequenceClassification.Bundle bundle() {
        return bundle;
    }

    /** Score the candidate labels against the input text and hypothesis template. */
    public List<Map<String, Object>> call(String text, List<String> candidateLabels, String hypothesisTemplate) {
        // TODO: encode each candidate via the template, run entailment scoring with the
        // base model, softmax across candidates, and return [{label, score}, ...].
        return List.of();
    }

    public void close() {
        try { bundle.model().close(); } catch (Throwable ignored) {}
    }
}
