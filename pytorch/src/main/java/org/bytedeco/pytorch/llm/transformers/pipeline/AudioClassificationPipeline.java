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
import org.bytedeco.pytorch.llm.transformers.AutoModelForMultimodalLM;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("audio-classification")}.
 *
 * <p>Takes a raw audio waveform tensor and sampling rate, and returns a list of
 * top-K label → score maps sorted descending by confidence.
 *
 * <pre>{@code
 * AudioClassificationPipeline pipe = AudioClassificationPipeline.fromPretrained(
 *     "MIT/ast-finetuned-audioset-10-10-0.4593", hub);
 * List<Map<String, Object>> out = pipe.call(audioTensor, 16000);
 * }</pre>
 */
public final class AudioClassificationPipeline {

    private final AutoModelForMultimodalLM.Bundle bundle;

    public AudioClassificationPipeline(AutoModelForMultimodalLM.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static AudioClassificationPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new AudioClassificationPipeline(AutoModelForMultimodalLM.fromPretrained(modelId, hub));
    }

    public static AudioClassificationPipeline fromDirectory(Path dir) throws IOException {
        return new AudioClassificationPipeline(AutoModelForMultimodalLM.fromDirectory(dir));
    }

    public AutoModelForMultimodalLM.Bundle bundle() {
        return bundle;
    }

    /** Returns the top-K label predictions for the audio input. */
    public List<Map<String, Object>> call(Tensor audio, int samplingRate) {
        // TODO: encode audio via the processor, run the audio encoder forward,
        // apply a classification head, and return ranked {label, score} maps.
        return List.of();
    }

    public void close() {
        try { bundle.model().close(); } catch (Throwable ignored) {}
    }
}
