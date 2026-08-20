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
 * HF {@code pipeline("automatic-speech-recognition")}.
 *
 * <p>Takes a raw audio waveform tensor and its sampling rate, and returns the
 * transcribed text. Supports Whisper, Wav2Vec2, and conformer ASR models.
 *
 * <pre>{@code
 * AutomaticSpeechRecognitionPipeline pipe = AutomaticSpeechRecognitionPipeline.fromPretrained(
 *     "openai/whisper-base", hub);
 * String transcript = pipe.call(audioTensor, 16000);
 * }</pre>
 */
public final class AutomaticSpeechRecognitionPipeline {

    private final AutoModelForMultimodalLM.Bundle bundle;

    public AutomaticSpeechRecognitionPipeline(AutoModelForMultimodalLM.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static AutomaticSpeechRecognitionPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new AutomaticSpeechRecognitionPipeline(AutoModelForMultimodalLM.fromPretrained(modelId, hub));
    }

    public static AutomaticSpeechRecognitionPipeline fromPretrained(String modelId, HfHub hub,
            AutoModelForMultimodalLM.LoadOptions opts) throws IOException {
        return new AutomaticSpeechRecognitionPipeline(AutoModelForMultimodalLM.fromPretrained(modelId, hub, opts));
    }

    public static AutomaticSpeechRecognitionPipeline fromDirectory(Path dir) throws IOException {
        return new AutomaticSpeechRecognitionPipeline(AutoModelForMultimodalLM.fromDirectory(dir));
    }

    public AutoModelForMultimodalLM.Bundle bundle() {
        return bundle;
    }

    /** Transcribes the audio to text using the model's default generation config. */
    public String call(Tensor audio, int samplingRate) {
        // TODO: encode audio via the audio processor, run ASR model forward,
        // decode token IDs back to text, and return the transcript.
        return "";
    }

    /** Transcribes the audio with an explicit generation config. */
    public String call(Tensor audio, int samplingRate, GenerationConfig gen) {
        // TODO: same as above but with explicit gen config for task / language control.
        return "";
    }

    public void close() {
        try { bundle.model().close(); } catch (Throwable ignored) {}
    }
}
