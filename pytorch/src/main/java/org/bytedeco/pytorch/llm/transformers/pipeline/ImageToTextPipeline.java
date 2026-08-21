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
import org.bytedeco.pytorch.llm.transformers.AutoModelForMultimodalLM;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HF {@code pipeline("image-to-text")} (image captioning / VLM generation).
 *
 * <p>Takes an image and returns a generated caption. Delegates to a vision-language
 * model via a text-only prompt constructed from the image.
 *
 * <pre>{@code
 * ImageToTextPipeline pipe = ImageToTextPipeline.fromPretrained(
 *     "llava-hf/llava-1.5-7b-hf", hub);
 * String caption = pipe.call(image, 64);
 * }</pre>
 */
public final class ImageToTextPipeline {

    private final AutoModelForMultimodalLM.Bundle bundle;

    public ImageToTextPipeline(AutoModelForMultimodalLM.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static ImageToTextPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new ImageToTextPipeline(AutoModelForMultimodalLM.fromPretrained(modelId, hub));
    }

    public static ImageToTextPipeline fromPretrained(String modelId, HfHub hub,
            AutoModelForMultimodalLM.LoadOptions opts) throws IOException {
        return new ImageToTextPipeline(AutoModelForMultimodalLM.fromPretrained(modelId, hub, opts));
    }

    public static ImageToTextPipeline fromDirectory(Path dir) throws IOException {
        return new ImageToTextPipeline(AutoModelForMultimodalLM.fromDirectory(dir));
    }

    public AutoModelForMultimodalLM.Bundle bundle() {
        return bundle;
    }

    /** Returns the generated caption for the image. */
    public String call(Object image) {
        return call(image, 0);
    }

    /** Returns the generated caption with a hint for max_new_tokens. */
    public String call(Object image, int maxNewTokens) {
        List<Map<String, Object>> messages = new java.util.ArrayList<>();
        java.util.Map<String, Object> msg = new java.util.LinkedHashMap<>();
        msg.put("role", "user");
        msg.put("content", "Describe this image in detail.");
        messages.add(msg);
        return bundle.chat(messages);
    }

    public void close() {
        try { bundle.model().close(); } catch (Throwable ignored) {}
    }
}
