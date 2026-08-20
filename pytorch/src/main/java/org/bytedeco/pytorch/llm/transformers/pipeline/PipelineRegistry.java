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

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry of HF-style pipelines keyed by task name. Mirrors HF's
 * {@code pipeline.task_aliases} + {@code SUPPORTED_TASKS} registry.
 *
 * <p>Add new pipelines via {@link #register}; resolve at runtime via
 * {@link #create(String, String, Map)} which loads the requested model and
 * wraps it in the appropriate {@link Pipeline} subclass.
 */
public final class PipelineRegistry {

    public interface Factory {
        Pipeline create(String modelId, Map<String, Object> options);
    }

    private static final Map<String, Factory> FACTORIES = new ConcurrentHashMap<>();
    private static final Map<String, String> ALIASES = new ConcurrentHashMap<>();

    static {
        // Default task aliases
        ALIASES.put("text-generation", "text-generation");
        ALIASES.put("text2text-generation", "text2text-generation");
        ALIASES.put("summarization", "text2text-generation");
        ALIASES.put("translation", "text2text-generation");
        ALIASES.put("fill-mask", "fill-mask");
        ALIASES.put("text-classification", "text-classification");
        ALIASES.put("sentiment-analysis", "text-classification");
        ALIASES.put("ner", "token-classification");
        ALIASES.put("token-classification", "token-classification");
        ALIASES.put("question-answering", "question-answering");
        ALIASES.put("feature-extraction", "feature-extraction");
        ALIASES.put("image-classification", "image-classification");
        ALIASES.put("image-to-text", "image-to-text");
        ALIASES.put("automatic-speech-recognition", "automatic-speech-recognition");
        ALIASES.put("audio-classification", "audio-classification");
        ALIASES.put("zero-shot-classification", "zero-shot-classification");
    }

    private PipelineRegistry() {}

    public static void register(String task, Factory factory) {
        FACTORIES.put(task.toLowerCase(), factory);
    }

    public static void alias(String alias, String realTask) {
        ALIASES.put(alias.toLowerCase(), realTask.toLowerCase());
    }

    public static String resolve(String task) {
        String t = task.toLowerCase();
        return ALIASES.getOrDefault(t, t);
    }

    public static List<String> supportedTasks() {
        return List.copyOf(FACTORIES.keySet());
    }

    public static Pipeline create(String task, String modelId, Map<String, Object> options) {
        String resolved = resolve(task);
        Factory f = FACTORIES.get(resolved);
        if (f == null) {
            throw new IllegalArgumentException("Unsupported pipeline task: " + task
                    + " (resolved: " + resolved + "). Registered: " + FACTORIES.keySet());
        }
        return f.create(modelId, options == null ? Map.of() : options);
    }

    public static boolean isRegistered(String task) {
        return FACTORIES.containsKey(resolve(task));
    }
}