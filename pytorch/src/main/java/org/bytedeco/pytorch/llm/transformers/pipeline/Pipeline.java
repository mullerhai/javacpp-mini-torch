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

/**
 * HuggingFace {@code Pipeline} base class. Mirrors
 * {@code transformers/pipelines/base.py:Pipeline}.
 *
 * <p>Subclasses implement the model-specific call logic:
 * <ul>
 *   <li>{@link #preprocess(Object)} — raw input → tokenized tensors</li>
 *   <li>{@link #forward(Object)} — tensors → model output</li>
 *   <li>{@link #postprocess(Object)} — model output → user-facing result</li>
 * </ul>
 *
 * <p>{@link #call(Object, Map)} chains them.
 */
public abstract class Pipeline<I, O> implements AutoCloseable {

    public static final class Input {
        public final String modelId;
        public final Map<String, Object> options;
        public final Object hub;        // HfHub if available; kept opaque
        public Input(String modelId, Map<String, Object> options, Object hub) {
            this.modelId = modelId;
            this.options = options == null ? Map.of() : options;
            this.hub = hub;
        }
    }

    private final Input input;
    private volatile boolean closed;

    protected Pipeline(Input input) {
        this.input = input;
    }

    public Input input() { return input; }
    public String modelId() { return input.modelId; }
    public Map<String, Object> options() { return input.options; }

    /** Run the pipeline end-to-end on a single input. */
    public O call(I inputData) {
        return call(inputData, Map.of());
    }

    /** Run the pipeline with per-call parameter overrides. */
    public O call(I inputData, Map<String, Object> params) {
        Object x = preprocess(inputData);
        Object y = forward(x);
        return postprocess(y);
    }

    /** Batch processing helper; subclasses may override with proper batching. */
    public List<O> callBatch(List<I> inputs) {
        java.util.List<O> out = new java.util.ArrayList<>(inputs.size());
        for (I in : inputs) out.add(call(in));
        return out;
    }

    protected abstract Object preprocess(I raw);

    protected abstract Object forward(Object x);

    protected abstract O postprocess(Object y);

    public boolean isClosed() { return closed; }

    @Override public void close() {
        if (closed) return;
        closed = true;
        onClose();
    }

    protected void onClose() {}
}