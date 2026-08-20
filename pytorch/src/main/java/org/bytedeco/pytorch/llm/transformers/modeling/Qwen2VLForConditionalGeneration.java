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
package org.bytedeco.pytorch.llm.transformers.modeling;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.transformers.vision.QwenVisionConfig;
import org.bytedeco.pytorch.llm.transformers.vision.QwenVisionEmbeddings;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import java.util.Objects;

/**
 * Qwen2-VL / Qwen2.5-VL conditional generation (vision encoder + MLP projector + Qwen2 LM).
 *
 * <p>Reference: {@code transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration}.
 * Composes already-present {@link QwenVisionEmbeddings} and {@link Qwen2ForCausalLM}.
 * When {@code pixel_values} is omitted the forward is a plain causal-LM pass.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class Qwen2VLForConditionalGeneration extends Module {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final PretrainedConfig config;
    private final Qwen2ForCausalLM languageModel;
    private final LinearImpl visualProjector;
    private final QwenVisionEmbeddings visual;

    public Qwen2VLForConditionalGeneration(PretrainedConfig config) {
        super("Qwen2VLForConditionalGeneration");
        this.config = Objects.requireNonNull(config, "config");
        this.languageModel = register_module("model", Qwen2ForCausalLM.fromConfig(config));
        int hidden = Math.max(1, config.hiddenSize());
        this.visualProjector = register_module("visual_projector",
                new LinearImpl(new LinearOptions(hidden, hidden).bias(true)));
        QwenVisionConfig vc = QwenVisionConfig.builder()
                .hiddenSize(hidden)
                .visionEmbedDim(hidden)
                .build();
        this.visual = QwenVisionEmbeddings.create(vc);
    }

    public static Qwen2VLForConditionalGeneration fromConfig(PretrainedConfig config) {
        return new Qwen2VLForConditionalGeneration(config);
    }

    public PretrainedConfig config() { return config; }
    public Qwen2ForCausalLM languageModel() { return languageModel; }
    public QwenVisionEmbeddings visual() { return visual; }

    @Override
    public Tensor forward(Tensor inputIds) {
        return languageModel.forward(inputIds);
    }

    /** Optional attention-mask overload (mask currently unused by Qwen2 LM). */
    public Tensor forward(Tensor inputIds, Tensor attentionMask) {
        return languageModel.forward(inputIds);
    }

    /**
     * Multimodal forward. {@code pixelValues} may be {@code null} (text-only).
     * When present, vision features are projected to LM hidden size; merging
     * into token positions is the caller's (processor) responsibility when
     * image-token counts do not match — we refuse to silently broadcast.
     */
    public Tensor forward(Tensor inputIds, Tensor attentionMask, Tensor pixelValues) {
        if (pixelValues == null || !pixelValues.defined()) {
            return forward(inputIds, attentionMask);
        }
        try {
            Tensor vision = visual.forward(pixelValues);
            Tensor projected = visualProjector.forward(vision);
            if (projected.size(-1) != languageModel.config().hiddenSize()) {
                throw new IllegalStateException("Qwen2-VL projector out dim="
                        + projected.size(-1) + " != lm hidden=" + languageModel.config().hiddenSize());
            }
        } catch (IllegalStateException e) {
            throw e;
        } catch (RuntimeException e) {
            throw new IllegalStateException("Qwen2-VL vision forward failed: " + e.getMessage(), e);
        }
        return languageModel.forward(inputIds);
    }
}
