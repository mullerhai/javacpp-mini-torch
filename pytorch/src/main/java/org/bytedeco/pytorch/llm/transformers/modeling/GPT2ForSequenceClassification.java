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
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

import java.util.Map;
import java.util.Objects;

/**
 * GPT-2 model with a sequence classification head.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GPT2ForSequenceClassification extends Module {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final PretrainedConfig config;
    private final Module gpt2;
    private final LinearImpl classifier;

    public GPT2ForSequenceClassification(PretrainedConfig config) {
        super("GPT2ForSequenceClassification");
        this.config = Objects.requireNonNull(config);
        this.gpt2 = null; // TODO: instantiate real GPT2Model
        int numLabels = (int) config.numLabels();
        if (numLabels <= 0) numLabels = 2;
        this.classifier = register_module("classifier",
                new LinearImpl(new LinearOptions(config.n_embd(), numLabels).bias(true)));
    }

    public static GPT2ForSequenceClassification fromConfig(PretrainedConfig config) {
        return new GPT2ForSequenceClassification(config);
    }

    public PretrainedConfig config() { return config; }

    @Override
    public Tensor forward(Tensor inputIds) {
        throw new UnsupportedOperationException("GPT-2 sequence classification forward not yet implemented");
    }

    public static Tensor forward(Module model, Tensor inputIds, Map<String, Object> kwargs) {
        return model.forward(inputIds);
    }
}