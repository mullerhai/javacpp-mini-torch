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
 * BERT model with a masked language-modeling head.
 * Outputs logits over the vocabulary for each masked position.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class BertForMaskedLM extends Module {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final PretrainedConfig config;
    private final Module bert;
    private final LinearImpl mlmHead;

    public BertForMaskedLM(PretrainedConfig config) {
        super("BertForMaskedLM");
        this.config = Objects.requireNonNull(config);
        this.bert = null; // TODO: instantiate real BertModel
        this.mlmHead = register_module("cls.predictions",
                new LinearImpl(new LinearOptions(config.hiddenSize(), config.vocabSize()).bias(false)));
    }

    public static BertForMaskedLM fromConfig(PretrainedConfig config) {
        return new BertForMaskedLM(config);
    }

    public PretrainedConfig config() { return config; }

    @Override
    public Tensor forward(Tensor inputIds) {
        throw new UnsupportedOperationException("BERT masked LM forward not yet implemented");
    }

    public static Tensor forward(Module model, Tensor inputIds, Map<String, Object> kwargs) {
        return model.forward(inputIds);
    }
}
