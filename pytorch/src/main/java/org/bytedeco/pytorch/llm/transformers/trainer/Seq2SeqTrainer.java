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
package org.bytedeco.pytorch.llm.transformers.trainer;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.datasets.HfDataset;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Sequence-to-sequence trainer that extends {@link Trainer} with
 * generation-based prediction via {@link #predictWithGenerate}.
 *
 * <p>Mirrors HF's {@code Seq2SeqTrainer}. When {@link #predictWithGenerate}
 * is {@code true}, the predict loop uses token generation (e.g. beam search)
 * instead of a plain forward pass.
 */
public class Seq2SeqTrainer extends Trainer {

    /** If true, call {@code Generator.generate()} during prediction instead of plain forward. */
    public boolean predictWithGenerate = false;

    public Seq2SeqTrainer(Module model, TrainingArguments args, HfDataset train,
                          HfDataset eval, FastTokenizer tokenizer,
                          List<TrainerCallback> callbacks,
                          ComputeMetrics computeMetrics) {
        super(model, args, train, eval, tokenizer, callbacks, computeMetrics);
    }

    public Seq2SeqTrainer(Module model, TrainingArguments args, HfDataset train,
                          HfDataset eval, FastTokenizer tokenizer,
                          List<TrainerCallback> callbacks) {
        super(model, args, train, eval, tokenizer, callbacks, null);
    }

    public Seq2SeqTrainer(Module model, TrainingArguments args, HfDataset train,
                          HfDataset eval, FastTokenizer tokenizer) {
        super(model, args, train, eval, tokenizer, List.of(), null);
    }

    /**
     * Override predict to use generation when {@link #predictWithGenerate} is true.
     */
    @Override
    public TrainerPredictionOutput predict(HfDataset dataset) {
        if (!state.isLocalProcessZero()) return null;

        if (predictWithGenerate) {
            System.out.println("*** Seq2SeqTrainer.predict — using predict_with_generate ***");
            // Stub: real implementation would:
            //   1. Encode each input text with the tokenizer
            //   2. Call Generator.generate(model, input_ids, genConfig, maxLen)
            //   3. Decode generated token ids
            //   4. Return TrainerPredictionOutput with decoded strings / metrics
            throw new UnsupportedOperationException(
                    "Seq2SeqTrainer.predict with predictWithGenerate=true is a stub — " +
                    "implement Generator.generate integration for encoder-decoder models.");
        }
        return super.predict(dataset);
    }

    /**
     * Run evaluation using generation (beam search / greedy).
     */
    public Map<String, Double> evaluate_with_generate() {
        predictWithGenerate = true;
        try {
            return evaluate();
        } finally {
            predictWithGenerate = false;
        }
    }
}
