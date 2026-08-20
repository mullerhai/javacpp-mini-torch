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
package org.bytedeco.pytorch.llm.transformers.evaluation;

import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Accuracy;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Bleu;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Cer;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.CharF;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.CocoEvaluator;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.F1;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Mauve;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.MeanIoU;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Mcc;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Meteor;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Metric;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.PanopticQuality;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Pearson;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Perplexity;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Precision;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Recall;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Rouge;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Sacrebleu;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.SequenceAccuracy;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Spearman;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Squad;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.TimeSeriesMetric;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.TokenizerAccuracy;
import org.bytedeco.pytorch.llm.transformers.evaluation.evaluate.Wer;

import java.io.IOException;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Factory for loading evaluation metrics by name.
 *
 * <p>Reference: HuggingFace {@code evaluate} library.
 *
 * <pre>{@code
 * Metric m = Evaluator.load("accuracy");
 * Map<String, Double> result = m.compute(predictions, references);
 * }</pre>
 */
public final class Evaluator {

    private Evaluator() {}

    /**
     * Load a metric by name.
     *
     * @param name one of: "accuracy", "precision", "recall", "f1", "mcc",
     *             "pearson", "spearman", "rouge", "bleu", "sacrebleu",
     *             "meteor", "perplexity", "wer", "cer", "chrf", "mauve",
     *             "mean_iou", "squad", "seq_accuracy"
     * @return the metric instance
     * @throws IOException if the name is not supported
     */
    public static Metric load(String name) throws IOException {
        return switch (name.toLowerCase(java.util.Locale.ROOT)) {
            case "accuracy"       -> new Accuracy();
            case "precision"      -> new Precision();
            case "recall"         -> new Recall();
            case "f1"            -> new F1();
            case "mcc"            -> new Mcc();
            case "pearson"        -> new Pearson();
            case "spearman"       -> new Spearman();
            case "rouge"          -> new Rouge();
            case "bleu"           -> new Bleu();
            case "sacrebleu"     -> new Sacrebleu();
            case "meteor"         -> new Meteor();
            case "perplexity"     -> new Perplexity();
            case "wer"            -> new Wer();
            case "cer"           -> new Cer();
            case "chrf"           -> new CharF();
            case "mauve"          -> new Mauve();
            case "mean_iou"       -> new MeanIoU();
            case "panoptic_quality", "pq" -> new PanopticQuality();
            case "coco"           -> new CocoEvaluator();
            case "squad"          -> new Squad();
            case "seq_accuracy", "sequence_accuracy" -> new SequenceAccuracy();
            case "tokenizer_accuracy" -> new TokenizerAccuracy();
            case "time_series"    -> new TimeSeriesMetric();
            default -> throw new IOException("Unsupported metric: " + name);
        };
    }
}
