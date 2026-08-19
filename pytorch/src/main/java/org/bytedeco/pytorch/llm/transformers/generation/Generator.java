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
package org.bytedeco.pytorch.llm.transformers.generation;

import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.IntConsumer;


import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.argmax;
import static org.bytedeco.pytorch.global.torch.full;
import static org.bytedeco.pytorch.global.torch.multinomial;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.tensor;
import static org.bytedeco.pytorch.global.torch.topk;

/**
 * Autoregressive token generator (prefill-every-step MVP; KV-cache later).
 *
 * <p>Works with any {@link Module} whose {@code forward(input_ids)} returns logits {@code [B,T,V]}.
 */
public final class Generator {

    private Generator() {}

    public static int[] generate(Module model, int[] promptIds, GenerationConfig gen, int maxContext) {
        return generate(model, promptIds, gen, maxContext, null);
    }

    /**
     * Generate with a {@link List} of pre-tokenized prompts (batched). Returns a list of
     * generated token sequences including the original prompt.
     */
    public static List<int[]> generateBatch(List<int[]> promptIdsList, Module model,
                                           GenerationConfig gen, int maxContext,
                                           IntConsumer onToken) {
        Objects.requireNonNull(promptIdsList, "promptIdsList");
        List<int[]> results = new ArrayList<>();
        for (int[] promptIds : promptIdsList) {
            results.add(generate(model, promptIds, gen, maxContext, onToken));
        }
        return results;
    }

    /**
     * Stream-style generation: invoke the callback for each generated token id.
     */
    public static int[] generateStreaming(Module model, int[] promptIds, GenerationConfig gen,
                                          int maxContext, IntConsumer onToken) {
        return generate(model, promptIds, gen, maxContext, onToken);
    }

    /**
     * Compute perplexity score over a tokenized sequence.
     *
     * <p>Perplexity = exp(average negative log-likelihood) over all positions.
     */
    public static double computePerplexity(Module model, int[] tokens, int stride) {
        if (tokens == null || tokens.length == 0) return Double.NaN;
        if (stride <= 0) stride = tokens.length;
        double totalNll = 0.0;
        long totalCount = 0;
        try {
            for (int end = stride; end <= tokens.length; end += stride) {
                int start = Math.max(0, end - stride);
                int[] chunk = new int[end - start];
                System.arraycopy(tokens, start, chunk, 0, chunk.length);
                long[][] inputB = new long[1][Math.max(0, chunk.length - 1)];
                long[][] labelB = new long[1][Math.max(0, chunk.length - 1)];
                for (int i = 0; i < chunk.length - 1; i++) {
                    inputB[0][i] = chunk[i];
                    labelB[0][i] = chunk[i + 1];
                }
                if (inputB[0].length == 0) continue;
                Tensor input = tensor(inputB);
                Tensor output = model.forward(input);
                long V = output.size(output.dim() - 1);
                Tensor reshaped = output.view(-1, V);
                Tensor labelT = tensor(labelB).view(-1);
                Tensor loss = org.bytedeco.pytorch.global.torch.cross_entropy_loss(reshaped, labelT);
                totalNll += loss.item_double() * inputB[0].length;
                totalCount += inputB[0].length;
                input.close();
                output.close();
                reshaped.close();
                labelT.close();
                loss.close();
            }
        } catch (Exception e) {
            return Double.NaN;
        }
        return totalCount > 0 ? Math.exp(totalNll / totalCount) : Double.NaN;
    }

    /**
     * Compute next-token loss over a sequence (input = tokens[0..N-2], labels = tokens[1..N-1]).
     */
    public static double computeLoss(Module model, int[] tokens) {
        if (tokens == null || tokens.length < 2) return 0.0;
        long[][] inputB = new long[1][tokens.length - 1];
        long[][] labelB = new long[1][tokens.length - 1];
        for (int i = 0; i < tokens.length - 1; i++) {
            inputB[0][i] = tokens[i];
            labelB[0][i] = tokens[i + 1];
        }
        try {
            Tensor input = tensor(inputB);
            Tensor output = model.forward(input);
            long V = output.size(output.dim() - 1);
            Tensor reshaped = output.view(-1, V);
            Tensor labelT = tensor(labelB).view(-1);
            Tensor loss = org.bytedeco.pytorch.global.torch.cross_entropy_loss(reshaped, labelT);
            double v = loss.item_double();
            input.close();
            output.close();
            reshaped.close();
            labelT.close();
            loss.close();
            return v;
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    public static int[] generate(Module model, int[] promptIds, GenerationConfig gen,
                                 int maxContext, IntConsumer onToken) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(promptIds, "promptIds");
        if (gen == null) gen = GenerationConfig.greedy();
        if (maxContext <= 0) maxContext = 2048;

        List<Integer> seq = new ArrayList<>(promptIds.length + gen.maxNewTokens);
        for (int id : promptIds) seq.add(id);

        Set<Integer> eos = new HashSet<>(gen.eosTokenIds);
        boolean wasTraining = model.is_training();
        model.eval();
        try {
            for (int step = 0; step < gen.maxNewTokens; step++) {
                int start = Math.max(0, seq.size() - maxContext);
                long[] cur = new long[seq.size() - start];
                for (int i = 0; i < cur.length; i++) cur[i] = seq.get(start + i);
                Tensor ids = tensor(cur).unsqueeze(0); // [1, T]
                Tensor logits = model.forward(ids);    // [1, T, V]
                Tensor last = logits
                        .slice(1, new LongOptional(logits.size(1) - 1), new LongOptional(logits.size(1)), 1)
                        .squeeze(0).squeeze(0); // [V]

                last = applyRepetitionPenalty(last, seq, gen.repetitionPenalty);
                if (gen.temperature > 0 && Math.abs(gen.temperature - 1.0) > 1e-6) {
                    last = last.div(new Scalar(gen.temperature));
                }

                int next;
                if (gen.doSample && gen.temperature > 0) {
                    if (gen.topK > 0) last = topKFilter(last, gen.topK);
                    if (gen.topP > 0 && gen.topP < 1.0) last = topPFilter(last, gen.topP);
                    Tensor probs = softmax(last, 0L);
                    next = (int) multinomial(probs, 1L).item_long();
                } else {
                    next = (int) argmax(last).item_long();
                }

                seq.add(next);
                if (onToken != null) onToken.accept(next);
                if (gen.eosStop && eos.contains(next)) break;
            }
        } finally {
            if (wasTraining) model.train(true);
        }

        int[] out = new int[seq.size()];
        for (int i = 0; i < seq.size(); i++) out[i] = seq.get(i);
        return out;
    }

    private static Tensor applyRepetitionPenalty(Tensor logits, List<Integer> seq, double penalty) {
        if (penalty <= 0 || Math.abs(penalty - 1.0) < 1e-6) return logits;
        Tensor out = logits.clone();
        Set<Integer> seen = new HashSet<>(seq);
        if (seen.isEmpty()) return out;
        for (int id : seen) {
            if (id < 0 || id >= (int) out.size(0)) continue;
            try {
                Tensor t = out.select(0, id);
                float v = t.item_float();
                float nv = v > 0 ? (float) (v / penalty) : (float) (v * penalty);
                t.fill_(new Scalar(nv));
            } catch (Throwable ignored) {
                // ignore if select/fill unavailable on this build
            }
        }
        return out;
    }

    private static Tensor topKFilter(Tensor logits, int k) {
        long V = logits.size(0);
        if (k <= 0 || k >= V) return logits;
        var top = topk(logits, k);
        Tensor values = top.get0();
        float threshold = values
                .slice(0, new LongOptional(values.size(0) - 1), new LongOptional(values.size(0)), 1)
                .squeeze()
                .item_float();
        Tensor negInf = full(new long[]{V}, new Scalar(-1e9f));
        Tensor mask = logits.gt(new Scalar(threshold - 1e-6f)).to(ScalarType.Float);
        Tensor ones = full(new long[]{V}, new Scalar(1.0f));
        return logits.mul(mask).add(negInf.mul(ones.sub(mask)));
    }

    /** Nucleus (top-p) filtering on 1-D logits. */
    private static Tensor topPFilter(Tensor logits, double topP) {
        long V = logits.size(0);
        if (topP <= 0 || topP >= 1.0) return logits;
        Tensor probs = softmax(logits, 0L);
        var sorted = topk(probs, (int) V); // descending
        Tensor sortedProbs = sorted.get0();
        Tensor sortedIdx = sorted.get1();
        // cumulative
        float cum = 0f;
        int cutoff = (int) V;
        for (int i = 0; i < V; i++) {
            cum += sortedProbs.select(0, i).item_float();
            if (cum >= topP) {
                cutoff = i + 1;
                break;
            }
        }
        Tensor negInf = full(new long[]{V}, new Scalar(-1e9f));
        Tensor out = negInf.clone();
        for (int i = 0; i < cutoff; i++) {
            int idx = (int) sortedIdx.select(0, i).item_long();
            try {
                out.select(0, idx).copy_(logits.select(0, idx));
            } catch (Throwable ignored) {}
        }
        return out;
    }
}
