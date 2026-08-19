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

import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * HuggingFace-style generation configuration.
 */
public final class GenerationConfig {

    public final boolean doSample;
    public final double temperature;
    public final int topK;
    public final double topP;
    public final double repetitionPenalty;
    public final int maxNewTokens;
    public final boolean eosStop;
    public final List<Integer> eosTokenIds;
    public final Integer padTokenId;
    public final Integer bosTokenId;
    // -------- Extended generation parameters (HF GenerationConfig parity) --------
    public final int numBeams;
    public final int numReturnSequences;
    public final double lengthPenalty;
    public final boolean earlyStopping;
    public final int noRepeatNgramSize;
    public final List<List<Integer>> badWordsIds;
    public final List<List<Integer>> forceWordsIds;
    public final boolean renormalizeLogits;
    public final boolean useCache;
    public final double minP;
    public final double typicalP;
    public final double epsilon;
    public final double eta;
    public final int encoderNoRepeatNgramSize;

    private GenerationConfig(Builder b) {
        this.doSample = b.doSample;
        this.temperature = b.temperature;
        this.topK = b.topK;
        this.topP = b.topP;
        this.repetitionPenalty = b.repetitionPenalty;
        this.maxNewTokens = b.maxNewTokens;
        this.eosStop = b.eosStop;
        this.eosTokenIds = Collections.unmodifiableList(new ArrayList<>(b.eosTokenIds));
        this.padTokenId = b.padTokenId;
        this.bosTokenId = b.bosTokenId;
        this.numBeams = b.numBeams;
        this.numReturnSequences = b.numReturnSequences;
        this.lengthPenalty = b.lengthPenalty;
        this.earlyStopping = b.earlyStopping;
        this.noRepeatNgramSize = b.noRepeatNgramSize;
        this.badWordsIds = b.badWordsIds == null ? null
                : Collections.unmodifiableList(new ArrayList<>(b.badWordsIds));
        this.forceWordsIds = b.forceWordsIds == null ? null
                : Collections.unmodifiableList(new ArrayList<>(b.forceWordsIds));
        this.renormalizeLogits = b.renormalizeLogits;
        this.useCache = b.useCache;
        this.minP = b.minP;
        this.typicalP = b.typicalP;
        this.epsilon = b.epsilon;
        this.eta = b.eta;
        this.encoderNoRepeatNgramSize = b.encoderNoRepeatNgramSize;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static GenerationConfig greedy() {
        return builder().doSample(false).temperature(1.0).maxNewTokens(64).build();
    }

    public static GenerationConfig sample(double temperature, int topK) {
        return builder().doSample(true).temperature(temperature).topK(topK).maxNewTokens(64).build();
    }

    public static GenerationConfig fromJson(String json) throws IOException {
        if (json == null || json.isBlank()) return greedy();
        Map<String, Object> m = Json.decodeObject(json);
        return fromMap(m);
    }

    public static GenerationConfig fromFile(Path path) throws IOException {
        return fromJson(Files.readString(path, StandardCharsets.UTF_8));
    }

    public static GenerationConfig fromMap(Map<String, Object> m) {
        Builder b = builder();
        if (m.containsKey("do_sample")) b.doSample(asBool(m.get("do_sample")));
        if (m.containsKey("temperature")) b.temperature(asDouble(m.get("temperature")));
        if (m.containsKey("top_k")) b.topK(asInt(m.get("top_k")));
        if (m.containsKey("top_p")) b.topP(asDouble(m.get("top_p")));
        if (m.containsKey("repetition_penalty")) b.repetitionPenalty(asDouble(m.get("repetition_penalty")));
        if (m.containsKey("max_new_tokens")) b.maxNewTokens(asInt(m.get("max_new_tokens")));
        else if (m.containsKey("max_length")) b.maxNewTokens(asInt(m.get("max_length")));
        if (m.containsKey("pad_token_id") && m.get("pad_token_id") != null) b.padTokenId(asInt(m.get("pad_token_id")));
        if (m.containsKey("bos_token_id") && m.get("bos_token_id") != null) b.bosTokenId(asInt(m.get("bos_token_id")));
        if (m.containsKey("num_beams")) b.numBeams(asInt(m.get("num_beams")));
        if (m.containsKey("num_return_sequences")) b.numReturnSequences(asInt(m.get("num_return_sequences")));
        if (m.containsKey("length_penalty")) b.lengthPenalty(asDouble(m.get("length_penalty")));
        if (m.containsKey("early_stopping")) b.earlyStopping(asBool(m.get("early_stopping")));
        if (m.containsKey("no_repeat_ngram_size")) b.noRepeatNgramSize(asInt(m.get("no_repeat_ngram_size")));
        if (m.containsKey("encoder_no_repeat_ngram_size")) b.encoderNoRepeatNgramSize(asInt(m.get("encoder_no_repeat_ngram_size")));
        if (m.containsKey("renormalize_logits")) b.renormalizeLogits(asBool(m.get("renormalize_logits")));
        if (m.containsKey("use_cache")) b.useCache(asBool(m.get("use_cache")));
        if (m.containsKey("min_p")) b.minP(asDouble(m.get("min_p")));
        if (m.containsKey("typical_p")) b.typicalP(asDouble(m.get("typical_p")));
        if (m.containsKey("epsilon")) b.epsilon(asDouble(m.get("epsilon")));
        if (m.containsKey("eta")) b.eta(asDouble(m.get("eta")));
        Object eos = m.get("eos_token_id");
        if (eos instanceof Number) {
            b.eosTokenId(asInt(eos));
        } else if (eos instanceof List<?> list) {
            for (Object o : list) b.eosTokenId(asInt(o));
        }
        return b.build();
    }

    public String toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("do_sample", doSample);
        m.put("temperature", temperature);
        m.put("top_k", topK);
        m.put("top_p", topP);
        m.put("repetition_penalty", repetitionPenalty);
        m.put("max_new_tokens", maxNewTokens);
        if (numBeams > 1) m.put("num_beams", numBeams);
        if (numReturnSequences > 1) m.put("num_return_sequences", numReturnSequences);
        if (lengthPenalty != 1.0) m.put("length_penalty", lengthPenalty);
        if (earlyStopping) m.put("early_stopping", true);
        if (noRepeatNgramSize > 0) m.put("no_repeat_ngram_size", noRepeatNgramSize);
        if (renormalizeLogits) m.put("renormalize_logits", true);
        if (useCache) m.put("use_cache", true);
        if (minP > 0) m.put("min_p", minP);
        if (typicalP > 0) m.put("typical_p", typicalP);
        if (epsilon > 0) m.put("epsilon", epsilon);
        if (eta > 0) m.put("eta", eta);
        if (padTokenId != null) m.put("pad_token_id", padTokenId);
        if (bosTokenId != null) m.put("bos_token_id", bosTokenId);
        if (eosTokenIds.size() == 1) m.put("eos_token_id", eosTokenIds.get(0));
        else if (!eosTokenIds.isEmpty()) m.put("eos_token_id", eosTokenIds);
        return Json.encode(m);
    }

    public Builder toBuilder() {
        Builder b = builder()
                .doSample(doSample).temperature(temperature).topK(topK).topP(topP)
                .repetitionPenalty(repetitionPenalty).maxNewTokens(maxNewTokens).eosStop(eosStop)
                .numBeams(numBeams).numReturnSequences(numReturnSequences).lengthPenalty(lengthPenalty)
                .earlyStopping(earlyStopping).noRepeatNgramSize(noRepeatNgramSize)
                .renormalizeLogits(renormalizeLogits).useCache(useCache)
                .minP(minP).typicalP(typicalP).epsilon(epsilon).eta(eta)
                .encoderNoRepeatNgramSize(encoderNoRepeatNgramSize);
        for (int id : eosTokenIds) b.eosTokenId(id);
        if (padTokenId != null) b.padTokenId(padTokenId);
        if (bosTokenId != null) b.bosTokenId(bosTokenId);
        return b;
    }

    private static int asInt(Object o) {
        if (o instanceof Number n) return n.intValue();
        return Integer.parseInt(String.valueOf(o));
    }

    private static double asDouble(Object o) {
        if (o instanceof Number n) return n.doubleValue();
        return Double.parseDouble(String.valueOf(o));
    }

    private static boolean asBool(Object o) {
        if (o instanceof Boolean b) return b;
        return Boolean.parseBoolean(String.valueOf(o));
    }

    public static final class Builder {
        private boolean doSample;
        private double temperature = 1.0;
        private int topK;
        private double topP = 1.0;
        private double repetitionPenalty = 1.0;
        private int maxNewTokens = 64;
        private boolean eosStop = true;
        private final List<Integer> eosTokenIds = new ArrayList<>();
        private Integer padTokenId;
        private Integer bosTokenId;
        // Extended fields
        private int numBeams = 1;
        private int numReturnSequences = 1;
        private double lengthPenalty = 1.0;
        private boolean earlyStopping = false;
        private int noRepeatNgramSize = 0;
        private List<List<Integer>> badWordsIds = null;
        private List<List<Integer>> forceWordsIds = null;
        private boolean renormalizeLogits = false;
        private boolean useCache = true;
        private double minP = 0.0;
        private double typicalP = 0.0;
        private double epsilon = 0.0;
        private double eta = 0.0;
        private int encoderNoRepeatNgramSize = 0;

        public Builder doSample(boolean v) { this.doSample = v; return this; }
        public Builder temperature(double v) { this.temperature = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder topP(double v) { this.topP = v; return this; }
        public Builder repetitionPenalty(double v) { this.repetitionPenalty = v; return this; }
        public Builder maxNewTokens(int v) { this.maxNewTokens = v; return this; }
        public Builder eosStop(boolean v) { this.eosStop = v; return this; }
        public Builder eosTokenId(int id) { this.eosTokenIds.add(id); return this; }
        public Builder eosTokenIds(List<Integer> ids) {
            this.eosTokenIds.clear();
            if (ids != null) this.eosTokenIds.addAll(ids);
            return this;
        }
        public Builder padTokenId(int id) { this.padTokenId = id; return this; }
        public Builder bosTokenId(int id) { this.bosTokenId = id; return this; }
        public Builder numBeams(int v) { this.numBeams = v; return this; }
        public Builder numReturnSequences(int v) { this.numReturnSequences = v; return this; }
        public Builder lengthPenalty(double v) { this.lengthPenalty = v; return this; }
        public Builder earlyStopping(boolean v) { this.earlyStopping = v; return this; }
        public Builder noRepeatNgramSize(int v) { this.noRepeatNgramSize = v; return this; }
        public Builder badWordsIds(List<List<Integer>> v) { this.badWordsIds = v; return this; }
        public Builder forceWordsIds(List<List<Integer>> v) { this.forceWordsIds = v; return this; }
        public Builder renormalizeLogits(boolean v) { this.renormalizeLogits = v; return this; }
        public Builder useCache(boolean v) { this.useCache = v; return this; }
        public Builder minP(double v) { this.minP = v; return this; }
        public Builder typicalP(double v) { this.typicalP = v; return this; }
        public Builder epsilon(double v) { this.epsilon = v; return this; }
        public Builder eta(double v) { this.eta = v; return this; }
        public Builder encoderNoRepeatNgramSize(int v) { this.encoderNoRepeatNgramSize = v; return this; }

        public GenerationConfig build() {
            return new GenerationConfig(this);
        }
    }
}
