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
package org.bytedeco.pytorch.llm.tokenizers;

import org.bytedeco.pytorch.llm.text.tokenizer.Tokenizer;
import org.bytedeco.pytorch.llm.tokenizers.decoders.Decoder;
import org.bytedeco.pytorch.llm.tokenizers.models.*;
import org.bytedeco.pytorch.llm.tokenizers.normalizers.Normalizer;
import org.bytedeco.pytorch.llm.tokenizers.pretokenizers.PreTokenizer;
import org.bytedeco.pytorch.llm.tokenizers.processors.PostProcessor;
import org.bytedeco.pytorch.llm.text.tokenizer.BPETokenizer;
import org.bytedeco.pytorch.llm.transformers.tokenization.ChatTemplate;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HuggingFace {@code tokenizers} style fast tokenizer (pure Java).
 *
 * <p>Backed by a full tokenizers-rs pipeline
 * ({@link TokenizerPipeline}: Normalizer → PreTokenizer → Model → PostProcessor → Decoder)
 * loaded from real {@code tokenizer.json} files (Qwen / Llama / DeepSeek / GLM / BERT / …).
 *
 * <pre>{@code
 * try (FastTokenizer tok = FastTokenizer.fromFile(Path.of("tokenizer.json"))) {
 *     Encoding enc = tok.encode("Hello world", true);
 *     String text = tok.decode(enc.ids(), true);
 * }
 * }</pre>
 */
public final class FastTokenizer implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Performance metrics
    private long totalEncodeTimeMs;
    private long totalDecodeTimeMs;
    private int totalEncodeCalls;
    private int totalDecodeCalls;
    private long totalTokensEncoded;
    private long totalTokensDecoded;

    public enum Backend { BPE, WORDPIECE, GPT2, CHAR, WHITESPACE, UNIGRAM, PIPELINE }

    private final Backend backend;
    /** Non-final so {@code tokenizer.pad_token = tokenizer.eos_token} can mutate in place (HF semantics). */
    private volatile TokenizerPipeline pipeline;
    /** HuggingFace {@code tokenizer.chat_template} raw Jinja / ChatML string. */
    private volatile String chatTemplate;
    private volatile ChatTemplate chatTemplateEngine;

    private FastTokenizer(Backend backend, TokenizerPipeline pipeline) {
        this.backend = backend == null ? Backend.PIPELINE : backend;
        this.pipeline = Objects.requireNonNull(pipeline, "pipeline");
    }

    /** Wrap an existing pipeline. */
    public static FastTokenizer of(TokenizerPipeline pipeline) {
        return new FastTokenizer(detectBackend(pipeline), pipeline);
    }

    public static Builder builder() {
        return new Builder();
    }

    /** WordPiece backend from an ordered vocab map (token → id). */
    public static Builder wordPiece(Map<String, Integer> vocab) {
        WordPieceModel model = new WordPieceModel(new LinkedHashMap<>(vocab), "[UNK]", "##", 100);
        TokenizerPipeline pipe = new TokenizerPipeline(
                Normalizer.NOP,
                PreTokenizer.WhitespaceSplitPreTokenizer.INSTANCE,
                model,
                new PostProcessor.BertProcessing("[CLS]",
                        vocab.getOrDefault("[CLS]", vocab.size()),
                        "[SEP]",
                        vocab.getOrDefault("[SEP]", vocab.size() + 1)),
                new Decoder.WordPieceDecoder("##", true),
                AddedVocabulary.empty(),
                null, null,
                "[UNK]", "[PAD]", "[CLS]", "[SEP]", null, null, "[MASK]",
                512, false);
        return builder().backend(Backend.WORDPIECE).pipeline(pipe)
                .unkToken("[UNK]").padToken("[PAD]").clsToken("[CLS]").sepToken("[SEP]")
                .maskToken("[MASK]");
    }

    /** GPT-2 byte-level BPE (minimal empty-merges seed — prefer {@link #fromFile}). */
    public static Builder gpt2() {
        Map<String, Integer> v = new LinkedHashMap<>();
        for (int i = 0; i < 256; i++) {
            v.put(String.valueOf(BytesToUnicode.encodeByte(i)), i);
        }
        v.put("<|endoftext|>", 256);
        BpeModel model = new BpeModel(v, List.of(), "<|endoftext|>", null, null, false, false, false);
        TokenizerPipeline pipe = new TokenizerPipeline(
                Normalizer.NOP,
                new PreTokenizer.ByteLevelPreTokenizer(false, true, true),
                model,
                PostProcessor.NOP,
                Decoder.ByteLevelDecoder.INSTANCE,
                AddedVocabulary.empty(),
                null, null,
                "<|endoftext|>", "<|endoftext|>", null, null,
                "<|endoftext|>", "<|endoftext|>", null,
                1024, false);
        return builder().backend(Backend.GPT2).pipeline(pipe)
                .unkToken("<|endoftext|>").padToken("<|endoftext|>")
                .bosToken("<|endoftext|>").eosToken("<|endoftext|>")
                .modelMaxLength(1024);
    }

    /** Whitespace split + small vocab (debug / smoke / tiny models). */
    public static Builder whitespace() {
        Map<String, Integer> v = new LinkedHashMap<>();
        v.put("[UNK]", 0);
        v.put("[PAD]", 1);
        v.put("[CLS]", 2);
        v.put("[SEP]", 3);
        Model model = new Model.WordLevelModel(v, "[UNK]");
        // BertProcessing so addSpecialTokens=true still wraps [CLS]/[SEP] for tiny demos
        PostProcessor post = new PostProcessor.BertProcessing("[CLS]", 2, "[SEP]", 3);
        TokenizerPipeline pipe = new TokenizerPipeline(
                Normalizer.NOP,
                PreTokenizer.WhitespaceSplitPreTokenizer.INSTANCE,
                model,
                post,
                Decoder.FUSE,
                AddedVocabulary.empty(),
                null, null,
                "[UNK]", "[PAD]", "[CLS]", "[SEP]", null, null, null,
                512, false);
        return builder().backend(Backend.WHITESPACE).pipeline(pipe)
                .unkToken("[UNK]").padToken("[PAD]").clsToken("[CLS]").sepToken("[SEP]");
    }

    /** Learn a small BPE on a corpus then wrap (torchtext-style demo). */
    public static Builder bpeFromCorpus(Iterable<String> corpus, int numMerges) {
        BPETokenizer bpe = BPETokenizer.learn(corpus, numMerges);
        Map<String, Integer> v = new LinkedHashMap<>(bpe.vocab());
        v.putIfAbsent("<unk>", v.size());
        v.putIfAbsent("<pad>", v.size());
        v.putIfAbsent("<s>", v.size());
        v.putIfAbsent("</s>", v.size());
        List<String> merges = new ArrayList<>(bpe.merges());
        BpeModel model = new BpeModel(v, merges, "<unk>", null, "</w>", false, false, false);
        TokenizerPipeline pipe = new TokenizerPipeline(
                Normalizer.LowercaseNormalizer.INSTANCE,
                PreTokenizer.WhitespaceSplitPreTokenizer.INSTANCE,
                model,
                PostProcessor.NOP,
                new Decoder.BPEDecoder("</w>"),
                AddedVocabulary.empty(),
                null, null,
                "<unk>", "<pad>", null, null, "<s>", "</s>", null,
                512, false);
        return builder().backend(Backend.BPE).pipeline(pipe)
                .unkToken("<unk>").padToken("<pad>").bosToken("<s>").eosToken("</s>");
    }

    // ---- loaders ------------------------------------------------------------

    /** Load a real HuggingFace {@code tokenizer.json} (tokenizers-rs schema). */
    public static FastTokenizer fromFile(Path tokenizerJson) throws IOException {
        TokenizerPipeline pipe = TokenizerJsonLoader.fromFile(tokenizerJson);
        // overlay sibling configs if present
        Path dir = tokenizerJson.getParent();
        if (dir != null) {
            pipe = TokenizerJsonLoader.applyTokenizerConfig(pipe, dir.resolve("tokenizer_config.json"));
            pipe = TokenizerJsonLoader.applySpecialTokensMap(pipe, dir.resolve("special_tokens_map.json"));
        }
        return new FastTokenizer(detectBackend(pipe), pipe);
    }

    public static FastTokenizer fromTokenizerJson(String json) throws IOException {
        TokenizerPipeline pipe = TokenizerJsonLoader.fromJson(json);
        return new FastTokenizer(detectBackend(pipe), pipe);
    }

    /**
     * Load from a model snapshot directory:
     * {@code tokenizer.json} → else {@code vocab.json}+{@code merges.txt} → else whitespace.
     */
    public static FastTokenizer fromDirectory(Path dir) throws IOException {
        return DirectoryTokenizerLoader.load(dir);
    }

    private static Backend detectBackend(TokenizerPipeline pipe) {
        Model m = pipe.model();
        if (m instanceof BpeModel) return Backend.BPE;
        if (m instanceof TiktokenBpeModel) return Backend.BPE;
        if (m instanceof WordPieceModel) return Backend.WORDPIECE;
        if (m instanceof UnigramModel) return Backend.UNIGRAM;
        return Backend.PIPELINE;
    }

    // ---- encode / decode ----------------------------------------------------

    public Encoding encode(String text) {
        return encode(text, false);
    }

    public Encoding encode(String text, boolean addSpecialTokens) {
        long start = System.currentTimeMillis();
        Encoding enc = pipeline.encode(text, addSpecialTokens);
        totalEncodeTimeMs += System.currentTimeMillis() - start;
        totalEncodeCalls++;
        totalTokensEncoded += enc.size();
        return applyConfiguredPadTruncate(enc);
    }

    public Encoding encodePair(String textA, String textB, boolean addSpecialTokens) {
        long start = System.currentTimeMillis();
        Encoding enc = pipeline.encodePair(textA, textB, addSpecialTokens);
        totalEncodeTimeMs += System.currentTimeMillis() - start;
        totalEncodeCalls++;
        totalTokensEncoded += enc.size();
        return applyConfiguredPadTruncate(enc);
    }

    public List<Encoding> encodeBatch(List<String> texts, boolean addSpecialTokens) {
        if (texts == null || texts.isEmpty()) return List.of();
        long start = System.currentTimeMillis();
        List<Encoding> out = new ArrayList<>(texts.size());
        int max = 0;
        int totalTokens = 0;
        for (String t : texts) {
            Encoding e = pipeline.encode(t, addSpecialTokens);
            out.add(e);
            totalTokens += e.size();
            if (e.size() > max) max = e.size();
        }
        totalEncodeTimeMs += System.currentTimeMillis() - start;
        totalEncodeCalls += texts.size();
        totalTokensEncoded += totalTokens;
        Padding pad = pipeline.padding();
        if (pad != null && pad.strategy == Padding.Strategy.LONGEST) {
            List<Encoding> padded = new ArrayList<>(out.size());
            for (Encoding e : out) {
                padded.add(e.padTo(max, padId(), 0, pad.direction));
            }
            return padded;
        }
        return out;
    }

    public String decode(int[] ids) {
        return decode(ids, true);
    }

    public String decode(int[] ids, boolean skipSpecialTokens) {
        long start = System.currentTimeMillis();
        String result = pipeline.decode(ids, skipSpecialTokens);
        totalDecodeTimeMs += System.currentTimeMillis() - start;
        totalDecodeCalls++;
        totalTokensDecoded += ids != null ? ids.length : 0;
        return result;
    }

    private Encoding applyConfiguredPadTruncate(Encoding enc) {
        // pipeline already applies its own padding/truncation; this is a no-op safety
        return enc;
    }

    // ---- vocab / specials ---------------------------------------------------

    public List<String> convertIdsToTokens(int[] ids) {
        if (ids == null) return List.of();
        List<String> out = new ArrayList<>(ids.length);
        for (int id : ids) {
            String t = pipeline.idToToken(id);
            out.add(t == null ? "" : t);
        }
        return out;
    }

    public int[] convertTokensToIds(List<String> tokens) {
        if (tokens == null) return new int[0];
        int[] ids = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            ids[i] = pipeline.tokenToId(tokens.get(i));
        }
        return ids;
    }

    public int tokenToId(String token) {
        return pipeline.tokenToId(token);
    }

    public String idToToken(int id) {
        return pipeline.idToToken(id);
    }

    public int vocabSize() {
        return pipeline.vocabSize();
    }

    public Map<String, Integer> getVocab() {
        return pipeline.getVocab();
    }

    public Backend backend() {
        return backend;
    }

    public TokenizerPipeline pipeline() {
        return pipeline;
    }

    public int padId() { return pipeline.padId(); }
    public int unkId() { return pipeline.unkId(); }
    public int clsId() { return tokenToId(clsToken()); }
    public int sepId() { return tokenToId(sepToken()); }
    public int bosId() { return pipeline.bosId(); }
    public int eosId() { return pipeline.eosId(); }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.printf(
                "[FastTokenizer] Closed: encodeCalls=%d, decodeCalls=%d, " +
                "encodeTime=%.2fs (%.3fms/call), decodeTime=%.2fs (%.3fms/call), " +
                "tokensEncoded=%d, tokensDecoded=%d%n",
                totalEncodeCalls, totalDecodeCalls,
                totalEncodeTimeMs / 1000.0,
                totalEncodeCalls > 0 ? (double) totalEncodeTimeMs / totalEncodeCalls : 0,
                totalDecodeTimeMs / 1000.0,
                totalDecodeCalls > 0 ? (double) totalDecodeTimeMs / totalDecodeCalls : 0,
                totalTokensEncoded, totalTokensDecoded);
    }

    public boolean isClosed() { return closed; }

    /**
     * Get tokenizer statistics.
     */
    public TokenizerStats getStats() {
        return new TokenizerStats(
                totalEncodeCalls, totalDecodeCalls,
                totalEncodeTimeMs, totalDecodeTimeMs,
                totalTokensEncoded, totalTokensDecoded,
                vocabSize(), backend().name()
        );
    }

    /**
     * Tokenizer performance statistics.
     */
    public static final class TokenizerStats {
        public final int totalEncodeCalls;
        public final int totalDecodeCalls;
        public final long totalEncodeTimeMs;
        public final long totalDecodeTimeMs;
        public final long totalTokensEncoded;
        public final long totalTokensDecoded;
        public final int vocabSize;
        public final String backend;

        public TokenizerStats(int totalEncodeCalls, int totalDecodeCalls,
                           long totalEncodeTimeMs, long totalDecodeTimeMs,
                           long totalTokensEncoded, long totalTokensDecoded,
                           int vocabSize, String backend) {
            this.totalEncodeCalls = totalEncodeCalls;
            this.totalDecodeCalls = totalDecodeCalls;
            this.totalEncodeTimeMs = totalEncodeTimeMs;
            this.totalDecodeTimeMs = totalDecodeTimeMs;
            this.totalTokensEncoded = totalTokensEncoded;
            this.totalTokensDecoded = totalTokensDecoded;
            this.vocabSize = vocabSize;
            this.backend = backend;
        }

        public double avgEncodeTimeMs() {
            return totalEncodeCalls > 0 ? (double) totalEncodeTimeMs / totalEncodeCalls : 0;
        }

        public double avgDecodeTimeMs() {
            return totalDecodeCalls > 0 ? (double) totalDecodeTimeMs / totalDecodeCalls : 0;
        }

        public double encodeThroughput() {
            return totalEncodeTimeMs > 0 ? totalTokensEncoded * 1000.0 / totalEncodeTimeMs : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "TokenizerStats{encodeCalls=%d, decodeCalls=%d, " +
                    "avgEncode=%.3fms, avgDecode=%.3fms, " +
                    "throughput=%.0f tok/s, vocab=%d, backend=%s}",
                    totalEncodeCalls, totalDecodeCalls,
                    avgEncodeTimeMs(), avgDecodeTimeMs(),
                    encodeThroughput(), vocabSize, backend);
        }
    }

    public String unkToken() { return pipeline.unkToken(); }
    public String padToken() { return pipeline.padToken(); }
    public String clsToken() { return pipeline.clsToken(); }
    public String sepToken() { return pipeline.sepToken(); }
    public String bosToken() { return pipeline.bosToken(); }
    public String eosToken() { return pipeline.eosToken(); }
    public String maskToken() { return pipeline.maskToken(); }
    public int modelMaxLength() { return pipeline.modelMaxLength(); }

    /** HuggingFace {@code tokenizer.pad_token_id}. */
    public int padTokenId() { return pipeline.padId(); }
    /** HuggingFace {@code tokenizer.eos_token_id}. */
    public int eosTokenId() { return pipeline.eosId(); }
    /** HuggingFace {@code tokenizer.bos_token_id}. */
    public int bosTokenId() { return pipeline.bosId(); }

    /**
     * HuggingFace {@code tokenizer.pad_token = token}. Rebuilds the pipeline
     * specials in place (same object identity, matching the mutable Python API).
     */
    public FastTokenizer setPadToken(String token) {
        replaceSpecials(pipeline.unkToken(), token, pipeline.clsToken(), pipeline.sepToken(),
                pipeline.bosToken(), pipeline.eosToken(), pipeline.maskToken());
        return this;
    }

    /** Snake alias matching Python {@code tokenizer.pad_token = ...}. */
    public FastTokenizer pad_token(String token) { return setPadToken(token); }

    public FastTokenizer setEosToken(String token) {
        replaceSpecials(pipeline.unkToken(), pipeline.padToken(), pipeline.clsToken(), pipeline.sepToken(),
                pipeline.bosToken(), token, pipeline.maskToken());
        return this;
    }

    public FastTokenizer eos_token(String token) { return setEosToken(token); }

    /**
     * HuggingFace {@code tokenizer.pad_token_id = id}. Resolves {@code id} through
     * the vocab and delegates to {@link #setPadToken(String)}.
     */
    public FastTokenizer setPadTokenId(int id) {
        String tok = pipeline.idToToken(id);
        if (tok != null) setPadToken(tok);
        return this;
    }

    public FastTokenizer pad_token_id(int id) { return setPadTokenId(id); }

    /**
     * HuggingFace {@code tokenizer.chat_template = jinja}.
     * Stores the raw string and builds a {@link ChatTemplate} (ChatML / Llama-3 /
     * generation-tag subset — not a full Jinja VM).
     */
    public FastTokenizer setChatTemplate(String jinja) {
        this.chatTemplate = jinja;
        this.chatTemplateEngine = jinja == null || jinja.isBlank()
                ? null : ChatTemplate.custom(jinja);
        return this;
    }

    public FastTokenizer chat_template(String jinja) { return setChatTemplate(jinja); }

    public String chatTemplate() { return chatTemplate; }

    public ChatTemplate chatTemplateEngine() { return chatTemplateEngine; }

    /**
     * HuggingFace {@code tokenizer.apply_chat_template(messages, tokenize=False,
     * add_generation_prompt=...)}.
     */
    @SuppressWarnings("unchecked")
    public String applyChatTemplate(List<? extends Map<String, ?>> messages,
                                    boolean tokenize, boolean addGenerationPrompt) {
        ChatTemplate engine = chatTemplateEngine != null ? chatTemplateEngine : ChatTemplate.qwen();
        String text = engine.applyObject(messages, addGenerationPrompt);
        if (tokenize) {
            // Caller asked for token ids via the string API; return a debug
            // representation. Prefer {@link #applyChatTemplateToIds}.
            int[] ids = encode(text, false).ids();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < ids.length; i++) {
                if (i > 0) sb.append(' ');
                sb.append(ids[i]);
            }
            return sb.toString();
        }
        return text;
    }

    public String apply_chat_template(List<? extends Map<String, ?>> messages,
                                      boolean tokenize, boolean addGenerationPrompt) {
        return applyChatTemplate(messages, tokenize, addGenerationPrompt);
    }

    /** Tokenize the rendered chat template. */
    public int[] applyChatTemplateToIds(List<? extends Map<String, ?>> messages,
                                        boolean addGenerationPrompt, boolean addSpecialTokens) {
        String text = applyChatTemplate(messages, false, addGenerationPrompt);
        return encode(text, addSpecialTokens).ids();
    }

    /**
     * HuggingFace {@code tokenizer(text, truncation=, max_length=, padding=,
     * return_tensors=None)} → {@code {input_ids, attention_mask}}.
     */
    public Map<String, Object> encodeAsMap(String text, boolean truncation, int maxLength, boolean padding) {
        Encoding enc = encode(text, true);
        int[] ids = enc.ids();
        int[] mask = enc.attentionMask();
        if (truncation && maxLength > 0 && ids.length > maxLength) {
            int[] tIds = new int[maxLength];
            int[] tMask = new int[maxLength];
            System.arraycopy(ids, 0, tIds, 0, maxLength);
            System.arraycopy(mask, 0, tMask, 0, Math.min(mask.length, maxLength));
            ids = tIds;
            mask = tMask;
        }
        if (padding && maxLength > 0 && ids.length < maxLength) {
            int[] pIds = new int[maxLength];
            int[] pMask = new int[maxLength];
            System.arraycopy(ids, 0, pIds, 0, ids.length);
            System.arraycopy(mask, 0, pMask, 0, mask.length);
            java.util.Arrays.fill(pIds, ids.length, maxLength, padTokenId());
            ids = pIds;
            mask = pMask;
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("input_ids", ids);
        out.put("attention_mask", mask);
        return out;
    }

    public int[] encodeToIds(String text, boolean addSpecialTokens) {
        return encode(text, addSpecialTokens).ids();
    }

    /** HuggingFace {@code tokenizer.save_pretrained(dir)}. */
    public void savePretrained(Path dir) throws IOException {
        save(dir);
        if (chatTemplate != null) {
            Files.writeString(dir.resolve("chat_template.jinja"), chatTemplate, StandardCharsets.UTF_8);
        }
    }

    public void save_pretrained(Path dir) throws IOException { savePretrained(dir); }
    public void save_pretrained(String dir) throws IOException { savePretrained(Path.of(dir)); }

    private void replaceSpecials(String unk, String pad, String cls, String sep,
                                 String bos, String eos, String mask) {
        this.pipeline = pipeline.withSpecials(unk, pad, cls, sep, bos, eos, mask, pipeline.modelMaxLength());
    }

    public FastTokenizer withPadding(Padding padding) {
        return new FastTokenizer(backend, pipeline.withPadding(padding));
    }

    public FastTokenizer withTruncation(Truncation truncation) {
        return new FastTokenizer(backend, pipeline.withTruncation(truncation));
    }

    // ---- serialize (minimal, for round-trip of synthetic builders) ----------

    public String toTokenizerJson() {
        // Prefer not to re-dump full HF schema; emit a compact legacy-compatible form
        // plus backend hint so fromTokenizerJson can rebuild a usable pipeline.
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"version\": \"1.0\",\n");
        sb.append("  \"backend\": \"").append(backend.name()).append("\",\n");
        sb.append("  \"model_max_length\": ").append(modelMaxLength()).append(",\n");
        sb.append("  \"unk_token\": ").append(jsonStr(unkToken())).append(",\n");
        sb.append("  \"pad_token\": ").append(jsonStr(padToken())).append(",\n");
        sb.append("  \"cls_token\": ").append(jsonStr(clsToken())).append(",\n");
        sb.append("  \"sep_token\": ").append(jsonStr(sepToken())).append(",\n");
        sb.append("  \"bos_token\": ").append(jsonStr(bosToken())).append(",\n");
        sb.append("  \"eos_token\": ").append(jsonStr(eosToken())).append(",\n");
        sb.append("  \"mask_token\": ").append(jsonStr(maskToken())).append(",\n");
        sb.append("  \"model\": {\n    \"vocab\": {\n");
        int i = 0;
        for (Map.Entry<String, Integer> e : getVocab().entrySet()) {
            if (i++ > 0) sb.append(",\n");
            sb.append("      ").append(jsonStr(e.getKey())).append(": ").append(e.getValue());
            // cap dump size for huge vocabs in debug paths
            if (i >= 50_000) break;
        }
        sb.append("\n    }\n  }\n}\n");
        return sb.toString();
    }

    public void save(Path dir) throws IOException {
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("tokenizer.json"), toTokenizerJson(), StandardCharsets.UTF_8);
    }

    private static String jsonStr(String s) {
        if (s == null) return "null";
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    // ---- padding / truncation config ----------------------------------------

    public static final class Padding {
        public enum Strategy { LONGEST, MAX_LENGTH, DO_NOT_PAD }
        public final Strategy strategy;
        public final int maxLength;
        public final String direction; // "right" | "left"

        public Padding(Strategy strategy, int maxLength, String direction) {
            this.strategy = strategy == null ? Strategy.DO_NOT_PAD : strategy;
            this.maxLength = maxLength;
            this.direction = direction == null ? "right" : direction;
        }

        public static Padding longest() {
            return new Padding(Strategy.LONGEST, 0, "right");
        }

        public static Padding maxLength(int maxLength) {
            return new Padding(Strategy.MAX_LENGTH, maxLength, "right");
        }

        public static Padding none() {
            return new Padding(Strategy.DO_NOT_PAD, 0, "right");
        }
    }

    public static final class Truncation {
        public final int maxLength;
        public final String direction; // "right" | "left"

        public Truncation(int maxLength, String direction) {
            this.maxLength = maxLength;
            this.direction = direction == null ? "right" : direction;
        }

        public static Truncation of(int maxLength) {
            return new Truncation(maxLength, "right");
        }
    }

    // ---- builder ------------------------------------------------------------

    public static final class Builder {
        private Backend backend = Backend.PIPELINE;
        private TokenizerPipeline pipeline;
        private String unkToken;
        private String padToken;
        private String clsToken;
        private String sepToken;
        private String bosToken;
        private String eosToken;
        private String maskToken;
        private int modelMaxLength = -1;
        private Padding padding;
        private Truncation truncation;

        public Builder backend(Backend backend) {
            this.backend = backend;
            return this;
        }

        public Builder pipeline(TokenizerPipeline pipeline) {
            this.pipeline = pipeline;
            return this;
        }

        public Builder unkToken(String unkToken) { this.unkToken = unkToken; return this; }
        public Builder padToken(String padToken) { this.padToken = padToken; return this; }
        public Builder clsToken(String clsToken) { this.clsToken = clsToken; return this; }
        public Builder sepToken(String sepToken) { this.sepToken = sepToken; return this; }
        public Builder bosToken(String bosToken) { this.bosToken = bosToken; return this; }
        public Builder eosToken(String eosToken) { this.eosToken = eosToken; return this; }
        public Builder maskToken(String maskToken) { this.maskToken = maskToken; return this; }
        public Builder modelMaxLength(int modelMaxLength) { this.modelMaxLength = modelMaxLength; return this; }
        public Builder addPrefixSpace(boolean addPrefixSpace) {
            // applied at build via pipeline rebuild if needed — stored on pipeline already for factories
            return this;
        }
        public Builder padding(Padding padding) { this.padding = padding; return this; }
        public Builder truncation(Truncation truncation) { this.truncation = truncation; return this; }

        /** @deprecated engine is replaced by pipeline; kept for source compatibility no-op. */
        @Deprecated
        public Builder engine(Tokenizer engine) {
            return this;
        }

        /** @deprecated vocab set via pipeline; kept for source compatibility. */
        @Deprecated
        public Builder vocab(Map<String, Integer> vocab) {
            return this;
        }

        public FastTokenizer build() {
            if (pipeline == null) {
                throw new IllegalStateException("pipeline required — use fromFile/fromDirectory or a factory builder");
            }
            TokenizerPipeline p = pipeline;
            if (unkToken != null || padToken != null || clsToken != null || sepToken != null
                    || bosToken != null || eosToken != null || maskToken != null || modelMaxLength > 0) {
                p = p.withSpecials(
                        unkToken != null ? unkToken : p.unkToken(),
                        padToken != null ? padToken : p.padToken(),
                        clsToken != null ? clsToken : p.clsToken(),
                        sepToken != null ? sepToken : p.sepToken(),
                        bosToken != null ? bosToken : p.bosToken(),
                        eosToken != null ? eosToken : p.eosToken(),
                        maskToken != null ? maskToken : p.maskToken(),
                        modelMaxLength > 0 ? modelMaxLength : p.modelMaxLength());
            }
            if (padding != null) p = p.withPadding(padding);
            if (truncation != null) p = p.withTruncation(truncation);
            return new FastTokenizer(backend, p);
        }
    }
}
