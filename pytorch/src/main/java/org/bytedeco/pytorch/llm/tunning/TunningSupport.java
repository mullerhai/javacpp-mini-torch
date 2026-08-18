/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2 or at your option under the terms of the GNU General
 * Public License as published by the Free Software Foundation (subject to
 * the "Classpath" exception), either version 2 of the License, or any
 * later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 */
package org.bytedeco.pytorch.llm.tunning;

import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.jit.*;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;

/**
 * Shared support utilities for the {@code llm.tunning} samples.
 *
 * <p>All dataset loading falls back to in-memory synthetic datasets with the same row schema,
 * so every Ex compiles + runs in offline mode. When network access is available, the
 * {@link #loadDataset} method can be extended to use HfDatasets.
 */
public final class TunningSupport {

    private TunningSupport() {}

    // ---------------- Chat message types ------------------------------------

    public static final class ChatMessage {
        private final String role;
        private final String content;
        public ChatMessage(String role, String content) {
            this.role = Objects.requireNonNull(role);
            this.content = Objects.requireNonNull(content);
        }
        public String role() { return role; }
        public String content() { return content; }
        public Map<String, String> toMap() {
            Map<String, String> m = new LinkedHashMap<>();
            m.put("role", role);
            m.put("content", content);
            return m;
        }
        @Override public String toString() { return role + ": " + content; }
    }

    public static List<Map<String, String>> chatMessages(Object... pairs) {
        if (pairs.length % 2 != 0) {
            throw new IllegalArgumentException("pairs must be (role, content) pairs");
        }
        List<Map<String, String>> out = new ArrayList<>();
        for (int i = 0; i < pairs.length; i += 2) {
            out.add(new ChatMessage((String) pairs[i], (String) pairs[i + 1]).toMap());
        }
        return out;
    }

    // ---------------- Records ----------------------------------------------

    public static final class AlpacaRecord {
        public final String instruction;
        public final String input;
        public final String output;
        public AlpacaRecord(String instruction, String input, String output) {
            this.instruction = instruction == null ? "" : instruction;
            this.input = input == null ? "" : input;
            this.output = output == null ? "" : output;
        }
        public String prompt() {
            if (input.isEmpty()) {
                return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                     + "### Instruction:\n" + instruction
                     + "\n\n### Response:\n";
            }
            return "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
                 + "### Instruction:\n" + instruction
                 + "\n\n### Input:\n" + input
                 + "\n\n### Response:\n";
        }
    }

    public static final class PreferenceRecord {
        public final String prompt;
        public final String chosen;
        public final String rejected;
        public PreferenceRecord(String prompt, String chosen, String rejected) {
            this.prompt = prompt;
            this.chosen = chosen;
            this.rejected = rejected;
        }
    }

    public static final class Document {
        public final String id;
        public final String text;
        public final Map<String, String> metadata;
        public Document(String id, String text, Map<String, String> metadata) {
            this.id = id;
            this.text = text;
            this.metadata = metadata == null ? new LinkedHashMap<>() : metadata;
        }
        @Override public String toString() { return "Doc{" + id + ", " + text.length() + " chars}"; }
    }

    public static final class RetrievalHit {
        public final Document doc;
        public final double score;
        public RetrievalHit(Document doc, double score) {
            this.doc = doc; this.score = score;
        }
        @Override public String toString() {
            return String.format("Hit{id=%s, score=%.4f}", doc.id, score);
        }
    }

    public static final class TokenTaggerRecord {
        public final List<String> tokens;
        public final List<String> tags;
        public TokenTaggerRecord(List<String> tokens, List<String> tags) {
            this.tokens = tokens;
            this.tags = tags;
        }
    }

    public static final class DatasetInfo {
        public final String name;
        public final int trainSize;
        public final int evalSize;
        public final String source;
        public DatasetInfo(String name, int trainSize, int evalSize, String source) {
            this.name = name;
            this.trainSize = trainSize;
            this.evalSize = evalSize;
            this.source = source;
        }
        @Override public String toString() {
            return String.format("Dataset{name=%s, train=%d, eval=%d, source=%s}",
                    name, trainSize, evalSize, source);
        }
    }

    public static void banner(int n, String title) {
        System.out.println();
        System.out.println("================================================================");
        System.out.printf("  Example %02d: %s%n", n, title);
        System.out.println("================================================================");
    }

    // ---------------- Dataset loaders (in-memory fallback) --------------------

    /**
     * Stub dataset loader - returns in-memory sample.
     * In production, this would call HfDatasets.loadDataset when network is available.
     */
    public static List<Map<String, Object>> loadDataset(String path) {
        return loadDataset(path, null, "train", 1024);
    }

    public static List<Map<String, Object>> loadDataset(String path, String config) {
        return loadDataset(path, config, "train", 1024);
    }

    public static List<Map<String, Object>> loadDataset(String path, String config, String split, int take) {
        // In production: call HfDatasets.loadDataset(path, config, split, lc)
        // For now, return in-memory sample
        return alpacaSample(Math.min(take, 64));
    }

    /**
     * In-memory Alpaca-format sample dataset.
     */
    public static List<Map<String, Object>> alpacaSample(int n) {
        List<Map<String, Object>> rows = new ArrayList<>();
        String[] topics = {"translation", "summarization", "classification", "rewriting", "computation"};
        Random rng = new Random(42);
        for (int i = 0; i < n; i++) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("instruction", "Task " + i + " : classify the following text.");
            row.put("input", "Topic " + topics[i % topics.length] + " sample " + i);
            row.put("output", "The answer is " + topics[(i + 1) % topics.length] + ".");
            rows.add(row);
        }
        return rows;
    }

    public static List<Map<String, Object>> alpacaLongSample(int n) { return alpacaSample(n); }

    public static List<Map<String, Object>> chatmlSample(int n) {
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            Map<String, Object> row = new LinkedHashMap<>();
            List<Map<String, String>> msgs = new ArrayList<>();
            msgs.add(msg("system", "You are a helpful assistant."));
            msgs.add(msg("user", "Question " + i));
            msgs.add(msg("assistant", "Answer " + i));
            row.put("messages", msgs);
            rows.add(row);
        }
        return rows;
    }

    public static List<Map<String, Object>> preferenceSample(int n) {
        List<Map<String, Object>> rows = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("prompt", "Question " + i);
            row.put("chosen", "Detailed answer " + i);
            row.put("rejected", "Short answer " + i);
            rows.add(row);
        }
        return rows;
    }

    // ---------------- Tokenizer loader -------------------------------------

    /**
     * Returns a tokenizer. In production uses AutoTokenizer.fromPretrained,
     * falls back to GPT-2 when unavailable.
     */
    public static org.bytedeco.pytorch.llm.tokenizers.FastTokenizer tokenizerFor(String modelIdOrPath) {
        if (modelIdOrPath == null || modelIdOrPath.isBlank()) {
            return org.bytedeco.pytorch.llm.transformers.AutoTokenizer.gpt2();
        }
        try {
            return org.bytedeco.pytorch.llm.transformers.AutoTokenizer.fromPretrained(modelIdOrPath);
        } catch (Exception e) {
            try {
                Path p = Path.of(modelIdOrPath);
                if (Files.isDirectory(p)) {
                    return org.bytedeco.pytorch.llm.transformers.AutoTokenizer.fromDirectory(p);
                }
            } catch (IOException ioe) {
                // fall through
            }
            return org.bytedeco.pytorch.llm.transformers.AutoTokenizer.gpt2();
        }
    }

    // ---------------- Prompt formatting (real HF format strings) -----------

    public static Map<String, String> msg(String role, String content) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    /** HF Alpaca prompt template applied to one Alpaca row. */
    public static String alpacaPrompt(Map<String, Object> row) {
        String ins = str(row.getOrDefault("instruction", ""));
        String in  = str(row.getOrDefault("input", ""));
        String out = str(row.getOrDefault("output", ""));
        StringBuilder sb = new StringBuilder();
        sb.append("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n");
        sb.append("### Instruction:\n").append(ins).append("\n\n");
        if (!in.isEmpty()) sb.append("### Input:\n").append(in).append("\n\n");
        sb.append("### Response:\n").append(out);
        return sb.toString();
    }

    /** HF Llama-2 chat template. */
    public static String llama2ChatPrompt(List<Map<String, String>> messages) {
        StringBuilder sb = new StringBuilder();
        String system = null;
        for (Map<String, String> m : messages) {
            if ("system".equals(m.get("role"))) { system = m.get("content"); break; }
        }
        if (system == null) system = "You are a helpful, respectful and honest assistant.";
        sb.append("<s>[INST] <<SYS>>\n").append(system).append("\n<</SYS>>\n\n");
        int i = 0;
        for (Map<String, String> m : messages) {
            if ("system".equals(m.get("role"))) continue;
            if (i % 2 == 0) sb.append(" ");
            else sb.append(" [/INST] ");
            if ("user".equals(m.get("role"))) sb.append(m.get("content"));
            else sb.append(m.get("content")).append(" </s>");
            i++;
        }
        return sb.toString();
    }

    /** HF Guanaco prompt template. */
    public static String guanacoPrompt(List<Map<String, String>> messages) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> m : messages) {
            String role = m.get("role");
            if ("user".equals(role) || "human".equals(role)) {
                sb.append("### Human: ").append(m.get("content")).append("\n");
            } else if ("assistant".equals(role) || "gpt".equals(role)) {
                sb.append("### Assistant: ").append(m.get("content")).append("\n");
            }
        }
        return sb.toString();
    }

    /** ChatGLM / Qwen prompt template. */
    public static String chatglmPrompt(List<Map<String, String>> messages) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> m : messages) {
            String role = m.get("role");
            String content = m.get("content");
            if ("user".equals(role)) sb.append("[Round 1]\n\n问：").append(content).append("\n\n");
            else if ("assistant".equals(role)) sb.append("答：").append(content).append("\n\n");
            else if ("system".equals(role)) sb.append(content).append("\n\n");
        }
        return sb.toString();
    }

    /** HF StableVicuna / multi-turn Vicuna template. */
    public static String stableVicunaPrompt(List<Map<String, String>> messages) {
        StringBuilder sb = new StringBuilder();
        sb.append("A chat between a curious human and an artificial intelligence assistant. ");
        for (Map<String, String> m : messages) {
            String role = m.get("role");
            if ("user".equals(role) || "human".equals(role)) sb.append("### Human: ").append(m.get("content")).append(" \n");
            else if ("assistant".equals(role) || "gpt".equals(role)) sb.append("### Assistant: ").append(m.get("content")).append(" \n");
            else if ("system".equals(role)) sb.insert(0, m.get("content") + " ");
        }
        return sb.toString();
    }

    public static String samanthaPrompt(List<Map<String, String>> messages) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> m : messages) {
            String role = m.get("role");
            if ("system".equals(role)) sb.append(m.get("content")).append("\n\n");
            else if ("user".equals(role)) sb.append("User: ").append(m.get("content")).append("\n");
            else if ("assistant".equals(role)) sb.append("Assistant: ").append(m.get("content")).append("\n");
        }
        return sb.toString();
    }

    public static String rlhfPrompt(String prompt, String chosen) {
        return "问：" + prompt + "\n\n答：" + chosen;
    }

    public static String phiDialogPrompt(String system, String user, String response) {
        return "### Instruction: " + system + "\n\n### Input:\n" + user + "\n\n### Response:\n" + response;
    }

    // ---------------- SFT row -> tokenized features --------------------------

    /**
     * Mirrors {@code formatting_prompts_func} used in every SFT example: build a prompt
     * string from each row, tokenize, then return {@code {input_ids, attention_mask, labels}}.
     */
    public static Function<Map<String, Object>, Map<String, Object>> sftFormattingFunc(
            Function<Map<String, Object>, String> promptBuilder,
            org.bytedeco.pytorch.llm.tokenizers.FastTokenizer tokenizer,
            int maxLength,
            boolean addEos) {
        return row -> {
            String text = promptBuilder.apply(row);
            int[] ids = tokenizer.encode(text, false).ids();
            int n = Math.min(ids.length, maxLength);
            int[] trunc = new int[n];
            int[] mask = new int[n];
            for (int i = 0; i < n; i++) { trunc[i] = ids[i]; mask[i] = 1; }
            if (addEos && (n == 0 || trunc[n - 1] != tokenizer.eosId())) {
                int[] grown = new int[n + 1];
                int[] gm = new int[n + 1];
                System.arraycopy(trunc, 0, grown, 0, n);
                System.arraycopy(mask, 0, gm, 0, n);
                grown[n] = tokenizer.eosId();
                gm[n] = 1;
                trunc = grown; mask = gm;
            }
            Map<String, Object> f = new LinkedHashMap<>();
            f.put("input_ids", trunc);
            f.put("attention_mask", mask);
            f.put("labels", trunc.clone());
            return f;
        };
    }

    // ---------------- Misc --------------------------------------------------

    public static Map<String, Object> buildStubModel(String name) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", name);
        m.put("num_parameters", 7_000_000_000L);
        return m;
    }

    public static double[] simulateTrainingLoop(int epochs, double decay) {
        double[] losses = new double[Math.max(1, epochs * 100)];
        double prev = 3.2;
        Random rng = new Random(42);
        for (int i = 0; i < losses.length; i++) {
            prev = Math.max(0.05, prev * decay + (rng.nextDouble() - 0.5) * 0.02);
            losses[i] = prev;
        }
        return losses;
    }

    public static void printTable(String[] headers, List<String[]> rows) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < headers.length; i++) {
            sb.append(String.format("%-22s", headers[i]));
        }
        sb.append("\n");
        for (String[] r : rows) {
            for (int i = 0; i < r.length; i++) sb.append(String.format("%-22s", r[i]));
            sb.append("\n");
        }
        System.out.println(sb.toString());
    }

    /** Write rows as JSONL and return the path. */
    public static String writeTmpJsonl(List<Map<String, Object>> rows) {
        Path p;
        try {
            p = Files.createTempFile("sft_", ".jsonl");
            try (java.io.BufferedWriter bw = Files.newBufferedWriter(p)) {
                for (Map<String, Object> r : rows) {
                    bw.write(toJsonString(r));
                    bw.newLine();
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return p.toString();
    }

    private static String str(Object o) { return o == null ? "" : o.toString(); }

    private static String toJsonString(Map<String, Object> row) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, Object> e : row.entrySet()) {
            if (!first) sb.append(",");
            sb.append("\"").append(e.getKey()).append("\":");
            Object v = e.getValue();
            if (v instanceof String) {
                sb.append("\"").append(((String) v).replace("\\", "\\\\").replace("\"", "\\\"")).append("\"");
            } else if (v instanceof List) {
                sb.append("[");
                List<?> lst = (List<?>) v;
                for (int i = 0; i < lst.size(); i++) {
                    if (i > 0) sb.append(",");
                    Object item = lst.get(i);
                    if (item instanceof Map) {
                        sb.append(toJsonString((Map<String, Object>) item));
                    } else if (item instanceof String) {
                        sb.append("\"").append(((String) item).replace("\\", "\\\\").replace("\"", "\\\"")).append("\"");
                    } else {
                        sb.append(item);
                    }
                }
                sb.append("]");
            } else {
                sb.append(v);
            }
            first = false;
        }
        sb.append("}");
        return sb.toString();
    }
}
