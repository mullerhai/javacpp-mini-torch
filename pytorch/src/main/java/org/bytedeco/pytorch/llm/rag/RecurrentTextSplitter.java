/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.rag;

import java.util.ArrayList;
import java.util.List;

/**
 * LangChain {@code RecurrentTextSplitter(chunk_size, chunk_overlap, length_function=tiktoken_len)}
 * mirror.
 */
public final class RecurrentTextSplitter {

    public interface LengthFunction {
        int length(String text);
    }

    private final int chunkSize;
    private final int chunkOverlap;
    private final RecursiveSplitter.Separators separators;
    private final LengthFunction lengthFunction;

    public RecurrentTextSplitter(int chunkSize, int chunkOverlap, LengthFunction lengthFunction) {
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
        this.lengthFunction = lengthFunction;
        this.separators = RecursiveSplitter.Separators.defaultText();
    }

    public List<String> splitText(String text) {
        if (text == null || text.isEmpty()) return new ArrayList<>();
        List<String> out = new ArrayList<>();
        for (String chunk : splitWorker(text, separators)) {
            out.addAll(simpleChunk(chunk));
        }
        return out;
    }

    private List<String> splitWorker(String text, RecursiveSplitter.Separators ss) {
        List<String> pieces = new ArrayList<>();
        for (RecursiveSplitter.Separators.Separator s : ss.separators) {
            String[] parts = text.split(java.util.regex.Pattern.quote(s.value), -1);
            if (parts.length > 1) {
                pieces.addAll(joinSmall(parts, s.value));
                return pieces;
            }
        }
        pieces.add(text);
        return pieces;
    }

    private List<String> joinSmall(String[] parts, String sep) {
        List<String> out = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (String p : parts) {
            if (lengthFunction.length(p) > chunkSize) {
                if (sb.length() > 0) { out.add(sb.toString()); sb.setLength(0); }
                out.add(p);
            } else {
                if (sb.length() > 0) sb.append(sep);
                sb.append(p);
            }
        }
        if (sb.length() > 0) out.add(sb.toString());
        return out;
    }

    private List<String> simpleChunk(String text) {
        List<String> out = new ArrayList<>();
        java.util.List<String> tokens = tokenizeWords(text);
        if (tokens.isEmpty()) return out;
        int cSize = Math.max(1, chunkSize);
        int cOver = Math.min(chunkOverlap, cSize - 1);
        int start = 0;
        while (start < tokens.size()) {
            int end = Math.min(tokens.size(), start + cSize);
            String chunk = String.join(" ", tokens.subList(start, end));
            if (lengthFunction.length(chunk) > cSize + 8) {
                // shrink
                while (end > start && lengthFunction.length(chunk) > cSize) {
                    end--;
                    chunk = String.join(" ", tokens.subList(start, end));
                }
            }
            out.add(chunk);
            if (end == tokens.size()) break;
            start = end - cOver;
            if (start < 0) start = 0;
        }
        return out;
    }

    private static List<String> tokenizeWords(String text) {
        List<String> out = new ArrayList<>();
        for (String s : text.split("\\s+")) if (!s.isEmpty()) out.add(s);
        return out;
    }
}