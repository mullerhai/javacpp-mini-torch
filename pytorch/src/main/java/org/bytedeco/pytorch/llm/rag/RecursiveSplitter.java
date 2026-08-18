/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.rag;

import java.util.ArrayList;
import java.util.List;

/**
 * LangChain {@code RecursiveCharacterTextSplitter(separators, chunk_size, chunk_overlap)}
 * mirror — splits on progressively smaller separators until every chunk fits.
 */
public final class RecursiveSplitter {

    public static final class Separators {
        public static final class Separator {
            public final String value;
            public Separator(String value) { this.value = value; }
        }
        public final List<Separator> separators;
        public Separators(List<Separator> separators) { this.separators = separators; }
        public static Separators defaultText() {
            return new Separators(java.util.Arrays.asList(
                    new Separator("\n\n"),
                    new Separator("\n"),
                    new Separator("."),
                    new Separator(" "),
                    new Separator("")));
        }
    }

    private final int chunkSize;
    private final int chunkOverlap;
    private final Separators separators;
    private final RecurrentTextSplitter.LengthFunction lengthFunction;

    public RecursiveSplitter(int chunkSize, int chunkOverlap,
                              RecurrentTextSplitter.LengthFunction lengthFunction) {
        this.chunkSize = chunkSize;
        this.chunkOverlap = chunkOverlap;
        this.lengthFunction = lengthFunction;
        this.separators = Separators.defaultText();
    }

    public List<String> splitText(String text) {
        if (text == null || text.isEmpty()) return new ArrayList<>();
        List<String> chunks = new ArrayList<>();
        for (String part : splitRecursive(text, separators)) {
            if (lengthFunction.length(part) <= chunkSize) {
                chunks.add(part);
            } else {
                // further split on smaller separators
                chunks.addAll(new RecurrentTextSplitter(chunkSize, chunkOverlap, lengthFunction).splitText(part));
            }
        }
        return chunks;
    }

    private List<String> splitRecursive(String text, Separators s) {
        List<String> pieces = new ArrayList<>();
        for (Separators.Separator sep : s.separators) {
            String[] parts = text.split(java.util.regex.Pattern.quote(sep.value), -1);
            for (String p : parts) {
                if (!p.isEmpty()) pieces.add(p);
            }
            if (pieces.size() > 1) return pieces;
        }
        pieces.add(text);
        return pieces;
    }
}