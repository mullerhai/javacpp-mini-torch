/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.knowledgegraph;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Lightweight in-process Wikipedia entity extractor.
 */
public final class WikipediaEntityRecognizer {

    private static final Pattern CAPITAL = Pattern.compile("\\b([A-Z][a-zA-Z]+(?:\\s+[A-Z][a-zA-Z]+)*)\\b");

    private final Set<String> blacklist = new HashSet<>(java.util.Arrays.asList(
            "I", "We", "You", "They", "He", "She", "It", "The", "A", "An", "And", "Or", "But", "So", "For"));

    public List<String> extract(String text) {
        if (text == null || text.isEmpty()) return new ArrayList<>();
        Set<String> out = new java.util.LinkedHashSet<>();
        Matcher m = CAPITAL.matcher(text);
        while (m.find()) {
            String s = m.group(1);
            if (s.length() <= 2) continue;
            if (blacklist.contains(s)) continue;
            out.add(s);
        }
        return new ArrayList<>(out);
    }
}
