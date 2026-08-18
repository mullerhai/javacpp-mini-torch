/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.knowledgegraph;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A `(subject, predicate, object)` triple produced by {@link LLMGraphTransformer}.
 */
public final class Triple {
    public final String subject;
    public final String predicate;
    public final String object;
    public Triple(String subject, String predicate, String object) {
        this.subject = subject;
        this.predicate = predicate;
        this.object = object;
    }
    public Map<String, Object> toMap() {
        return java.util.Map.of("sub", subject, "rel", predicate, "obj", object);
    }
    public static List<Triple> parse(String text) {
        List<Triple> out = new ArrayList<>();
        for (String line : text.split("\\r?\\n")) {
            if (line.isBlank()) continue;
            String[] parts = line.split("\\|");
            if (parts.length == 3) out.add(new Triple(parts[0].trim(), parts[1].trim(), parts[2].trim()));
        }
        return out;
    }
}