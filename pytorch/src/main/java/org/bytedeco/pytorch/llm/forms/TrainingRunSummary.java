/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.forms;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Per-example metrics container. Holds whatever key/value pairs the example wants to report.
 * Used by every Ex in the v2 rewrite.
 */
public final class TrainingRunSummary {

    public final String exampleName;
    public final Map<String, Object> records = new LinkedHashMap<>();

    public TrainingRunSummary(String exampleName) {
        this.exampleName = exampleName;
    }

    public TrainingRunSummary record(String key, Object value) {
        records.put(key, value);
        return this;
    }

    public Object get(String key) {
        return records.get(key);
    }

    public void print() {
        System.out.println("---- " + exampleName + " summary ----");
        for (Map.Entry<String, Object> e : records.entrySet()) {
            System.out.printf("  %-22s = %s%n", e.getKey(), e.getValue());
        }
    }

    public Map<String, Object> asMap() {
        return records;
    }
}