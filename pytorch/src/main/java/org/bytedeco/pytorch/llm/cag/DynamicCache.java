/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.cag;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Spans across requests: caches the K/V projection of a context so we don't need to re-encode
 * on every call. IBM 23 / CAG notebook (Notebook 23) uses this exact pattern.
 */
public final class DynamicCache {

    private final Map<String, long[][]> cacheKeys = new LinkedHashMap<>();
    private final Map<String, long[][]> cacheValues = new LinkedHashMap<>();

    public synchronized void update(String layer, long[][] key, long[][] value) {
        cacheKeys.put(layer, key.clone());
        cacheValues.put(layer, value.clone());
    }

    public synchronized long[][] getKey(String layer) {
        return cacheKeys.get(layer);
    }

    public synchronized long[][] getValue(String layer) {
        return cacheValues.get(layer);
    }

    public synchronized List<String> layers() {
        return new java.util.ArrayList<>(cacheKeys.keySet());
    }

    public synchronized int seqLen() {
        long[][] k = cacheKeys.values().stream().findFirst().orElse(null);
        return k == null ? 0 : k[0].length;
    }

    public synchronized void clear() {
        cacheKeys.clear();
        cacheValues.clear();
    }

    public synchronized int sizeBytes() {
        long elems = 0;
        for (long[][] k : cacheKeys.values()) elems += k.length * k[0].length;
        for (long[][] v : cacheValues.values()) elems += v.length * v[0].length;
        return (int) (elems * 8L);
    }
}