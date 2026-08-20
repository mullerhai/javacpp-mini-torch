/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Minimal adapter_config.json reader. Uses {@link java.util.Properties} as a fallback
 * when JSON parsing is unavailable; production deployments should swap in a JSON parser.
 */
public final class AdapterConfigIO {

    private AdapterConfigIO() {}

    @SuppressWarnings("unchecked")
    public static Map<String, Object> read(String dir) {
        Path p = Paths.get(dir, "adapter_config.json");
        if (!Files.exists(p)) return new LinkedHashMap<>();
        try {
            java.io.BufferedReader r = Files.newBufferedReader(p);
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = r.readLine()) != null) sb.append(line);
            r.close();
            // Lightweight key=value parser; production code should use a JSON library.
            Map<String, Object> map = new LinkedHashMap<>();
            String body = sb.toString().trim();
            if (body.startsWith("{") && body.endsWith("}")) body = body.substring(1, body.length() - 1);
            int depth = 0;
            StringBuilder cur = new StringBuilder();
            for (char ch : body.toCharArray()) {
                if (ch == '{' || ch == '[') depth++;
                else if (ch == '}' || ch == ']') depth--;
                if (ch == ',' && depth == 0) {
                    parseKV(cur.toString(), map); cur.setLength(0);
                } else cur.append(ch);
            }
            if (cur.length() > 0) parseKV(cur.toString(), map);
            return map;
        } catch (java.io.IOException e) {
            throw new RuntimeException("adapter_config.json read failed: " + e.getMessage(), e);
        }
    }

    private static void parseKV(String kv, Map<String, Object> map) {
        int colon = kv.indexOf(':');
        if (colon < 0) return;
        String k = kv.substring(0, colon).trim().replaceAll("^\"|\"$", "");
        String v = kv.substring(colon + 1).trim();
        if (v.startsWith("\"")) v = v.substring(1, v.length() - 1);
        map.put(k, v);
    }
}
