/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.rag;

import java.io.BufferedReader;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * LangChain {@code WebBaseLoader.load()} mirror. Fetches HTML synchronously and converts
 * to a {@link Document}.
 */
public final class WebBaseLoader {

    private final List<String> urls;
    private final String encoding;

    public WebBaseLoader(List<String> urls) {
        this(urls, "UTF-8");
    }

    public WebBaseLoader(List<String> urls, String encoding) {
        this.urls = urls;
        this.encoding = encoding;
    }

    public List<Document> load() {
        List<Document> out = new ArrayList<>();
        for (String url : urls) {
            try {
                URL u = new URL(url);
                HttpURLConnection conn = (HttpURLConnection) u.openConnection();
                conn.setRequestProperty("User-Agent", "Mozilla/5.0");
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(15000);
                try (BufferedReader br = new BufferedReader(new java.io.InputStreamReader(conn.getInputStream(), encoding))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line).append('\n');
                    out.add(Document.fromHtml(sb.toString(), url));
                }
            } catch (IOException e) {
                throw new RuntimeException("WebBaseLoader failed for " + url + " : " + e.getMessage(), e);
            }
        }
        return out;
    }

    public static WebBaseLoader fromPaths(List<String> urls) {
        return new WebBaseLoader(urls);
    }
}