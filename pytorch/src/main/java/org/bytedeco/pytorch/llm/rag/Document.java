/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.rag;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * LangChain {@code Document(store, page_content, metadata)} mirror.
 */
public final class Document {

    public final String pageContent;
    public final Map<String, Object> metadata;

    public Document(String pageContent, Map<String, Object> metadata) {
        this.pageContent = pageContent == null ? "" : pageContent;
        this.metadata = metadata == null ? new LinkedHashMap<>() : metadata;
    }

    public static Document fromHtml(String html, String sourceUrl) {
        String text = html.replaceAll("<[^>]+>", " ").replaceAll("\\s+", " ").trim();
        Map<String, Object> md = new LinkedHashMap<>();
        md.put("source", sourceUrl);
        return new Document(text, md);
    }

    public static List<Document> split(Document doc, RecursiveSplitter splitter) {
        List<Document> out = new ArrayList<>();
        for (String chunk : splitter.splitText(doc.pageContent)) {
            Map<String, Object> md = new LinkedHashMap<>(doc.metadata);
            out.add(new Document(chunk, md));
        }
        return out;
    }
}