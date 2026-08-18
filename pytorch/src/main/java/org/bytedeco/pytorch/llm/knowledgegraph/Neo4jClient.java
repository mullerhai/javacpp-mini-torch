/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.knowledgegraph;

import java.util.ArrayList;
import java.util.List;

/**
 * Neo4j-like client. We don't talk to a real Neo4j server; we keep an in-memory representation
 * of the database and feed the {@code GraphCypherQAChain} with the same shape of data.
 */
public final class Neo4jClient {

    private final String url;
    private final String username;
    private final String password;
    private final KnowledgeGraph store;

    public Neo4jClient(String url, String username, String password) {
        this.url = url;
        this.username = username;
        this.password = password;
        this.store = new KnowledgeGraph();
    }

    public Neo4jClient addTriples(List<Triple> triples) {
        store.addTriples(triples);
        return this;
    }

    public String url() { return url; }
    public String username() { return username; }
    public String password() { return password; }
    public KnowledgeGraph graph() { return store; }

    public List<Triple> triples() {
        return new ArrayList<>(store.triples());
    }
}
