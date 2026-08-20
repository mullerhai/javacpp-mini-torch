/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import java.util.Objects;

/**
 * Lightweight descriptor for a single chat "tool" — a base model or a LoRA adapter.
 * Lives in {@code llm.ui} so the engine has zero compile-time dependency on the
 * {@code samples.llm} package; callers in {@code samples.llm.web} translate from
 * their richer types ({@code ToolkitDemo.Tool}, etc.) into this DTO at construction
 * time.
 */
public final class ToolEntry {

    public final String name;
    public final String adapterPath;
    public final String description;

    public ToolEntry(String name, String adapterPath, String description) {
        this.name = Objects.requireNonNull(name, "name");
        this.adapterPath = adapterPath; // may be null for the base model
        this.description = description == null ? "" : description;
    }

    public ToolEntry(String name, String adapterPath) {
        this(name, adapterPath, "");
    }

    @Override public String toString() {
        return "ToolEntry{" + name + (adapterPath == null ? "" : " -> " + adapterPath) + "}";
    }
}