/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners.prompt;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

import java.util.ArrayList;
import java.util.List;

/**
 * Java analog of HuggingFace {@code peft.tuners.multitask_prompt_embedding.MultitaskPromptEmbedding}.
 *
 * <p>Stores one prompt-embedding per task, then at forward time indexes into the
 * requested task ID and broadcasts to the batch.
 */
public class MultitaskPromptEncoder extends Module {

    private final List<Tensor> taskPrompts = new ArrayList<>();
    private final long numVirtualTokens;
    private final long tokenDim;
    private final int numTasks;

    public MultitaskPromptEncoder(long numVirtualTokens, long tokenDim, int numTasks) {
        super("MultitaskPromptEmbedding");
        this.numVirtualTokens = numVirtualTokens;
        this.tokenDim = tokenDim;
        this.numTasks = numTasks;
    }

    public void addTaskEmbedding(String taskName, Tensor w) {
        taskPrompts.add(w);
    }

    public Tensor forward(long batchSize, int taskId) {
        Tensor w = taskPrompts.get(Math.min(taskId, taskPrompts.size() - 1));
        return w.unsqueeze(0).expand(batchSize, numVirtualTokens, tokenDim);
    }

    public int numTasks() { return numTasks; }
}
