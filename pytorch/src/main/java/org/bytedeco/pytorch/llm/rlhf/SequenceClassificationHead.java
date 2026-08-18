/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.rlhf;

import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.bytedeco.pytorch.global.torch.ScalarType;

/**
 * Mirror of Hugging Face's AutoModelForSequenceClassification with optional custom
 * classification head. Used by the Baichuan classifier from LLM-Tuning-master.
 */
public final class SequenceClassificationHead {

    public final Module base;
    public final Module head;

    public SequenceClassificationHead(Module base, int hiddenSize, int numLabels) {
        this.base = base;
        // Use LinearImpl directly without Module.apply wrapper
        this.head = new org.bytedeco.pytorch.nn.modules.LinearImpl(hiddenSize, numLabels);
    }

    /**
     * Baichuan pooled-logits trick:
     *   sequence_lengths = -1
     *   pooled = logits[arange(batch), sequence_lengths]
     */
    public Tensor pooledLogits(Tensor logits, Tensor inputIds, long padTokenId) {
        // Take last non-pad token per sequence: select(dim=1, index=seq-1)
        long seq = logits.sizes().get(1);
        Tensor last = logits.select(1, (int) seq - 1);
        return last;
    }

    public int numLabels() {
        return 0;
    }
}
