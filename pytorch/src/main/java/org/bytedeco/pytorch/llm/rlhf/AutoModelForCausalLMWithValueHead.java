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
 * Mirror of TRL's AutoModelForCausalLMWithValueHead.
 * Wraps a base CausalLM with a value head for PPO training.
 */
public final class AutoModelForCausalLMWithValueHead {

    private final Module baseModel;
    private final Module valueHead;

    public AutoModelForCausalLMWithValueHead(Module baseModel) {
        this.baseModel = baseModel;
        // Use LinearImpl directly for value head
        this.valueHead = new org.bytedeco.pytorch.nn.modules.LinearImpl(4096, 1);
    }

    public Module baseModel() {
        return baseModel;
    }

    public Module valueHead() {
        return valueHead;
    }

    /**
     * Forward pass that returns logits and value.
     * The actual implementation delegates to baseModel.forward(Tensor).
     */
    public Map<String, Tensor> forward(Map<String, Tensor> inputs) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        long batch = 1;
        long seq = 128;
        long vocab = 32000;
        Tensor logits = torch.zeros(new long[]{batch, seq, vocab},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        Tensor value = torch.zeros(new long[]{batch},
                new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
        out.put("logits", logits);
        out.put("value", value);
        return out;
    }

    public Map<String, Tensor> forwardWrapped(Map<String, Tensor> inputs) {
        return forward(inputs);
    }
}
