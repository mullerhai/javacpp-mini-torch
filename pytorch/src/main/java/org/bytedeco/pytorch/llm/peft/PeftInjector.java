/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.peft;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.modeling.LlamaForCausalLM;
import org.bytedeco.pytorch.llm.transformers.modeling.MistralForCausalLM;
import org.bytedeco.pytorch.llm.transformers.modeling.Qwen2ForCausalLM;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.lang.reflect.Method;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Architecture-aware LoRA / IA3 injection used by {@link PeftModel#getPeftModel}.
 *
 * <p>Keeps the growing {@code instanceof} ladder out of {@link PeftModel} and
 * documents freeze-order: CausalLM registers adapters as children so freeze
 * must happen <em>before</em> attach; Llama/Qwen {@code borrowBase} adapters
 * are not children so freeze happens <em>after</em> attach.
 */
public final class PeftInjector {

    private PeftInjector() {}

    public static final class Result {
        public final Map<String, LoraLinear> adapters;
        public final boolean welded;

        Result(Map<String, LoraLinear> adapters, boolean welded) {
            this.adapters = adapters;
            this.welded = welded;
        }
    }

    public static Result injectLora(Module model, LoraConfig config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        if (model instanceof CausalLM clm) {
            if (config.freezeBase()) freezeModule(clm);
            clm.attachLora(config);
            return new Result(new LinkedHashMap<>(clm.loraAdapters()), true);
        }
        if (model instanceof MistralForCausalLM mis) {
            return injectLlamaFamily(mis.backbone(), config);
        }
        if (model instanceof LlamaForCausalLM llama) {
            return injectLlamaFamily(llama, config);
        }
        if (model instanceof Qwen2ForCausalLM qwen) {
            qwen.attachLora(config);
            if (config.freezeBase()) qwen.freezeBaseModelParameters();
            return new Result(new LinkedHashMap<>(qwen.loraAdapters()), true);
        }
        Result reflected = tryReflective(model, config);
        if (reflected.welded) return reflected;
        System.err.println("[peft] getPeftModel: no weld hook for " + model.getClass().getName()
                + " — returning empty PeftModel; caller must wrapLinear/add.");
        return new Result(Collections.emptyMap(), false);
    }

    private static Result injectLlamaFamily(LlamaForCausalLM llama, LoraConfig config) {
        llama.attachLora(config);
        if (config.freezeBase()) llama.freezeBaseModelParameters();
        return new Result(new LinkedHashMap<>(llama.loraAdapters()), true);
    }

    @SuppressWarnings("unchecked")
    private static Result tryReflective(Module model, LoraConfig config) {
        try {
            Method attach = model.getClass().getMethod("attachLora", LoraConfig.class);
            attach.invoke(model, config);
            if (config.freezeBase()) {
                try {
                    Method freeze = model.getClass().getMethod("freezeBaseModelParameters");
                    freeze.invoke(model);
                } catch (NoSuchMethodException ignored) {
                    freezeModule(model);
                }
            }
            try {
                Method adapters = model.getClass().getMethod("loraAdapters");
                Object raw = adapters.invoke(model);
                if (raw instanceof Map<?, ?> map) {
                    LinkedHashMap<String, LoraLinear> out = new LinkedHashMap<>();
                    for (Map.Entry<?, ?> e : map.entrySet()) {
                        if (e.getKey() instanceof String k && e.getValue() instanceof LoraLinear v) {
                            out.put(k, v);
                        }
                    }
                    return new Result(out, !out.isEmpty());
                }
            } catch (NoSuchMethodException ignored) {}
            return new Result(Collections.emptyMap(), true);
        } catch (NoSuchMethodException e) {
            return new Result(Collections.emptyMap(), false);
        } catch (Exception e) {
            System.err.println("[peft] reflective attachLora failed: " + e.getMessage());
            return new Result(Collections.emptyMap(), false);
        }
    }

    public static void freezeModule(Module model) {
        if (model == null) return;
        try {
            TensorVector pv = model.parameters();
            for (long i = 0, n = pv.size(); i < n; i++) {
                Tensor p = pv.get(i);
                if (p != null && !p.isNull() && p.defined()) {
                    try { p.requires_grad_(false); } catch (Exception ignored) {
                        try { p.set_requires_grad(false); } catch (Exception ignored2) {}
                    }
                }
            }
        } catch (Exception ignored) {}
    }

    public static Map<String, LinearImpl> namedLinearsOf(Module model) {
        if (model instanceof CausalLM clm) return clm.namedLinears();
        if (model instanceof LlamaForCausalLM llama) return llama.namedLinears();
        if (model instanceof Qwen2ForCausalLM qwen) return qwen.namedLinears();
        if (model instanceof MistralForCausalLM mis) return mis.backbone().namedLinears();
        try {
            Method m = model.getClass().getMethod("namedLinears");
            Object raw = m.invoke(model);
            if (raw instanceof Map<?, ?> map) {
                LinkedHashMap<String, LinearImpl> out = new LinkedHashMap<>();
                for (Map.Entry<?, ?> e : map.entrySet()) {
                    if (e.getKey() instanceof String k && e.getValue() instanceof LinearImpl v) {
                        out.put(k, v);
                    }
                }
                return out;
            }
        } catch (Exception ignored) {}
        return Collections.emptyMap();
    }

    /**
     * IA3 injection for models that expose {@code namedLinears} and a per-layer
     * {@code attachIa3} or, failing that, a map stored on {@link PeftModel}.
     * Llama/Qwen applyProj currently only dispatches LoRA; IA3 is wired for
     * {@link CausalLM} via {@link IA3Linear} registration next to LoRA.
     */
    public static Map<String, IA3Linear> injectIa3(Module model, IA3Config config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        Map<String, LinearImpl> named = namedLinearsOf(model);
        LinkedHashMap<String, IA3Linear> out = new LinkedHashMap<>();
        List<String> targets = List.of(config.targetModules());
        List<String> ff = List.of(config.feedforwardModules());
        for (Map.Entry<String, LinearImpl> e : named.entrySet()) {
            String name = e.getKey();
            String leaf = PeftModelHelper.leafName(name);
            boolean hit = false;
            for (String t : targets) {
                if (t != null && (leaf.equalsIgnoreCase(t)
                        || name.endsWith("." + t) || name.endsWith("/" + t))) {
                    hit = true;
                    break;
                }
            }
            if (!hit) continue;
            boolean feedforward = false;
            for (String t : ff) {
                if (t != null && (leaf.equalsIgnoreCase(t)
                        || name.endsWith("." + t) || name.endsWith("/" + t))) {
                    feedforward = true;
                    break;
                }
            }
            IA3Linear layer = IA3Linear.borrowBase(e.getValue(), config, feedforward);
            out.put(name, layer);
        }
        if (model instanceof CausalLM clm) {
            // Freeze first: register_module would otherwise put ia3_l into
            // CausalLM.parameters() and a later freeze would kill the adapter.
            freezeModule(clm);
            for (Map.Entry<String, IA3Linear> e : out.entrySet()) {
                clm.registerIa3(e.getKey(), e.getValue());
            }
        }
        return out;
    }
}
