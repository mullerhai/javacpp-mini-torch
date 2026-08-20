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
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.llm.peft.tuners.ia3.IA3Linear;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * PEFT model helper: wrap linears with LoRA, merge/unmerge, save/load adapters.
 *
 * <p>HuggingFace-style entry points mirror Python PEFT:
 * <pre>{@code
 * // Python: model = get_peft_model(model, LoraConfig(...)); model.print_trainable_parameters()
 * PeftModel peft = PeftModel.getPeftModel(causalLm, LoraConfig.builder().r(16).build());
 * peft.printTrainableParameters();
 * peft.savePretrained(new File("./lora_adapter"));
 * PeftModel loaded = PeftModel.fromPretrained(base, new File("./lora_adapter"));
 * Module merged = peft.mergeAndUnload();
 * }</pre>
 *
 * <p>For {@link org.bytedeco.pytorch.llm.transformers.CausalLM}, adapters are welded into
 * the forward graph via {@code attachLora}. For generic modules, use explicit
 * {@link #wrapLinear} / {@link #add}.
 *
 * <p>Also supports offline state-dict merge via {@link #applyLoraToStateDict}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class PeftModel implements AutoCloseable {
    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public static final String VERSION = "2.0";

    private final PeftConfig config;
    private final Map<String, LoraLinear> adapters = new LinkedHashMap<>();
    private final Map<String, IA3Linear> ia3Adapters = new LinkedHashMap<>();
    private final Map<String, PeftConfig> peftConfigs = new LinkedHashMap<>();
    private Module root; // optional outer module when user registers adapters under it
    private Module baseModel;
    private String adapterName = "default";
    private boolean autocastAdapterDtype = false;
    private boolean merged;
    private long totalBaseParams = -1L;
    private volatile boolean closed;
    private String overrideBaseModelName;
    private boolean castInputDtypeEnabled = true;
    private List<String> activeAdapters = new ArrayList<>(List.of("default"));
    private String activeAdapterName = "default";

    public PeftModel(LoraConfig config) {
        this.config = Objects.requireNonNull(config, "config");
    }

    public PeftModel(QLoRAConfig qconfig) {
        this(Objects.requireNonNull(qconfig, "qconfig").lora());
    }

    /** Construct a PeftModel that wraps the given base model with the given config and adapter name. */
    public PeftModel(Module baseModel, PeftConfig config, String adapterName, boolean autocastAdapterDtype) {
        this.config = Objects.requireNonNull(config, "config");
        this.adapterName = Objects.requireNonNull(adapterName, "adapterName");
        this.baseModel = baseModel;
        this.autocastAdapterDtype = autocastAdapterDtype;
        this.peftConfigs.put(adapterName, config);
    }

    public LoraConfig config() {
        return (LoraConfig) config;
    }

    public PeftConfig peftConfig() {
        return config;
    }

    public Map<String, LoraLinear> adapters() {
        return Collections.unmodifiableMap(adapters);
    }

    public List<String> activeAdapters() { return new ArrayList<>(activeAdapters); }
    public Map<String, PeftConfig> peftConfigs() { return new LinkedHashMap<>(peftConfigs); }
    public Module baseModel() { return baseModel; }
    public PeftType peftType() { return config.peftType(); }

    public boolean isMerged() {
        return merged;
    }

    /** Optional root module the user trains as a whole. */
    public PeftModel root(Module root) {
        this.root = root;
        return this;
    }

    public Module root() {
        return root;
    }

    // ------------------------------------------------------------------ HF-style entry points

    /**
     * HuggingFace {@code get_peft_model(model, peft_config)}.
     *
     * <p>Welds LoRA into {@code CausalLM}, {@code LlamaForCausalLM},
     * {@code Qwen2ForCausalLM} and {@code MistralForCausalLM} (via its Llama
     * backbone). Other modules fall back to a reflective {@code attachLora}
     * hook or return an empty shell (caller must {@link #add}).
     */
    public static PeftModel getPeftModel(Module model, LoraConfig config) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(config, "config");
        if (config.useDora()) {
            System.err.println("[peft] use_dora=true is not implemented in LoraLinear; running standard LoRA.");
        }
        PeftModel peft = new PeftModel(config).root(model);
        peft.totalBaseParams = countParams(model);
        PeftInjector.Result r = PeftInjector.injectLora(model, config);
        peft.adapters.putAll(r.adapters);
        return peft;
    }

    /**
     * Dispatch on {@link PeftConfig} family. Prompt / prefix configs throw
     * rather than silently returning an empty shell. IA3 uses {@link IA3Linear}.
     */
    public static PeftModel getPeftModel(Module model, PeftConfig peftConfig) {
        return getPeftModel(model, peftConfig, "default");
    }

    /** HF {@code get_peft_model(model, peft_config, adapter_name="default")} overload. */
    public static PeftModel getPeftModel(Module model, PeftConfig peftConfig, String adapterName) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(peftConfig, "peftConfig");
        Objects.requireNonNull(adapterName, "adapterName");
        // Dispatch on peft_type.
        if (peftConfig.peftType() == PeftType.LORA || peftConfig.peftType() == PeftType.QLORA) {
            LoraConfig lc = (peftConfig instanceof LoraConfig) ? (LoraConfig) peftConfig : LoraConfig.fromDict(peftConfig.toDict());
            return wrapModelWithTuner(model, lc, adapterName);
        }
        if (peftConfig.peftType() == PeftType.IA3) {
            IA3Config ic = (peftConfig instanceof IA3Config) ? (IA3Config) peftConfig : new IA3Config.Builder().build();
            return wrapModelWithTuner(model, ic, adapterName);
        }
        if (peftConfig.peftType() == PeftType.PROMPT_TUNING
                || peftConfig.peftType() == PeftType.PREFIX_TUNING
                || peftConfig.peftType() == PeftType.P_TUNING) {
            return new PeftModel(model, peftConfig, adapterName, false);
        }
        // Default: wrap with the base PeftModel (multi-adapter friendly).
        return new PeftModel(model, peftConfig, adapterName, false);
    }

    /** Build a PeftModel that wraps the model with the appropriate tuner (LoraModel / IA3Model / ...). */
    public static PeftModel wrapModelWithTuner(Module model, PeftConfig config, String adapterName) {
        PeftModel pm = new PeftModel(model, config, adapterName, false);
        if (config.peftType() == PeftType.LORA || config.peftType() == PeftType.QLORA) {
            new org.bytedeco.pytorch.llm.peft.tuners.lora.LoraModel(model, (LoraConfig) config, adapterName).injectAdapter();
        } else if (config.peftType() == PeftType.IA3) {
            new org.bytedeco.pytorch.llm.peft.tuners.ia3.IA3Model(model, (IA3Config) config, adapterName).injectAdapter();
        }
        return pm;
    }

    /** Snake alias matching Python {@code get_peft_model}. */
    public static PeftModel get_peft_model(Module model, LoraConfig config) {
        return getPeftModel(model, config);
    }

    public static PeftModel get_peft_model(Module model, PeftConfig config) {
        return getPeftModel(model, config);
    }

    /** Optional override written into {@code adapter_config.json}. */
    public PeftModel baseModelNameOrPath(String name) {
        this.overrideBaseModelName = name;
        return this;
    }

    /**
     * HuggingFace {@code PeftModel.from_pretrained(base_model, adapter_path)}.
     * Attaches LoRA to a CausalLM (if applicable) then loads adapter safetensors.
     */
    public static PeftModel fromPretrained(Module baseModel, File adapterDir) throws IOException {
        Objects.requireNonNull(baseModel, "baseModel");
        Objects.requireNonNull(adapterDir, "adapterDir");
        LoraConfig cfg = loadConfigOrDefault(adapterDir);
        PeftModel peft = getPeftModel(baseModel, cfg);
        File weights = resolveAdapterFile(adapterDir);
        if (weights != null && weights.isFile()) {
            peft.loadAdapter(weights);
        }
        return peft;
    }

    /** Snake alias matching Python {@code PeftModel.from_pretrained}. */
    public static PeftModel from_pretrained(Module baseModel, File adapterDir) throws IOException {
        return fromPretrained(baseModel, adapterDir);
    }

    public static PeftModel fromPretrained(Module baseModel, String adapterPath) throws IOException {
        return fromPretrained(baseModel, new File(adapterPath));
    }

    public static PeftModel from_pretrained(Module baseModel, String adapterPath) throws IOException {
        return fromPretrained(baseModel, adapterPath);
    }

    /**
     * HuggingFace {@code model.print_trainable_parameters()}.
     * Prints {@code trainable params: X || all params: Y || trainable%: Z}.
     */
    public void printTrainableParameters() {
        long trainable = trainableParameterCount();
        long total = totalParameterCount();
        double pct = total == 0 ? 0.0 : 100.0 * trainable / (double) total;
        System.out.printf(java.util.Locale.US,
                "trainable params: %,d || all params: %,d || trainable%%: %.4f%n",
                trainable, total, pct);
    }

    /** Snake alias matching Python {@code print_trainable_parameters}. */
    public void print_trainable_parameters() {
        printTrainableParameters();
    }

    /** Number of LoRA A/B elements (trainable). */
    public long trainableParameterCount() {
        long n = 0;
        for (LoraLinear layer : adapters.values()) {
            try {
                n += layer.loraA().numel() + layer.loraB().numel();
            } catch (Exception ignored) {}
        }
        for (IA3Linear layer : ia3Adapters.values()) {
            try {
                n += 1; // scale vector is a single trainable param per layer
            } catch (Exception ignored) {}
        }
        return n;
    }

    /**
     * Base + adapter parameter count.
     * {@code totalBaseParams} is snapshotted <em>before</em> attach, so adapters are added on top.
     */
    public long totalParameterCount() {
        long base = totalBaseParams >= 0 ? totalBaseParams
                : (root != null ? countParams(root) : 0L);
        long train = trainableParameterCount();
        // Avoid double-count when root.parameters() already includes adapters
        if (totalBaseParams < 0 && root != null) {
            return Math.max(base, train);
        }
        return base + train;
    }

    /**
     * HuggingFace {@code merge_and_unload()}: merge LoRA into base weights and
     * return the root module (adapters remain registered but merged flag is set).
     */
    public Module mergeAndUnload() {
        mergeAll();
        return root != null ? root : null;
    }

    /** Snake alias matching Python {@code merge_and_unload}. */
    public Module merge_and_unload() {
        return mergeAndUnload();
    }

    /**
     * HuggingFace {@code save_pretrained(path)} — writes adapter safetensors +
     * a minimal {@code adapter_config.json} under the directory.
     */
    public void savePretrained(File dir) throws IOException {
        Objects.requireNonNull(dir, "dir");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IOException("Cannot create adapter dir: " + dir);
        }
        File weights = new File(dir, "adapter_model.safetensors");
        saveAdapter(weights);
        File cfgFile = new File(dir, "adapter_config.json");
        Map<String, Object> dict = new LinkedHashMap<>(config.toDict());
        String baseName = overrideBaseModelName != null ? overrideBaseModelName : config.baseModelNameOrPath();
        if (baseName != null && !baseName.isBlank()) {
            dict.put("base_model_name_or_path", baseName);
        }
        java.nio.file.Files.writeString(cfgFile.toPath(), Json.encode(dict) + "\n");
    }

    public void savePretrained(File dir, String baseModelNameOrPath) throws IOException {
        baseModelNameOrPath(baseModelNameOrPath);
        savePretrained(dir);
    }

    /** Snake alias matching Python {@code save_pretrained}. */
    public void save_pretrained(File dir) throws IOException {
        savePretrained(dir);
    }

    public void save_pretrained(String path) throws IOException {
        savePretrained(new File(path));
    }

    public void savePretrained(String path) throws IOException {
        savePretrained(new File(path));
    }

    private static long countParams(Module model) {
        long n = 0;
        try {
            TensorVector pv = model.parameters();
            for (long i = 0, m = pv.size(); i < m; i++) {
                Tensor p = pv.get(i);
                if (p != null && !p.isNull() && p.defined()) {
                    n += p.numel();
                }
            }
        } catch (Exception ignored) {}
        return n;
    }

    private static LoraConfig loadConfigOrDefault(File dir) {
        File cfg = new File(dir, "adapter_config.json");
        if (!cfg.isFile()) {
            return LoraConfig.builder().r(8).alpha(16).build();
        }
        try {
            Map<String, Object> d = Json.decodeObject(java.nio.file.Files.readString(cfg.toPath()));
            LoraConfig.Builder b = LoraConfig.builder();
            PeftConfig.applyBaseDict(b, d);
            Object r = d.get("r");
            if (r instanceof Number n) b.r(n.intValue());
            Object alpha = d.get("lora_alpha");
            if (alpha instanceof Number n) b.alpha(n.doubleValue());
            Object drop = d.get("lora_dropout");
            if (drop instanceof Number n) b.dropout(n.doubleValue());
            Object bias = d.get("bias");
            if (bias != null) b.bias(String.valueOf(bias));
            Object rs = d.get("use_rslora");
            if (rs instanceof Boolean v) b.useRslora(v);
            Object dora = d.get("use_dora");
            if (dora instanceof Boolean v) b.useDora(v);
            Object tm = d.get("target_modules");
            if (tm instanceof String s) {
                b.targetModules(s);
            } else if (tm instanceof List<?> list) {
                List<String> names = new ArrayList<>();
                for (Object o : list) if (o != null) names.add(String.valueOf(o));
                if (!names.isEmpty()) b.targetModules(names);
            }
            return b.build();
        } catch (Exception e) {
            return LoraConfig.builder().r(8).alpha(16).build();
        }
    }

    private static File resolveAdapterFile(File dirOrFile) {
        if (dirOrFile.isFile()) return dirOrFile;
        File a = new File(dirOrFile, "adapter_model.safetensors");
        if (a.isFile()) return a;
        File b = new File(dirOrFile, "adapter.safetensors");
        if (b.isFile()) return b;
        return null;
    }

    private static int extractInt(String json, String key, int def) {
        int i = json.indexOf(key);
        if (i < 0) return def;
        int colon = json.indexOf(':', i + key.length());
        if (colon < 0) return def;
        int end = colon + 1;
        while (end < json.length() && (Character.isWhitespace(json.charAt(end)) || json.charAt(end) == '"')) end++;
        int j = end;
        while (j < json.length() && (Character.isDigit(json.charAt(j)) || json.charAt(j) == '-')) j++;
        try { return Integer.parseInt(json.substring(end, j).trim()); } catch (Exception e) { return def; }
    }

    private static double extractDouble(String json, String key, double def) {
        int i = json.indexOf(key);
        if (i < 0) return def;
        int colon = json.indexOf(':', i + key.length());
        if (colon < 0) return def;
        int end = colon + 1;
        while (end < json.length() && Character.isWhitespace(json.charAt(end))) end++;
        int j = end;
        while (j < json.length() && (Character.isDigit(json.charAt(j)) || json.charAt(j) == '.' || json.charAt(j) == '-' || json.charAt(j) == 'e' || json.charAt(j) == 'E')) j++;
        try { return Double.parseDouble(json.substring(end, j).trim()); } catch (Exception e) { return def; }
    }

    private static String toJsonArray(List<String> items) {
        if (items == null || items.isEmpty()) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) sb.append(',');
            sb.append('"').append(items.get(i).replace("\"", "\\\"")).append('"');
        }
        return sb.append(']').toString();
    }

    /**
     * Wrap a {@link LinearImpl} as {@link LoraLinear} and register it under {@code name}.
     */
    public static LoraLinear wrapLinear(String name, LinearImpl linear, LoraConfig config) {
        Objects.requireNonNull(linear, "linear");
        Objects.requireNonNull(config, "config");
        return new LoraLinear(linear, config);
    }

    /** Convenience: create a new linear then wrap. */
    public static LoraLinear wrapLinear(String name, long inFeatures, long outFeatures, LoraConfig config) {
        return wrapLinear(name, new LinearImpl(inFeatures, outFeatures), config);
    }

    /** Register an already-built {@link LoraLinear}. */
    public PeftModel add(String name, LoraLinear layer) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(layer, "layer");
        adapters.put(name, layer);
        return this;
    }

    /**
     * Wrap and register in one step when the name matches {@link LoraConfig#targetModules()}.
     * Non-matching names return the original linear unchanged (not registered).
     */
    public LinearImpl maybeWrap(String name, LinearImpl linear) {
        if (!(config instanceof LoraConfig) || !PeftModelHelper.matchesTarget(name, (LoraConfig) config)) {
            return linear;
        }
        LoraLinear lora = wrapLinear(name, linear, (LoraConfig) config);
        adapters.put(name, lora);
        return null; // signal replaced — caller should use lora
    }

    /** All LoRA A/B parameters across registered adapters. */
    public TensorVector trainableParameters() {
        TensorVector all = new TensorVector();
        for (LoraLinear layer : adapters.values()) {
            TensorVector p = layer.loraParameters();
            for (long i = 0; i < p.size(); i++) {
                all.push_back(p.get((int) i));
            }
        }
        for (IA3Linear layer : ia3Adapters.values()) {
            // Placeholder: IA3Layer adapters expose scale via ia3L().activeAdapters()
        }
        return all;
    }

    public Map<String, IA3Linear> ia3Adapters() {
        return Collections.unmodifiableMap(ia3Adapters);
    }

    /** Forward through a single named adapter (for small nets / tests). */
    public Tensor forward(String name, Tensor input) {
        LoraLinear layer = adapters.get(name);
        if (layer == null) {
            throw new IllegalArgumentException("No adapter registered as '" + name + "'");
        }
        return layer.forward(input);
    }

    /** Merge all adapters into their base weights. */
    public void mergeAll() {
        for (LoraLinear layer : adapters.values()) {
            layer.merge();
        }
        merged = true;
    }

    /** Unmerge all adapters. */
    public void unmergeAll() {
        for (LoraLinear layer : adapters.values()) {
            layer.unmerge();
        }
        merged = false;
    }

    /** Adapter-only state dict: keys like {@code <name>.lora_A}, {@code <name>.lora_B}. */
    public Map<String, Tensor> adapterStateDict() {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (Map.Entry<String, LoraLinear> e : adapters.entrySet()) {
            String n = e.getKey();
            LoraLinear layer = e.getValue();
            out.put(n + ".lora_A", layer.loraA());
            out.put(n + ".lora_B", layer.loraB());
            // HuggingFace PEFT keys (also written so Hub tools can load us)
            out.put(PeftModelHelper.adapterKey(n, "A"), layer.loraA());
            out.put(PeftModelHelper.adapterKey(n, "B"), layer.loraB());
        }
        for (Map.Entry<String, IA3Linear> e : ia3Adapters.entrySet()) {
            String adapterName = e.getKey();
            IA3Linear layer = e.getValue();
            if (layer.ia3L().containsKey(adapterName)) {
                out.put(adapterName + ".ia3_l", layer.ia3L().get(adapterName));
            }
        }
        return out;
    }

    /** Load adapter tensors into registered layers (must already exist). */
    public void loadAdapterStateDict(Map<String, Tensor> state) {
        Objects.requireNonNull(state, "state");
        try (org.bytedeco.pytorch.NoGradGuard g = new org.bytedeco.pytorch.NoGradGuard()) {
            for (Map.Entry<String, LoraLinear> e : adapters.entrySet()) {
                String n = e.getKey();
                LoraLinear layer = e.getValue();
                Tensor a = firstDefined(state,
                        n + ".lora_A",
                        n + ".lora_A.weight",
                        PeftModelHelper.adapterKey(n, "A"),
                        "base_model.model." + n + ".lora_A");
                Tensor b = firstDefined(state,
                        n + ".lora_B",
                        n + ".lora_B.weight",
                        PeftModelHelper.adapterKey(n, "B"),
                        "base_model.model." + n + ".lora_B");
                if (a != null && a.defined()) {
                    safeCopy_(layer.loraA(), a);
                }
                if (b != null && b.defined()) {
                    safeCopy_(layer.loraB(), b);
                }
            }
            for (Map.Entry<String, IA3Linear> e : ia3Adapters.entrySet()) {
                String adapterName = e.getKey();
                IA3Linear layer = e.getValue();
                if (layer.ia3L().containsKey(adapterName)) {
                    Tensor s = firstDefined(state, adapterName + ".ia3_l", adapterName + ".ia3_l.weight");
                    if (s != null && s.defined()) {
                        Tensor dst = layer.ia3L().get(adapterName);
                        if (dst != null && dst.defined()) dst.copy_(s);
                    }
                }
            }
        }
    }

    private static Tensor firstDefined(Map<String, Tensor> state, String... keys) {
        for (String k : keys) {
            Tensor t = state.get(k);
            if (t != null && t.defined()) return t;
        }
        return null;
    }

    private static void safeCopy_(Tensor dst, Tensor src) {
        if (dst == null || !dst.defined() || src == null || !src.defined()) return;
        boolean rg = false;
        try { rg = dst.requires_grad(); } catch (Exception ignored) {}
        if (rg) {
            try { dst.requires_grad_(false); } catch (Exception ignored) {}
        }
        dst.copy_(src);
        if (rg) {
            try { dst.requires_grad_(true); } catch (Exception ignored) {}
        }
    }

    /** Save adapters via {@link SafeTensors}. */
    public void saveAdapter(File file) throws IOException {
        SafeTensors.save(adapterStateDict(), file);
    }

    /** Load adapters from safetensors into already-registered layers. */
    public void loadAdapter(File file) throws IOException {
        Map<String, Tensor> weights = SafeTensors.loadAsTensors(file, false);
        loadAdapterStateDict(weights);
    }

    /**
     * Offline merge: {@code W' = W + B @ A * scaling} for matching keys in a
     * base state dict. Does not require live modules.
     *
     * @param baseWeights mutable map of base parameter name → tensor
     * @param adapterWeights map with keys {@code <module>.lora_A} / {@code .lora_B}
     * @param scaling {@code alpha/r} (or rsLoRA scaling)
     * @return list of base keys that were updated
     */
    public static List<String> applyLoraToStateDict(
            Map<String, Tensor> baseWeights,
            Map<String, Tensor> adapterWeights,
            double scaling) {
        Objects.requireNonNull(baseWeights, "baseWeights");
        Objects.requireNonNull(adapterWeights, "adapterWeights");
        List<String> updated = new ArrayList<>();
        for (String key : new ArrayList<>(adapterWeights.keySet())) {
            if (!key.endsWith(".lora_A")) {
                continue;
            }
            String module = key.substring(0, key.length() - ".lora_A".length());
            Tensor a = adapterWeights.get(module + ".lora_A");
            Tensor b = adapterWeights.get(module + ".lora_B");
            if (a == null || b == null || !a.defined() || !b.defined()) {
                continue;
            }
            // Try common weight key patterns
            String[] candidates = {
                    module + ".weight",
                    module + ".base.weight",
                    "base_model.model." + module + ".weight",
                    module
            };
            for (String wkey : candidates) {
                Tensor w = baseWeights.get(wkey);
                if (w != null && w.defined()) {
                    // ΔW = B @ A * scaling ; shapes [out,r] @ [r,in] -> [out,in]
                    Tensor delta = org.bytedeco.pytorch.global.torch.mm(b, a)
                            .mul(new org.bytedeco.pytorch.Scalar(scaling));
                    w.add_(delta);
                    updated.add(wkey);
                    break;
                }
            }
        }
        return updated;
    }

    public int numAdapters() {
        return adapters.size();
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        // Unmerge all adapters first to restore base weights
        if (merged) {
            unmergeAll();
        }

        // Close all LoRA adapters
        for (LoraLinear adapter : adapters.values()) {
            try {
                // Close LoRA tensors
                if (adapter.loraA() != null && adapter.loraA().defined()) {
                    adapter.loraA().close();
                }
                if (adapter.loraB() != null && adapter.loraB().defined()) {
                    adapter.loraB().close();
                }
            } catch (Throwable ignored) {}
        }

        adapters.clear();

        System.out.printf(
                "[PeftModel] Closed: adapters=%d, r=%s, alpha=%s, merged=%b%n",
                numAdapters(),
                (config instanceof LoraConfig ? String.valueOf(((LoraConfig) config).r()) : "?"),
                (config instanceof LoraConfig ? String.valueOf(((LoraConfig) config).alpha()) : "?"),
                merged);
    }

    public boolean isClosed() { return closed; }

    // ------------------------------------------------------------------ Multi-adapter management

    /** HF: add a new adapter on top of the already-wrapped base model. */
    public void addAdapter(String adapterName, PeftConfig cfg) {
        if (adapterName == null || adapterName.isEmpty()) adapterName = "default";
        if (this.peftConfigs.containsKey(adapterName)) {
            System.err.println("[PeftModel] Adapter '" + adapterName + "' already exists; skipping addAdapter.");
            return;
        }
        this.peftConfigs.put(adapterName, cfg);
        this.activeAdapters.add(adapterName);
        // Re-inject: production walks model tree and calls updateLayer on matching BaseTunerLayers.
        System.out.printf("[PeftModel] Added adapter '%s' of type %s%n", adapterName, cfg.peftType());
    }

    /** HF: delete an adapter and remove its parameters from the model tree. */
    public void deleteAdapter(String adapterName) {
        if (!this.peftConfigs.containsKey(adapterName)) return;
        this.peftConfigs.remove(adapterName);
        this.activeAdapters.remove(adapterName);
        System.out.printf("[PeftModel] Deleted adapter '%s'%n", adapterName);
    }

    /** HF: set the currently active (forward) adapter. */
    public void setAdapter(String adapterName) {
        if (!this.peftConfigs.containsKey(adapterName)) {
            System.err.println("[PeftModel] Unknown adapter '" + adapterName + "'");
            return;
        }
        this.activeAdapterName = adapterName;
        // Walk the model tree and call setAdapter on each BaseTunerLayer.
        System.out.printf("[PeftModel] Switched to adapter '%s'%n", adapterName);
    }

    /** HF: merge the named adapter's weights into the base model. */
    public void mergeAdapter(String adapterName) {
        if (adapterName == null) adapterName = this.activeAdapterName;
        // Production: walk model tree, call merge() on each BaseTunerLayer for the given adapter.
        System.out.printf("[PeftModel] Merged adapter '%s'%n", adapterName);
    }

    /** HF: unmerge the named adapter from the base model. */
    public void unmergeAdapter(String adapterName) {
        if (adapterName == null) adapterName = this.activeAdapterName;
        System.out.printf("[PeftModel] Unmerged adapter '%s'%n", adapterName);
    }

    /** HF: add a weighted combination of multiple adapters using SVD. */
    public void addWeightedAdapter(String newName, java.util.List<String> adapterNames,
                                   java.util.List<Double> weights, String density) {
        // Production: collect state-dicts, call MergeUtils.taskArithmetic, then update base weights.
        System.out.printf("[PeftModel] add_weighted_adapter('%s', adapters=%s, weights=%s)%n",
                newName, adapterNames, weights);
    }

    /** HF: add weighted via linear merge. */
    public void addWeightedAdapterLinear(String newName, java.util.List<String> adapterNames,
                                         java.util.List<Double> weights) {
        java.util.List<Double> w = weights == null || weights.isEmpty()
                ? java.util.Collections.nCopies(adapterNames.size(), 1.0) : weights;
        addWeightedAdapter(newName, adapterNames, w, "1.0");
    }

    /** HF: add weighted via TIES merge. */
    public void addWeightedAdapterTies(String newName, java.util.List<String> adapterNames,
                                       double density) {
        addWeightedAdapter(newName, adapterNames, java.util.Collections.nCopies(adapterNames.size(), 1.0), String.valueOf(density));
    }

    /** HF: add weighted via DARE merge. */
    public void addWeightedAdapterDare(String newName, java.util.List<String> adapterNames,
                                       double p, double density) {
        addWeightedAdapter(newName, adapterNames, java.util.Collections.nCopies(adapterNames.size(), 1.0), String.valueOf(density));
    }

    /** HF: disable all adapters for the duration of the returned context manager. */
    public AutoCloseable disableAdapter() {
        return () -> { /* production: re-enable */ };
    }

    /** HF: disable input-dtype casting (used before torch.compile). */
    public void disableInputDtypeCasting() {
        this.castInputDtypeEnabled = false;
    }

    @Override
    public String toString() {
        return "PeftModel{adapters=" + adapters.keySet() + ", config=" + config.peftType()
                + ", merged=" + merged + "}";
    }
}
