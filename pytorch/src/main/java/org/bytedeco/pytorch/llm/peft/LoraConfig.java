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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * LoRA configuration (mirrors Hugging Face {@code LoraConfig}).
 *
 * <p>Default targets match common attention projections:
 * {@code q_proj}, {@code v_proj}, {@code k_proj}, {@code o_proj}, {@code linear}.
 */
public final class LoraConfig extends PeftConfig {
    private final int r;
    private final double alpha;
    private final double dropout;
    private final List<String> targetModules;
    private final List<String> excludeModules;
    private final boolean freezeBase;
    private final boolean useRslora;
    private final String bias; // "none" | "all" | "lora_only"
    private final boolean fanInFanOut;
    private final List<String> modulesToSave;
    private final String initLoraWeights;
    private final Object layersToTransform; // Integer or List<Integer>; null => all
    private final List<String> layersPattern;
    private final Map<String, Integer> rankPattern;
    private final Map<String, Integer> alphaPattern;
    private final boolean useDora;
    private final int loraBias;
    private final int cordaConfigId;
    private final int montecloraConfigId;

    protected LoraConfig(Builder b) {
        super(b);
        if (b.r <= 0) {
            throw new IllegalArgumentException("r must be > 0");
        }
        this.r = b.r;
        this.alpha = b.alpha;
        this.dropout = b.dropout;
        this.targetModules = Collections.unmodifiableList(new ArrayList<>(b.targetModules));
        this.excludeModules = b.excludeModules == null ? null
                : Collections.unmodifiableList(new ArrayList<>(b.excludeModules));
        this.freezeBase = b.freezeBase;
        this.useRslora = b.useRslora;
        this.bias = b.bias;
        this.fanInFanOut = b.fanInFanOut;
        this.modulesToSave = b.modulesToSave == null ? null
                : Collections.unmodifiableList(new ArrayList<>(b.modulesToSave));
        this.initLoraWeights = b.initLoraWeights;
        this.layersToTransform = b.layersToTransform;
        this.layersPattern = b.layersPattern == null ? null
                : Collections.unmodifiableList(new ArrayList<>(b.layersPattern));
        this.rankPattern = b.rankPattern == null ? null
                : Collections.unmodifiableMap(new LinkedHashMap<>(b.rankPattern));
        this.alphaPattern = b.alphaPattern == null ? null
                : Collections.unmodifiableMap(new LinkedHashMap<>(b.alphaPattern));
        this.useDora = b.useDora;
        this.loraBias = b.loraBias;
        this.cordaConfigId = b.cordaConfigId;
        this.montecloraConfigId = b.montecloraConfigId;
    }

    public int r() { return r; }
    public double alpha() { return alpha; }
    public double dropout() { return dropout; }
    public List<String> targetModules() { return targetModules; }
    public List<String> excludeModules() { return excludeModules; }
    public boolean freezeBase() { return freezeBase; }
    public boolean useRslora() { return useRslora; }
    public String bias() { return bias; }
    public boolean fanInFanOut() { return fanInFanOut; }
    public List<String> modulesToSave() { return modulesToSave; }
    public String initLoraWeights() { return initLoraWeights; }
    public Object layersToTransform() { return layersToTransform; }
    public List<String> layersPattern() { return layersPattern; }
    public Map<String, Integer> rankPattern() { return rankPattern; }
    public Map<String, Integer> alphaPattern() { return alphaPattern; }
    public boolean useDora() { return useDora; }
    public int loraBias() { return loraBias; }
    public int cordaConfigId() { return cordaConfigId; }
    public int montecloraConfigId() { return montecloraConfigId; }

    /** {@code alpha / r} or {@code alpha / sqrt(r)} when rsLoRA is enabled. */
    public double scaling() {
        return useRslora ? alpha / Math.sqrt(r) : alpha / (double) r;
    }

    /** Resolve the effective rank for a given module name (with rank_pattern). */
    public int effectiveRank(String moduleName) {
        if (rankPattern == null || rankPattern.isEmpty()) return r;
        Integer match = rankPattern.get(moduleName);
        if (match != null) return match;
        for (Map.Entry<String, Integer> e : rankPattern.entrySet()) {
            if (moduleName.matches(e.getKey())) return e.getValue();
        }
        return r;
    }

    /** Resolve the effective alpha for a given module name (with alpha_pattern). */
    public double effectiveAlpha(String moduleName) {
        if (alphaPattern == null || alphaPattern.isEmpty()) return alpha;
        Integer match = alphaPattern.get(moduleName);
        if (match != null) return match;
        for (Map.Entry<String, Integer> e : alphaPattern.entrySet()) {
            if (moduleName.matches(e.getKey())) return e.getValue();
        }
        return alpha;
    }

    /** Whether a layer with {@code moduleName} should be transformed. */
    public boolean shouldTransformLayer(String moduleName, int layerIndex) {
        if (layersToTransform == null) return true;
        if (layersToTransform instanceof Integer) {
            return ((Integer) layersToTransform).intValue() == layerIndex;
        }
        if (layersToTransform instanceof List) {
            @SuppressWarnings("unchecked")
            List<Integer> list = (List<Integer>) layersToTransform;
            return list.contains(layerIndex);
        }
        return true;
    }

    /** Whether a module name is excluded by {@code exclude_modules}. */
    public boolean isExcluded(String moduleName) {
        if (excludeModules == null || excludeModules.isEmpty()) return false;
        for (String pat : excludeModules) {
            if (moduleName.equals(pat) || moduleName.endsWith(pat) || moduleName.matches(pat)) {
                return true;
            }
        }
        return false;
    }

    /** Standard PEFT {@code adapter_config.json} serialization. */
    @Override
    public Map<String, Object> toDict() {
        Map<String, Object> m = super.toDict();
        m.put("r", r);
        m.put("lora_alpha", alpha);
        m.put("lora_dropout", dropout);
        m.put("target_modules", new ArrayList<>(targetModules));
        if (excludeModules != null) m.put("exclude_modules", new ArrayList<>(excludeModules));
        m.put("bias", bias);
        m.put("use_rslora", useRslora);
        m.put("use_dora", useDora);
        m.put("fan_in_fan_out", fanInFanOut);
        m.put("init_lora_weights", initLoraWeights);
        if (modulesToSave != null) m.put("modules_to_save", new ArrayList<>(modulesToSave));
        if (layersToTransform != null) m.put("layers_to_transform", layersToTransform);
        if (layersPattern != null) m.put("layers_pattern", new ArrayList<>(layersPattern));
        if (rankPattern != null) m.put("rank_pattern", rankPattern);
        if (alphaPattern != null) m.put("alpha_pattern", alphaPattern);
        m.put("lora_bias", loraBias);
        m.put("corda_config_id", cordaConfigId);
        m.put("monteclora_config_id", montecloraConfigId);
        return m;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder extends PeftConfig.Builder<Builder> {
        private int r = 8;
        private double alpha = 16.0;
        private double dropout = 0.0;
        private List<String> targetModules = new ArrayList<>(
                Arrays.asList("q_proj", "v_proj", "k_proj", "o_proj", "linear", "lin"));
        private List<String> excludeModules = null;
        private boolean freezeBase = true;
        private boolean useRslora = false;
        private String bias = "none";
        private boolean fanInFanOut = false;
        private List<String> modulesToSave = null;
        private String initLoraWeights = "true";
        private Object layersToTransform = null;
        private List<String> layersPattern = null;
        private Map<String, Integer> rankPattern = null;
        private Map<String, Integer> alphaPattern = null;
        private boolean useDora = false;
        private int loraBias = 0;       // 0 = false, 1 = true (HF enum)
        private int cordaConfigId = -1;
        private int montecloraConfigId = -1;

        public Builder() {
            peftType(PeftType.LORA);
        }

        public Builder r(int r) {
            this.r = r;
            return this;
        }

        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        /** Snake alias matching Python {@code lora_alpha=}. */
        public Builder lora_alpha(double alpha) {
            return alpha(alpha);
        }

        /** Camel alias for HF field {@code lora_alpha}. */
        public Builder loraAlpha(double alpha) {
            return alpha(alpha);
        }

        public Builder dropout(double dropout) {
            this.dropout = dropout;
            return this;
        }

        /** Snake alias matching Python {@code lora_dropout=}. */
        public Builder lora_dropout(double dropout) {
            return dropout(dropout);
        }

        /** Camel alias for HF field {@code lora_dropout}. */
        public Builder loraDropout(double dropout) {
            return dropout(dropout);
        }

        public Builder targetModules(String... modules) {
            this.targetModules = new ArrayList<>(Arrays.asList(modules));
            return this;
        }

        public Builder target_modules(String... modules) {
            return targetModules(modules);
        }

        public Builder targetModules(List<String> modules) {
            this.targetModules = new ArrayList<>(Objects.requireNonNull(modules));
            return this;
        }

        public Builder target_modules(List<String> modules) {
            return targetModules(modules);
        }

        public Builder excludeModules(String... modules) {
            this.excludeModules = new ArrayList<>(Arrays.asList(modules));
            return this;
        }

        public Builder exclude_modules(String... modules) {
            return excludeModules(modules);
        }

        public Builder excludeModules(List<String> modules) {
            this.excludeModules = new ArrayList<>(Objects.requireNonNull(modules));
            return this;
        }

        public Builder exclude_modules(List<String> modules) {
            return excludeModules(modules);
        }

        public Builder freezeBase(boolean freezeBase) {
            this.freezeBase = freezeBase;
            return this;
        }

        public Builder useRslora(boolean useRslora) {
            this.useRslora = useRslora;
            return this;
        }

        /** Snake alias matching Python {@code use_rslora=}. */
        public Builder use_rslora(boolean useRslora) {
            return useRslora(useRslora);
        }

        public Builder bias(String bias) {
            this.bias = bias != null ? bias : "none";
            return this;
        }

        public Builder fanInFanOut(boolean v) {
            this.fanInFanOut = v;
            return this;
        }

        public Builder fan_in_fan_out(boolean v) {
            return fanInFanOut(v);
        }

        public Builder modulesToSave(String... modules) {
            this.modulesToSave = new ArrayList<>(Arrays.asList(modules));
            return this;
        }

        public Builder modules_to_save(String... modules) {
            return modulesToSave(modules);
        }

        public Builder modulesToSave(List<String> modules) {
            this.modulesToSave = new ArrayList<>(Objects.requireNonNull(modules));
            return this;
        }

        public Builder modules_to_save(List<String> modules) {
            return modulesToSave(modules);
        }

        /** {@code true} | {@code false} | {@code gaussian} | {@code eva} | {@code pissa} | {@code corda} | {@code loftq} | {@code orthogonal}. */
        public Builder initLoraWeights(String v) {
            this.initLoraWeights = v == null ? "true" : v;
            return this;
        }

        public Builder init_lora_weights(String v) {
            return initLoraWeights(v);
        }

        public Builder layersToTransform(Integer... indices) {
            this.layersToTransform = indices.length == 1
                    ? (Object) indices[0]
                    : new ArrayList<>(Arrays.asList(indices));
            return this;
        }

        public Builder layers_to_transform(Integer... indices) {
            return layersToTransform(indices);
        }

        public Builder layersToTransform(List<Integer> indices) {
            this.layersToTransform = indices.size() == 1
                    ? (Object) indices.get(0)
                    : new ArrayList<>(indices);
            return this;
        }

        public Builder layers_to_transform(List<Integer> indices) {
            return layersToTransform(indices);
        }

        public Builder layersPattern(String... patterns) {
            this.layersPattern = new ArrayList<>(Arrays.asList(patterns));
            return this;
        }

        public Builder layers_pattern(String... patterns) {
            return layersPattern(patterns);
        }

        public Builder rankPattern(Map<String, Integer> v) {
            this.rankPattern = v;
            return this;
        }

        public Builder rank_pattern(Map<String, Integer> v) {
            return rankPattern(v);
        }

        public Builder alphaPattern(Map<String, Integer> v) {
            this.alphaPattern = v;
            return this;
        }

        public Builder alpha_pattern(Map<String, Integer> v) {
            return alphaPattern(v);
        }

        public Builder useDora(boolean v) {
            this.useDora = v;
            return this;
        }

        public Builder use_dora(boolean v) {
            return useDora(v);
        }

        public Builder loraBias(boolean v) {
            this.loraBias = v ? 1 : 0;
            return this;
        }

        public Builder lora_bias(boolean v) {
            return loraBias(v);
        }

        public Builder cordaConfigId(int v) {
            this.cordaConfigId = v;
            return this;
        }

        public Builder montecloraConfigId(int v) {
            this.montecloraConfigId = v;
            return this;
        }

        /** Snake alias matching Python {@code task_type=} (delegates to PeftConfig). */
        public Builder task_type(String taskType) {
            return taskType(taskType);
        }

        @Override
        public LoraConfig build() {
            return new LoraConfig(this);
        }
    }
}