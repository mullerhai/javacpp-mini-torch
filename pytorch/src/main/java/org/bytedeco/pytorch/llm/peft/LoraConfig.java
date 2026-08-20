/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * LoRA configuration (mirrors Hugging Face {@code peft.LoraConfig} v0.20.0).
 *
 * <p>Carries full nested-config support for LoftQ / EVA / CorDA / LoRA-GA / VeLoRA /
 * Monteclora / Bd-LoRA / Arrow / RuntimeConfig, plus 11 init_lora_weights variants
 * (true / false / gaussian / olora / pissa / pissa_niter_K / corda / loftq / eva /
 * orthogonal / mica / lora_ga).
 *
 * <p>This is a fat extension of the existing 18-field config; all previously shipped
 * public methods and field accessors are preserved.
 */
public final class LoraConfig extends PeftConfig {

    // ---------- existing core fields ----------
    private final int r;
    private final double alpha;
    private final double dropout;
    private final List<String> targetModules;
    private final List<String> excludeModules;
    private final boolean freezeBase;
    private final boolean useRslora;
    private final String bias;
    private final boolean fanInFanOut;
    private final List<String> modulesToSave;
    private final String initLoraWeights;
    private final Object layersToTransform;
    private final List<String> layersPattern;
    private final Map<String, Integer> rankPattern;
    private final Map<String, Integer> alphaPattern;
    private final boolean useDora;
    private final int loraBias;
    private final boolean allLinear;

    // ---------- newly added nested dataclass fields ----------
    private final LoftQConfig loftQConfig;
    private final EvaConfig evaConfig;
    private final CordaConfig cordaConfig;
    private final LoraGAConfig loraGaConfig;
    private final VeloraConfig veloraConfig;
    private final MontecloraConfig montecloraConfig;
    private final BdLoraConfig bdLoraConfig;
    private final ArrowConfig arrowConfig;
    private final LoraRuntimeConfig runtimeConfig;

    // ---------- new top-level fields ----------
    private final List<String> targetParameters;
    private final boolean useBdLora;
    private final boolean useArrow;
    private final boolean useMonteclora;
    private final boolean useQalora;
    private final int qaloraGroupSize;
    private final List<Integer> aloraInvocationTokens;
    private final List<List<Integer>> layerReplication;
    private final List<Integer> trainableTokenIndices;
    private final Map<String, List<Integer>> trainableTokenIndicesByAdapter;
    private final boolean ensureWeightTying;
    /** HF compat: extra_kwarg_for_hf ↔ runtime_config. */
    private final int montecloraConfigId;
    private final int cordaConfigId;

    /** Canonical set of valid {@code init_lora_weights} values (plus PiSSA family). */
    public static final Set<String> LORA_INIT_VALUES = Collections.unmodifiableSet(new java.util.HashSet<>(Arrays.asList(
            "true", "false", "gaussian", "olora", "pissa", "corda", "loftq",
            "eva", "orthogonal", "mica", "lora_ga")));

    /** True if {@code v} starts with {@code "pissa_niter_"} (configurable-SVD variant). */
    public static boolean isPissaVariant(String v) { return v != null && v.startsWith("pissa_niter_"); }
    /** Parses the numeric suffix of {@code "pissa_niter_K"}; returns -1 on mismatch. */
    public static int pissaNiter(String v) {
        if (!isPissaVariant(v)) return -1;
        try { return Integer.parseInt(v.substring("pissa_niter_".length())); } catch (Exception e) { return -1; }
    }

    protected LoraConfig(Builder b) {
        super(b);
        if (b.r <= 0) throw new IllegalArgumentException("r must be > 0");
        if (b.alpha <= 0) throw new IllegalArgumentException("alpha must be > 0");
        if (b.dropout < 0.0 || b.dropout > 1.0) throw new IllegalArgumentException("dropout must be in [0, 1]");
        if (b.bias != null && !Arrays.asList("none", "all", "lora_only").contains(b.bias)) {
            throw new IllegalArgumentException("bias must be one of [none, all, lora_only]");
        }
        validateInitLoraWeights(b.initLoraWeights);
        if (b.useDora && b.targetParameters != null && !b.targetParameters.isEmpty()) {
            throw new IllegalArgumentException("use_dora=True is not compatible with target_parameters");
        }
        if (b.useDora && b.loraGaConfig != null) {
            throw new IllegalArgumentException("lora_ga init is not compatible with use_dora=True");
        }
        if (b.useArrow && b.useBdLora) {
            throw new IllegalArgumentException("use_arrow and use_bdlora are mutually exclusive");
        }
        if (b.cordaConfig != null && b.initLoraWeights != null && b.initLoraWeights.contains("loftq")) {
            throw new IllegalArgumentException("corda_config and loftq init are mutually exclusive");
        }
        if (b.targetModules != null && !b.targetModules.isEmpty()
            && b.targetParameters != null && !b.targetParameters.isEmpty()) {
            throw new IllegalArgumentException("target_modules and target_parameters are mutually exclusive");
        }

        this.r = b.r;
        this.alpha = b.alpha;
        this.dropout = b.dropout;
        this.allLinear = b.allLinear;
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

        this.loftQConfig = b.loftQConfig;
        this.evaConfig = b.evaConfig;
        this.cordaConfig = b.cordaConfig;
        this.loraGaConfig = b.loraGaConfig;
        this.veloraConfig = b.veloraConfig;
        this.montecloraConfig = b.montecloraConfig;
        this.bdLoraConfig = b.bdLoraConfig;
        this.arrowConfig = b.arrowConfig;
        this.runtimeConfig = b.runtimeConfig;

        this.targetParameters = b.targetParameters == null ? null
                : Collections.unmodifiableList(new ArrayList<>(b.targetParameters));
        this.useBdLora = b.useBdLora;
        this.useArrow = b.useArrow;
        this.useMonteclora = b.useMonteclora;
        this.useQalora = b.useQalora;
        this.qaloraGroupSize = b.qaloraGroupSize;
        this.aloraInvocationTokens = b.aloraInvocationTokens == null ? null
                : Collections.unmodifiableList(new ArrayList<>(b.aloraInvocationTokens));
        this.layerReplication = b.layerReplication == null ? null
                : Collections.unmodifiableList(new ArrayList<>(b.layerReplication));
        this.trainableTokenIndices = b.trainableTokenIndices == null ? null
                : Collections.unmodifiableList(new ArrayList<>(b.trainableTokenIndices));
        this.trainableTokenIndicesByAdapter = b.trainableTokenIndicesByAdapter == null ? null
                : Collections.unmodifiableMap(new LinkedHashMap<>(b.trainableTokenIndicesByAdapter));
        this.ensureWeightTying = b.ensureWeightTying;
        this.montecloraConfigId = b.montecloraConfigId;
        this.cordaConfigId = b.cordaConfigId;
    }

    private static void validateInitLoraWeights(String v) {
        if (v == null) return;
        if (LORA_INIT_VALUES.contains(v) || isPissaVariant(v)) return;
        throw new IllegalArgumentException("Unsupported init_lora_weights=" + v
            + "; choose one of " + LORA_INIT_VALUES + " or pissa_niter_K");
    }

    // ---------- existing public accessors ----------
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
    public boolean isAllLinear() { return allLinear; }
    public int cordaConfigId() { return cordaConfigId; }
    public int montecloraConfigId() { return montecloraConfigId; }

    // ---------- new public accessors ----------
    public LoftQConfig loftqConfig() { return loftQConfig; }
    public EvaConfig evaConfig() { return evaConfig; }
    public CordaConfig cordaConfig() { return cordaConfig; }
    public LoraGAConfig loraGaConfig() { return loraGaConfig; }
    public VeloraConfig veloraConfig() { return veloraConfig; }
    public MontecloraConfig montecloraConfig() { return montecloraConfig; }
    public BdLoraConfig bdLoraConfig() { return bdLoraConfig; }
    public ArrowConfig arrowConfig() { return arrowConfig; }
    public LoraRuntimeConfig runtimeConfig() { return runtimeConfig; }
    public List<String> targetParameters() { return targetParameters; }
    public boolean useBdLora() { return useBdLora; }
    public boolean useArrow() { return useArrow; }
    public boolean useMonteclora() { return useMonteclora; }
    public boolean useQalora() { return useQalora; }
    public int qaloraGroupSize() { return qaloraGroupSize; }
    public List<Integer> aloraInvocationTokens() { return aloraInvocationTokens; }
    public List<List<Integer>> layerReplication() { return layerReplication; }
    public List<Integer> trainableTokenIndices() { return trainableTokenIndices; }
    public Map<String, List<Integer>> trainableTokenIndicesByAdapter() { return trainableTokenIndicesByAdapter; }
    public boolean ensureWeightTying() { return ensureWeightTying; }

    /** HF {@code __post_init__} helper: check the init_lora_weights supports further setup. */
    public boolean isEffectiveDora() { return useDora; }
    public boolean isEffectiveRslora() { return useRslora; }

    /** Resolve which LoraVariant to wire into this layer at injection time. */
    public String resolveLoraVariantName() {
        if (useDora) return "Dora";
        if (useMonteclora) return "Monteclora";
        if (useArrow) return "Arrow";
        if (useBdLora) return "BdLoRa";
        if (veloraConfig != null) return "VeLoRa";
        if (evaConfig != null) return "Eva";
        if (aloraInvocationTokens != null) return "ALoRa";
        return "default";
    }

    public double scaling() {
        return useRslora ? alpha / Math.sqrt(r) : alpha / (double) r;
    }

    public int effectiveRank(String moduleName) {
        if (rankPattern == null || rankPattern.isEmpty()) return r;
        Integer match = rankPattern.get(moduleName);
        if (match != null) return match;
        for (Map.Entry<String, Integer> e : rankPattern.entrySet()) {
            if (moduleName.matches(e.getKey())) return e.getValue();
        }
        return r;
    }

    public double effectiveAlpha(String moduleName) {
        if (alphaPattern == null || alphaPattern.isEmpty()) return alpha;
        Integer match = alphaPattern.get(moduleName);
        if (match != null) return match.doubleValue();
        for (Map.Entry<String, Integer> e : alphaPattern.entrySet()) {
            if (moduleName.matches(e.getKey())) return e.getValue().doubleValue();
        }
        return alpha;
    }

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

    public boolean isExcluded(String moduleName) {
        if (excludeModules == null || excludeModules.isEmpty()) return false;
        for (String pat : excludeModules) {
            if (moduleName.equals(pat) || moduleName.endsWith(pat) || moduleName.matches(pat)) return true;
        }
        return false;
    }

    @Override
    public Map<String, Object> toDict() {
        Map<String, Object> m = super.toDict();
        m.put("r", r);
        m.put("lora_alpha", alpha);
        m.put("lora_dropout", dropout);
        if (allLinear) {
            m.put("target_modules", "all-linear");
        } else {
            m.put("target_modules", new ArrayList<>(targetModules));
        }
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

        // Nested configs — encode only when non-empty/non-default.
        if (loftQConfig != null && loftQConfig.nonDefault()) {
            Map<String, Object> lq = new LinkedHashMap<>();
            lq.put("loftq_bits", loftQConfig.loftqBits);
            lq.put("loftq_iter", loftQConfig.loftqIter);
            m.put("loftq_config", lq);
        }
        if (evaConfig != null) m.put("eva_config", evaConfig.toMap());
        if (cordaConfig != null) m.put("corda_config", cordaConfig.toMap());
        if (loraGaConfig != null) m.put("lora_ga_config", loraGaConfig.toMap());
        if (veloraConfig != null) m.put("velora_config", veloraConfig.toMap());
        if (montecloraConfig != null) m.put("monteclora_config", montecloraConfig.toMap());
        if (bdLoraConfig != null) m.put("bd_lora_config", bdLoraConfig.toMap());
        if (arrowConfig != null) m.put("arrow_config", arrowConfig.toMap());
        if (runtimeConfig != null) m.put("runtime_config", runtimeConfig.toMap());

        if (targetParameters != null && !targetParameters.isEmpty()) m.put("target_parameters", new ArrayList<>(targetParameters));
        if (useBdLora) m.put("use_bdlora", true);
        if (useArrow) m.put("use_arrow", true);
        if (useMonteclora) m.put("use_monteclora", true);
        if (useQalora) {
            m.put("use_qalora", true);
            m.put("qalora_group_size", qaloraGroupSize);
        }
        if (aloraInvocationTokens != null) m.put("alora_invocation_tokens", new ArrayList<>(aloraInvocationTokens));
        if (layerReplication != null) m.put("layer_replication", layerReplication);
        if (trainableTokenIndices != null) m.put("trainable_token_indices", new ArrayList<>(trainableTokenIndices));
        if (trainableTokenIndicesByAdapter != null) m.put("trainable_token_indices_by_adapter", trainableTokenIndicesByAdapter);
        if (ensureWeightTying) m.put("ensure_weight_tying", true);
        return m;
    }

    /** Snake-case alias for Python {@code to_dict()} parity. */
    public Map<String, Object> to_dict() { return toDict(); }

    /** Build a config from a parsed JSON dict (HF {@code LoraConfig.from_dict}). */
    @SuppressWarnings("unchecked")
    public static LoraConfig fromDict(Map<String, Object> d) {
        Builder b = new Builder();
        if (d.get("r") instanceof Number) b.r(((Number) d.get("r")).intValue());
        if (d.get("lora_alpha") instanceof Number) b.alpha(((Number) d.get("lora_alpha")).doubleValue());
        if (d.get("lora_dropout") instanceof Number) b.dropout(((Number) d.get("lora_dropout")).doubleValue());
        if (d.get("target_modules") instanceof String) {
            String s = (String) d.get("target_modules");
            if ("all-linear".equals(s)) b.allLinear(true);
            else b.targetModules(s);
        } else if (d.get("target_modules") instanceof List) {
            b.targetModules((List<String>) d.get("target_modules"));
        }
        if (d.get("exclude_modules") instanceof List) b.excludeModules((List<String>) d.get("exclude_modules"));
        if (d.get("bias") instanceof String) b.bias((String) d.get("bias"));
        if (d.get("use_rslora") instanceof Boolean) b.useRslora((Boolean) d.get("use_rslora"));
        if (d.get("use_dora") instanceof Boolean) b.useDora((Boolean) d.get("use_dora"));
        if (d.get("fan_in_fan_out") instanceof Boolean) b.fanInFanOut((Boolean) d.get("fan_in_fan_out"));
        if (d.get("init_lora_weights") instanceof String) b.initLoraWeights((String) d.get("init_lora_weights"));
        if (d.get("modules_to_save") instanceof List) b.modulesToSave((List<String>) d.get("modules_to_save"));
        if (d.get("layers_to_transform") != null) {
            Object ltt = d.get("layers_to_transform");
            if (ltt instanceof Number) b.layersToTransform(((Number) ltt).intValue());
            else if (ltt instanceof List) b.layersToTransform((List<Integer>) ltt);
        }
        if (d.get("layers_pattern") instanceof List) b.layersPattern((List<String>) d.get("layers_pattern"));
        if (d.get("rank_pattern") instanceof Map) b.rankPattern((Map<String, Integer>) d.get("rank_pattern"));
        if (d.get("alpha_pattern") instanceof Map) b.alphaPattern((Map<String, Integer>) d.get("alpha_pattern"));
        if (d.get("lora_bias") instanceof Boolean) b.loraBias((Boolean) d.get("lora_bias"));
        if (d.get("trainable_token_indices") instanceof List) {
            List<Integer> ti = new ArrayList<>();
            for (Object v : (List<?>) d.get("trainable_token_indices")) ti.add(((Number) v).intValue());
            b.trainableTokenIndices(ti);
        }
        if (d.get("use_monteclora") instanceof Boolean) b.useMonteclora((Boolean) d.get("use_monteclora"));
        if (d.get("use_bdlora") instanceof Boolean) b.useBdLora((Boolean) d.get("use_bdlora"));
        if (d.get("use_arrow") instanceof Boolean) b.useArrow((Boolean) d.get("use_arrow"));
        if (d.get("use_qalora") instanceof Boolean) b.useQalora((Boolean) d.get("use_qalora"));
        if (d.get("qalora_group_size") instanceof Number) b.qaloraGroupSize(((Number) d.get("qalora_group_size")).intValue());
        if (d.get("alora_invocation_tokens") instanceof List) {
            List<Integer> ai = new ArrayList<>();
            for (Object v : (List<?>) d.get("alora_invocation_tokens")) ai.add(((Number) v).intValue());
            b.aloraInvocationTokens(ai);
        }
        if (d.get("loftq_config") instanceof Map) {
            Map<String, Object> lq = (Map<String, Object>) d.get("loftq_config");
            int bits = lq.get("loftq_bits") instanceof Number ? ((Number) lq.get("loftq_bits")).intValue() : 4;
            int iter = lq.get("loftq_iter") instanceof Number ? ((Number) lq.get("loftq_iter")).intValue() : 1;
            b.loftQConfig(new LoftQConfig(bits, iter));
        }
        if (d.get("eva_config") instanceof Map) b.evaConfig(EvaConfig.fromMap((Map<String, Object>) d.get("eva_config")));
        if (d.get("corda_config") instanceof Map) b.cordaConfig(CordaConfig.fromMap((Map<String, Object>) d.get("corda_config")));
        if (d.get("lora_ga_config") instanceof Map) b.loraGaConfig(LoraGAConfig.fromMap((Map<String, Object>) d.get("lora_ga_config")));
        if (d.get("velora_config") instanceof Map) b.veloraConfig(VeloraConfig.fromMap((Map<String, Object>) d.get("velora_config")));
        if (d.get("monteclora_config") instanceof Map) b.montecloraConfig(MontecloraConfig.fromMap((Map<String, Object>) d.get("monteclora_config")));
        if (d.get("bd_lora_config") instanceof Map) b.bdLoraConfig(BdLoraConfig.fromMap((Map<String, Object>) d.get("bd_lora_config")));
        if (d.get("arrow_config") instanceof Map) b.arrowConfig(ArrowConfig.fromMap((Map<String, Object>) d.get("arrow_config")));
        if (d.get("runtime_config") instanceof Map) b.runtimeConfig(LoraRuntimeConfig.fromMap((Map<String, Object>) d.get("runtime_config")));
        if (d.get("target_parameters") instanceof List) b.targetParameters((List<String>) d.get("target_parameters"));
        if (d.get("ensure_weight_tying") instanceof Boolean) b.ensureWeightTying((Boolean) d.get("ensure_weight_tying"));
        LoraConfig cfg = b.build();
        // Apply PeftConfig-level fields (peft_type / task_type / ...).
        // Re-build through Builder so we can call the static applyBaseDict.
        PeftConfig.applyBaseDict(b, d);
        return b.build();
    }

    public static Builder builder() { return new Builder(); }

    // ====================================================================================
    //                              NESTED DATACLASSES
    // ====================================================================================

    /** HuggingFace {@code LoraConfig.LoftQConfig}. */
    public static final class LoftQConfig {
        public final int loftqBits;
        public final int loftqIter;
        public LoftQConfig(int loftqBits, int loftqIter) {
            this.loftqBits = loftqBits;
            this.loftqIter = loftqIter;
        }
        public LoftQConfig() { this(4, 1); }
        public boolean nonDefault() { return loftqBits != 4 || loftqIter != 1; }
        public LoftQConfig copy() { return new LoftQConfig(loftqBits, loftqIter); }
    }

    /** HuggingFace {@code LoraConfig.EvaConfig}. */
    public static final class EvaConfig {
        public double rho = 2.0;
        public double tau = 0.99;
        public boolean useLabelMask = true;
        public double labelMaskValue = -100.0;
        public boolean whiten = false;
        public boolean adjustScalingFactors = true;
        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("rho", rho); m.put("tau", tau);
            m.put("use_label_mask", useLabelMask);
            m.put("label_mask_value", labelMaskValue);
            m.put("whiten", whiten);
            m.put("adjust_scaling_factors", adjustScalingFactors);
            return m;
        }
        @SuppressWarnings("unchecked")
        public static EvaConfig fromMap(Map<String, Object> m) {
            EvaConfig c = new EvaConfig();
            if (m.get("rho") instanceof Number) c.rho = ((Number) m.get("rho")).doubleValue();
            if (m.get("tau") instanceof Number) c.tau = ((Number) m.get("tau")).doubleValue();
            if (m.get("use_label_mask") instanceof Boolean) c.useLabelMask = (Boolean) m.get("use_label_mask");
            if (m.get("label_mask_value") instanceof Number) c.labelMaskValue = ((Number) m.get("label_mask_value")).doubleValue();
            if (m.get("whiten") instanceof Boolean) c.whiten = (Boolean) m.get("whiten");
            if (m.get("adjust_scaling_factors") instanceof Boolean) c.adjustScalingFactors = (Boolean) m.get("adjust_scaling_factors");
            return c;
        }
        public EvaConfig copy() { EvaConfig c = new EvaConfig(); c.rho = rho; c.tau = tau; c.useLabelMask = useLabelMask; c.labelMaskValue = labelMaskValue; c.whiten = whiten; c.adjustScalingFactors = adjustScalingFactors; return c; }
    }

    /** HuggingFace {@code LoraConfig.CordaConfig}. */
    public static final class CordaConfig {
        public String cacheFile;
        public String covarianceFile;
        public String cordaMethod = "ipm";
        public boolean verbose = false;
        public boolean useFloat16ForCovariance = false;
        public boolean pruneTemporaryFields = true;
        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            if (cacheFile != null) m.put("cache_file", cacheFile);
            if (covarianceFile != null) m.put("covariance_file", covarianceFile);
            m.put("corda_method", cordaMethod);
            m.put("verbose", verbose);
            m.put("use_float16_for_covariance", useFloat16ForCovariance);
            m.put("prune_temporary_fields", pruneTemporaryFields);
            return m;
        }
        @SuppressWarnings("unchecked")
        public static CordaConfig fromMap(Map<String, Object> m) {
            CordaConfig c = new CordaConfig();
            if (m.get("cache_file") instanceof String) c.cacheFile = (String) m.get("cache_file");
            if (m.get("covariance_file") instanceof String) c.covarianceFile = (String) m.get("covariance_file");
            if (m.get("corda_method") instanceof String) c.cordaMethod = (String) m.get("corda_method");
            if (m.get("verbose") instanceof Boolean) c.verbose = (Boolean) m.get("verbose");
            if (m.get("use_float16_for_covariance") instanceof Boolean) c.useFloat16ForCovariance = (Boolean) m.get("use_float16_for_covariance");
            if (m.get("prune_temporary_fields") instanceof Boolean) c.pruneTemporaryFields = (Boolean) m.get("prune_temporary_fields");
            return c;
        }
        public CordaConfig copy() { CordaConfig c = new CordaConfig(); c.cacheFile = cacheFile; c.covarianceFile = covarianceFile; c.cordaMethod = cordaMethod; c.verbose = verbose; c.useFloat16ForCovariance = useFloat16ForCovariance; c.pruneTemporaryFields = pruneTemporaryFields; return c; }
    }

    /** HuggingFace {@code LoraConfig.LoraGAConfig}. */
    public static final class LoraGAConfig {
        public String direction = "ArB2r";
        public String scale = "stable";
        public int stableGamma = 16;
        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("direction", direction); m.put("scale", scale); m.put("stable_gamma", stableGamma);
            return m;
        }
        @SuppressWarnings("unchecked")
        public static LoraGAConfig fromMap(Map<String, Object> m) {
            LoraGAConfig c = new LoraGAConfig();
            if (m.get("direction") instanceof String) c.direction = (String) m.get("direction");
            if (m.get("scale") instanceof String) c.scale = (String) m.get("scale");
            if (m.get("stable_gamma") instanceof Number) c.stableGamma = ((Number) m.get("stable_gamma")).intValue();
            return c;
        }
        public LoraGAConfig copy() { LoraGAConfig c = new LoraGAConfig(); c.direction = direction; c.scale = scale; c.stableGamma = stableGamma; return c; }
    }

    /** HuggingFace {@code LoraConfig.VeloraConfig}. */
    public static final class VeloraConfig {
        public int numGroups = 64;
        public double scale = 1.0;
        public String initType = "batch_average";
        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("num_groups", numGroups); m.put("scale", scale); m.put("init_type", initType);
            return m;
        }
        @SuppressWarnings("unchecked")
        public static VeloraConfig fromMap(Map<String, Object> m) {
            VeloraConfig c = new VeloraConfig();
            if (m.get("num_groups") instanceof Number) c.numGroups = ((Number) m.get("num_groups")).intValue();
            if (m.get("scale") instanceof Number) c.scale = ((Number) m.get("scale")).doubleValue();
            if (m.get("init_type") instanceof String) c.initType = (String) m.get("init_type");
            return c;
        }
        public VeloraConfig copy() { VeloraConfig c = new VeloraConfig(); c.numGroups = numGroups; c.scale = scale; c.initType = initType; return c; }
    }

    /** HuggingFace {@code MontecloraConfig}. */
    public static final class MontecloraConfig {
        public int numSamples = 8;
        public boolean useEntropy = false;
        public double dirichletPrior = 0.1;
        public double sampleScaler = 1e-4;
        public double klLossWeight = 1e-5;
        public int bufferSize = 150;
        public Map<String, Object> toMap() { Map<String, Object> m = new LinkedHashMap<>(); m.put("num_samples", numSamples); m.put("use_entropy", useEntropy); m.put("dirichlet_prior", dirichletPrior); m.put("sample_scaler", sampleScaler); m.put("kl_loss_weight", klLossWeight); m.put("buffer_size", bufferSize); return m; }
        @SuppressWarnings("unchecked")
        public static MontecloraConfig fromMap(Map<String, Object> m) {
            MontecloraConfig c = new MontecloraConfig();
            if (m.get("num_samples") instanceof Number) c.numSamples = ((Number) m.get("num_samples")).intValue();
            if (m.get("use_entropy") instanceof Boolean) c.useEntropy = (Boolean) m.get("use_entropy");
            if (m.get("dirichlet_prior") instanceof Number) c.dirichletPrior = ((Number) m.get("dirichlet_prior")).doubleValue();
            if (m.get("sample_scaler") instanceof Number) c.sampleScaler = ((Number) m.get("sample_scaler")).doubleValue();
            if (m.get("kl_loss_weight") instanceof Number) c.klLossWeight = ((Number) m.get("kl_loss_weight")).doubleValue();
            if (m.get("buffer_size") instanceof Number) c.bufferSize = ((Number) m.get("buffer_size")).intValue();
            return c;
        }
        public MontecloraConfig copy() { MontecloraConfig c = new MontecloraConfig(); c.numSamples = numSamples; c.useEntropy = useEntropy; c.dirichletPrior = dirichletPrior; c.sampleScaler = sampleScaler; c.klLossWeight = klLossWeight; c.bufferSize = bufferSize; return c; }
    }

    /** HuggingFace {@code BdLoraConfig}. */
    public static final class BdLoraConfig {
        public List<String> targetModulesBdA;
        public List<String> targetModulesBdB;
        public int nblocks = 1;
        public boolean matchStrict = true;
        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            if (targetModulesBdA != null) m.put("target_modules_bd_a", new ArrayList<>(targetModulesBdA));
            if (targetModulesBdB != null) m.put("target_modules_bd_b", new ArrayList<>(targetModulesBdB));
            m.put("nblocks", nblocks); m.put("match_strict", matchStrict);
            return m;
        }
        @SuppressWarnings("unchecked")
        public static BdLoraConfig fromMap(Map<String, Object> m) {
            BdLoraConfig c = new BdLoraConfig();
            if (m.get("target_modules_bd_a") instanceof List) c.targetModulesBdA = new ArrayList<>((List<String>) m.get("target_modules_bd_a"));
            if (m.get("target_modules_bd_b") instanceof List) c.targetModulesBdB = new ArrayList<>((List<String>) m.get("target_modules_bd_b"));
            if (m.get("nblocks") instanceof Number) c.nblocks = ((Number) m.get("nblocks")).intValue();
            if (m.get("match_strict") instanceof Boolean) c.matchStrict = (Boolean) m.get("match_strict");
            return c;
        }
        public BdLoraConfig copy() { BdLoraConfig c = new BdLoraConfig(); c.targetModulesBdA = targetModulesBdA == null ? null : new ArrayList<>(targetModulesBdA); c.targetModulesBdB = targetModulesBdB == null ? null : new ArrayList<>(targetModulesBdB); c.nblocks = nblocks; c.matchStrict = matchStrict; return c; }
    }

    /** HuggingFace {@code ArrowConfig}. */
    public static final class ArrowConfig {
        public int topK = 3;
        public double routerTemperature = 1.0;
        public boolean useGks = false;
        public Long rngSeed = null;
        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("top_k", topK);
            m.put("router_temperature", routerTemperature);
            m.put("use_gks", useGks);
            if (rngSeed != null) m.put("rng_seed", rngSeed);
            return m;
        }
        @SuppressWarnings("unchecked")
        public static ArrowConfig fromMap(Map<String, Object> m) {
            ArrowConfig c = new ArrowConfig();
            if (m.get("top_k") instanceof Number) c.topK = ((Number) m.get("top_k")).intValue();
            if (m.get("router_temperature") instanceof Number) c.routerTemperature = ((Number) m.get("router_temperature")).doubleValue();
            if (m.get("use_gks") instanceof Boolean) c.useGks = (Boolean) m.get("use_gks");
            if (m.get("rng_seed") instanceof Number) c.rngSeed = ((Number) m.get("rng_seed")).longValue();
            return c;
        }
        public ArrowConfig copy() { ArrowConfig c = new ArrowConfig(); c.topK = topK; c.routerTemperature = routerTemperature; c.useGks = useGks; c.rngSeed = rngSeed; return c; }
    }

    /** HuggingFace {@code LoraConfig.runtime_config} (ephemeral GPU offload, etc.). */
    public static final class LoraRuntimeConfig {
        public boolean ephemeralGpuOffload = false;
        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("ephemeral_gpu_offload", ephemeralGpuOffload);
            return m;
        }
        @SuppressWarnings("unchecked")
        public static LoraRuntimeConfig fromMap(Map<String, Object> m) {
            LoraRuntimeConfig c = new LoraRuntimeConfig();
            if (m.get("ephemeral_gpu_offload") instanceof Boolean) c.ephemeralGpuOffload = (Boolean) m.get("ephemeral_gpu_offload");
            return c;
        }
        public LoraRuntimeConfig copy() { LoraRuntimeConfig c = new LoraRuntimeConfig(); c.ephemeralGpuOffload = ephemeralGpuOffload; return c; }
    }

    // ====================================================================================
    //                                  BUILDER
    // ====================================================================================

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
        private int loraBias = 0;
        private boolean allLinear = false;
        private int cordaConfigId = -1;
        private int montecloraConfigId = -1;

        private LoftQConfig loftQConfig = null;
        private EvaConfig evaConfig = null;
        private CordaConfig cordaConfig = null;
        private LoraGAConfig loraGaConfig = null;
        private VeloraConfig veloraConfig = null;
        private MontecloraConfig montecloraConfig = null;
        private BdLoraConfig bdLoraConfig = null;
        private ArrowConfig arrowConfig = null;
        private LoraRuntimeConfig runtimeConfig = null;

        private List<String> targetParameters = null;
        private boolean useBdLora = false;
        private boolean useArrow = false;
        private boolean useMonteclora = false;
        private boolean useQalora = false;
        private int qaloraGroupSize = 16;
        private List<Integer> aloraInvocationTokens = null;
        private List<List<Integer>> layerReplication = null;
        private List<Integer> trainableTokenIndices = null;
        private Map<String, List<Integer>> trainableTokenIndicesByAdapter = null;
        private boolean ensureWeightTying = false;

        public Builder() { peftType(PeftType.LORA); }

        public Builder r(int r) { this.r = r; return this; }
        public Builder alpha(double alpha) { this.alpha = alpha; return this; }
        public Builder lora_alpha(double alpha) { return alpha(alpha); }
        public Builder loraAlpha(double alpha) { return alpha(alpha); }
        public Builder dropout(double dropout) { this.dropout = dropout; return this; }
        public Builder lora_dropout(double dropout) { return dropout(dropout); }
        public Builder loraDropout(double dropout) { return dropout(dropout); }

        public Builder targetModules(String... modules) {
            if (modules != null && modules.length == 1 && isAllLinearSpec(modules[0])) {
                return allLinear(true);
            }
            this.allLinear = false;
            this.targetModules = new ArrayList<>(Arrays.asList(modules));
            return this;
        }
        public Builder targetModules(String spec) {
            if (isAllLinearSpec(spec)) return allLinear(true);
            this.allLinear = false;
            this.targetModules = new ArrayList<>(List.of(spec));
            return this;
        }
        public Builder allLinear(boolean v) {
            this.allLinear = v;
            if (v) this.targetModules = new ArrayList<>();
            return this;
        }
        public Builder all_linear(boolean v) { return allLinear(v); }
        public Builder target_modules(String... modules) { return targetModules(modules); }
        public Builder targetModules(List<String> modules) { this.targetModules = new ArrayList<>(Objects.requireNonNull(modules)); return this; }
        public Builder target_modules(List<String> modules) { return targetModules(modules); }
        public Builder excludeModules(String... modules) { this.excludeModules = new ArrayList<>(Arrays.asList(modules)); return this; }
        public Builder exclude_modules(String... modules) { return excludeModules(modules); }
        public Builder excludeModules(List<String> modules) { this.excludeModules = new ArrayList<>(Objects.requireNonNull(modules)); return this; }
        public Builder exclude_modules(List<String> modules) { return excludeModules(modules); }
        public Builder freezeBase(boolean v) { this.freezeBase = v; return this; }
        public Builder useRslora(boolean v) { this.useRslora = v; return this; }
        public Builder use_rslora(boolean v) { return useRslora(v); }
        public Builder bias(String b) { this.bias = b != null ? b : "none"; return this; }
        public Builder fanInFanOut(boolean v) { this.fanInFanOut = v; return this; }
        public Builder fan_in_fan_out(boolean v) { return fanInFanOut(v); }
        public Builder modulesToSave(String... modules) { this.modulesToSave = new ArrayList<>(Arrays.asList(modules)); return this; }
        public Builder modules_to_save(String... modules) { return modulesToSave(modules); }
        public Builder modulesToSave(List<String> modules) { this.modulesToSave = new ArrayList<>(Objects.requireNonNull(modules)); return this; }
        public Builder modules_to_save(List<String> modules) { return modulesToSave(modules); }
        public Builder initLoraWeights(String v) { this.initLoraWeights = v == null ? "true" : v; return this; }
        public Builder init_lora_weights(String v) { return initLoraWeights(v); }
        public Builder layersToTransform(Integer... indices) {
            this.layersToTransform = indices.length == 1 ? (Object) indices[0] : new ArrayList<>(Arrays.asList(indices));
            return this;
        }
        public Builder layers_to_transform(Integer... indices) { return layersToTransform(indices); }
        public Builder layersToTransform(List<Integer> indices) {
            this.layersToTransform = indices.size() == 1 ? (Object) indices.get(0) : new ArrayList<>(indices);
            return this;
        }
        public Builder layers_to_transform(List<Integer> indices) { return layersToTransform(indices); }
        public Builder layersPattern(String... patterns) {
            this.layersPattern = new ArrayList<>(Arrays.asList(patterns));
            return this;
        }
        public Builder layersPattern(java.util.List<String> patterns) {
            this.layersPattern = new ArrayList<>(patterns);
            return this;
        }
        public Builder layers_pattern(String... patterns) { return layersPattern(patterns); }
        public Builder rankPattern(Map<String, Integer> v) { this.rankPattern = v; return this; }
        public Builder rank_pattern(Map<String, Integer> v) { return rankPattern(v); }
        public Builder alphaPattern(Map<String, Integer> v) { this.alphaPattern = v; return this; }
        public Builder alpha_pattern(Map<String, Integer> v) { return alphaPattern(v); }
        public Builder useDora(boolean v) { this.useDora = v; return this; }
        public Builder use_dora(boolean v) { return useDora(v); }
        public Builder loraBias(boolean v) { this.loraBias = v ? 1 : 0; return this; }
        public Builder lora_bias(boolean v) { return loraBias(v); }
        public Builder cordaConfigId(int v) { this.cordaConfigId = v; return this; }
        public Builder montecloraConfigId(int v) { this.montecloraConfigId = v; return this; }

        public Builder loftQConfig(LoftQConfig v) { this.loftQConfig = v; return this; }
        public Builder loftqConfig(LoftQConfig v) { return loftQConfig(v); }
        public Builder loftq_config(LoftQConfig v) { return loftQConfig(v); }
        public Builder evaConfig(EvaConfig v) { this.evaConfig = v; return this; }
        public Builder eva_config(EvaConfig v) { return evaConfig(v); }
        public Builder cordaConfig(CordaConfig v) { this.cordaConfig = v; return this; }
        public Builder corda_config(CordaConfig v) { return cordaConfig(v); }
        public Builder loraGaConfig(LoraGAConfig v) { this.loraGaConfig = v; return this; }
        public Builder lora_ga_config(LoraGAConfig v) { return loraGaConfig(v); }
        public Builder veloraConfig(VeloraConfig v) { this.veloraConfig = v; return this; }
        public Builder velora_config(VeloraConfig v) { return veloraConfig(v); }
        public Builder montecloraConfig(MontecloraConfig v) { this.montecloraConfig = v; return this; }
        public Builder monteclora_config(MontecloraConfig v) { return montecloraConfig(v); }
        public Builder bdLoraConfig(BdLoraConfig v) { this.bdLoraConfig = v; return this; }
        public Builder bd_lora_config(BdLoraConfig v) { return bdLoraConfig(v); }
        public Builder arrowConfig(ArrowConfig v) { this.arrowConfig = v; return this; }
        public Builder arrow_config(ArrowConfig v) { return arrowConfig(v); }
        public Builder runtimeConfig(LoraRuntimeConfig v) { this.runtimeConfig = v; return this; }
        public Builder runtime_config(LoraRuntimeConfig v) { return runtimeConfig(v); }

        public Builder targetParameters(List<String> v) { this.targetParameters = v; return this; }
        public Builder target_parameters(List<String> v) { return targetParameters(v); }
        public Builder useBdLora(boolean v) { this.useBdLora = v; return this; }
        public Builder use_bdlora(boolean v) { return useBdLora(v); }
        public Builder useArrow(boolean v) { this.useArrow = v; return this; }
        public Builder use_arrow(boolean v) { return useArrow(v); }
        public Builder useMonteclora(boolean v) { this.useMonteclora = v; return this; }
        public Builder use_monteclora(boolean v) { return useMonteclora(v); }
        public Builder useQalora(boolean v) { this.useQalora = v; return this; }
        public Builder use_qalora(boolean v) { return useQalora(v); }
        public Builder qaloraGroupSize(int v) { this.qaloraGroupSize = v; return this; }
        public Builder qalora_group_size(int v) { return qaloraGroupSize(v); }
        public Builder aloraInvocationTokens(List<Integer> v) { this.aloraInvocationTokens = v; return this; }
        public Builder alora_invocation_tokens(List<Integer> v) { return aloraInvocationTokens(v); }
        public Builder layerReplication(List<List<Integer>> v) { this.layerReplication = v; return this; }
        public Builder layer_replication(List<List<Integer>> v) { return layerReplication(v); }
        public Builder trainableTokenIndices(List<Integer> v) { this.trainableTokenIndices = v; return this; }
        public Builder trainable_token_indices(List<Integer> v) { return trainableTokenIndices(v); }
        public Builder trainableTokenIndicesByAdapter(Map<String, List<Integer>> v) { this.trainableTokenIndicesByAdapter = v; return this; }
        public Builder trainable_token_indices_by_adapter(Map<String, List<Integer>> v) { return trainableTokenIndicesByAdapter(v); }
        public Builder ensureWeightTying(boolean v) { this.ensureWeightTying = v; return this; }
        public Builder ensure_weight_tying(boolean v) { return ensureWeightTying(v); }

        private static boolean isAllLinearSpec(String spec) {
            if (spec == null) return false;
            String s = spec.trim().toLowerCase();
            return "all-linear".equals(s) || "all_linear".equals(s);
        }

        public Builder task_type(String taskType) { return taskType(taskType); }

        @Override
        public LoraConfig build() { return new LoraConfig(this); }
    }
}