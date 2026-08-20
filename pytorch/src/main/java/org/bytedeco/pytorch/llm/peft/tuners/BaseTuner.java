/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License").
 */
package org.bytedeco.pytorch.llm.peft.tuners;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDict;
import org.bytedeco.pytorch.llm.peft.PeftConfig;
import org.bytedeco.pytorch.llm.peft.PeftType;
import org.bytedeco.pytorch.llm.peft.PeftWarning;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Java analog of HuggingFace {@code peft.tuners.tuners_utils.BaseTuner}.
 *
 * <p>Walks a host model, identifies target modules matching the configuration,
 * and replaces each with a tuner-layer subclass via {@link #_createNewLayer}.
 * Concrete tuners (Lora, IA3, OFT, etc.) extend this base and override the
 * layer-construction hook.
 */
public abstract class BaseTuner extends Module {

    protected final Module model;
    protected final Map<String, PeftConfig> peftConfig;
    protected final String activeAdapter;
    protected final String prefix;

    protected final Map<String, BaseTunerLayer> tunerLayers = new LinkedHashMap<>();
    protected final List<Pattern> targetPatterns = new ArrayList<>();
    protected final List<Pattern> excludePatterns = new ArrayList<>();
    protected final List<Integer> layers = new ArrayList<>(java.util.Arrays.asList(-1));
    protected Object targetModulesSpec;
    protected BaseTuner(Module model, Map<String, PeftConfig> peftConfig, String adapterName, String prefix) {
        super();
        this.model = model;
        this.peftConfig = peftConfig;
        this.activeAdapter = adapterName;
        this.prefix = prefix;
    }

    public void injectAdapter() {
        PeftConfig cfg = peftConfig.get(activeAdapter);
        Map<String, Object> keyList = _checkTargetModuleExists(cfg);
        keyList = _checkMatch(cfg, keyList);
        targetModulesSpec = keyList.get("target_modules");
        for (Object key : toStringList(keyList.get("target_modules"))) {
            targetPatterns.add(compilePattern(String.valueOf(key)));
        }
        for (Object key : toStringList(keyList.get("exclude_modules"))) {
            excludePatterns.add(compilePattern(String.valueOf(key)));
        }
        layers.clear();
        layers.addAll(resolveLayersToTransform(cfg));
        _createAndReplace(cfg);
        _freezeAdapter();
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    protected void _createAndReplace(PeftConfig cfg) {
        StringSharedModuleDict nm = model.named_modules();
        List<String> names = new ArrayList<>();
        long n = nm.size();
        for (long i = 0; i < n; i++) {
            org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDictItem item = nm.get(i);
            names.add(item.key().getString());
        }
        for (String name : names) {
            Module target = nm.get(name);
            if (target == null) continue;
            if (!_checkTargetModuleExists(name)) continue;
            if (!_checkLayerIndex(name)) continue;
            if (!_checkMatchPattern(name)) continue;
            BaseTunerLayer newLayer = _createNewLayer(cfg, activeAdapter, name, target);
            if (newLayer == null) continue;
            String parentPath = parentPath(name);
            String leafName = leafName(name);
            if (parentPath.isEmpty()) continue;
            Module parent = nm.get(parentPath);
            if (parent == null) continue;
            parent.register_module(leafName, newLayer);
            tunerLayers.put(name, newLayer);
        }
        _postProcess(cfg);
    }

    protected abstract BaseTunerLayer _createNewLayer(PeftConfig cfg, String adapterName,
                                                       String targetName, Module target);

    public static Pattern compilePattern(String key) {
        if (key == null || key.isEmpty()) return Pattern.compile(".*");
        if (key.equals("all-linear")) return Pattern.compile(".*linear.*");
        String regex = key.replace(".", "\\.").replace("**", ".*").replace("*", "[^.]*");
        return Pattern.compile("^" + regex + "$");
    }

    protected static List<String> toStringList(Object o) {
        if (o == null) return new ArrayList<>();
        if (o instanceof List) return new ArrayList<>((List<String>) o);
        return new ArrayList<>(java.util.Arrays.asList(String.valueOf(o)));
    }

    protected boolean _checkTargetModuleExists(String name) {
        for (Pattern p : targetPatterns) {
            if (p.matcher(name).find()) return true;
        }
        return false;
    }

    protected boolean _checkMatchPattern(String name) {
        for (Pattern p : excludePatterns) {
            if (p.matcher(name).find()) return false;
        }
        return true;
    }

    protected boolean _checkLayerIndex(String name) {
        if (layers == null || layers.isEmpty()) return true;
        Integer idx = parseLayerIndex(name);
        if (idx == null) return false;
        return layers.contains(idx);
    }

    public static Integer parseLayerIndex(String name) {
        Matcher m = Pattern.compile(".*\\.layers\\.(\\d+)\\..*").matcher(name);
        if (m.find()) return Integer.parseInt(m.group(1));
        return null;
    }

    @SuppressWarnings("unchecked")
    protected List<Integer> resolveLayersToTransform(PeftConfig cfg) {
        Object ltt = null;
        try { ltt = cfg.getClass().getMethod("layersToTransform").invoke(cfg); } catch (Exception ignored) {}
        if (ltt == null) return new ArrayList<>();
        if (ltt instanceof Number) {
            int k = ((Number) ltt).intValue();
            List<Integer> result = new ArrayList<>();
            int maxLayers = maxLayerIndex() + 1;
            for (int i = k; i < maxLayers; i++) result.add(i);
            return result;
        }
        if (ltt instanceof List) return new ArrayList<>((List<Integer>) ltt);
        throw new IllegalArgumentException("layersToTransform must be int, list, or null");
    }

    protected int maxLayerIndex() {
        StringSharedModuleDict nm = model.named_modules();
        int max = 0;
        long n = nm.size();
        for (long i = 0; i < n; i++) {
            org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDictItem item = nm.get(i);
            String nmName = item.key().getString();
            Integer k = parseLayerIndex(nmName);
            if (k != null && k > max) max = k;
        }
        return max;
    }

    @SuppressWarnings("unchecked")
    protected Map<String, Object> _checkTargetModuleExists(PeftConfig cfg) {
        Object tm = null;
        try { tm = cfg.getClass().getMethod("targetModules").invoke(cfg); } catch (Exception ignored) {}
        if (tm == null) {
            throw new IllegalArgumentException("target_modules must be specified");
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("target_modules", tm);
        try { result.put("exclude_modules", cfg.getClass().getMethod("excludeModules").invoke(cfg)); } catch (Exception ignored) {}
        if ("all-linear".equals(tm)) {
            result.put("target_modules", new ArrayList<>(java.util.Arrays.asList("all-linear")));
        }
        return result;
    }

    protected Map<String, Object> _checkAndConstructTargetModules(PeftConfig cfg) {
        return _checkTargetModuleExists(cfg);
    }

    @SuppressWarnings("unchecked")
    protected Map<String, Object> _checkMatch(PeftConfig cfg, Map<String, Object> keyList) {
        Object rp = null, ap = null;
        try {
            try { rp = cfg.getClass().getMethod("rankPattern").invoke(cfg); } catch (NoSuchMethodException ignored) {}
            try { ap = cfg.getClass().getMethod("alphaPattern").invoke(cfg); } catch (NoSuchMethodException ignored) {}
        } catch (Exception ignored) {}
        if (rp != null && ap != null) {
            List<String> matched = new ArrayList<>();
            for (Object k : toStringList(keyList.get("target_modules"))) {
                Pattern p = compilePattern(String.valueOf(k));
                if (matchesAnyPattern(p, rp) || matchesAnyPattern(p, ap)) matched.add(String.valueOf(k));
            }
            if (matched.isEmpty()) {
                new PeftWarning("rank_pattern/alpha_pattern did not match any target module");
            }
        }
        return keyList;
    }

    @SuppressWarnings("unchecked")
    private static boolean matchesAnyPattern(Pattern p, Object patternDict) {
        if (patternDict == null) return false;
        if (patternDict instanceof Map) {
            for (Object key : ((Map<String, ?>) patternDict).keySet()) {
                if (p.matcher(String.valueOf(key)).find()) return true;
            }
        }
        return false;
    }

    protected void _freezeAdapter() {
        if (model.modules() == null) return;
        org.bytedeco.pytorch.nn.modules.container.SharedModuleVector mods = model.modules();
        long n = mods.size();
        for (long i = 0; i < n; i++) {
            Module m = mods.get(i);
            if (m instanceof BaseTunerLayer) continue;
            if (m.parameters() == null) continue;
            org.bytedeco.pytorch.TensorVector ps = m.parameters();
            long pn = ps.size();
            for (long j = 0; j < pn; j++) {
                Tensor p = ps.get(j);
                if (p != null) p.requires_grad_(false);
            }
        }
    }

    protected void _postProcess(PeftConfig cfg) {
        // no-op default
    }

    public static String parentPath(String name) {
        int idx = name.lastIndexOf('.');
        return idx < 0 ? "" : name.substring(0, idx);
    }

    public static String leafName(String name) {
        int idx = name.lastIndexOf('.');
        return idx < 0 ? name : name.substring(idx + 1);
    }

    public List<Tensor> trainableParameters() {
        List<Tensor> out = new ArrayList<>();
        for (BaseTunerLayer l : tunerLayers.values()) {
            for (Module m : l.adapterModules()) {
                if (m == null || m.parameters() == null) continue;
                org.bytedeco.pytorch.TensorVector ps = m.parameters();
                long pn = ps.size();
                for (long j = 0; j < pn; j++) {
                    Tensor p = ps.get(j);
                    if (p != null && p.requires_grad()) out.add(p);
                }
            }
        }
        return out;
    }

    public long[] getNbTrainableParameters() {
        long trainable = 0, total = 0;
        for (BaseTunerLayer l : tunerLayers.values()) {
            for (Module m : l.adapterModules()) {
                if (m == null || m.parameters() == null) continue;
                org.bytedeco.pytorch.TensorVector ps = m.parameters();
                long pn = ps.size();
                for (long j = 0; j < pn; j++) {
                    Tensor p = ps.get(j);
                    if (p != null) {
                        total += p.numel();
                        if (p.requires_grad()) trainable += p.numel();
                    }
                }
            }
        }
        return new long[]{trainable, total};
    }

    public Module model() { return model; }
    public Map<String, BaseTunerLayer> tunerLayers() { return tunerLayers; }
    public Map<String, PeftConfig> peftConfig() { return peftConfig; }
    public String activeAdapter() { return activeAdapter; }
    public String prefix() { return prefix; }
    public PeftType peftType() {
        return peftConfig.get(activeAdapter).peftType();
    }
}