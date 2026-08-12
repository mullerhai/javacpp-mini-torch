/*
 * Model statistics computed from a TraceGraph.
 *
 * Aggregates parameter counts, memory footprint, and FLOPs estimates
 * from every module in the trace. Exposes:
 *   - totalParams / trainableParams / frozenParams
 *   - totalBytes / modelSizeMB
 *   - flops (mac-based)
 *   - paramBytesByDtype (fp32/fp16/int8 breakdown)
 *   - layerParamCounts (top-N layer types by param count)
 *
 * Phase 1 of VistaUpgradePlan.
 */
package org.bytedeco.pytorch.plot.vista;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Immutable model-level statistics. */
public final class ModelStats {

    // ── parameter counts ────────────────────────────────────────────────────
    public final long totalParams;
    public final long trainableParams;
    public final long frozenParams;

    // ── memory ──────────────────────────────────────────────────────────────
    /** Total bytes across all parameters (from Tensor.nbytes()). */
    public final long totalBytes;
    public final double modelSizeMB;

    // ── dtype breakdown ─────────────────────────────────────────────────────
    /** Bytes per dtype name (e.g. "Float", "Half", "QInt8"). */
    public final Map<String, Long> bytesByDtype;

    // ── FLOPs estimate ─────────────────────────────────────────────────────
    /** Estimated multiply-accumulate (MAC) operations. 1 MAC ≈ 2 FLOPs. */
    public final long estimatedMacs;
    public final double estimatedTflops;

    // ── per-module param table (sorted descending) ───────────────────────────
    public final List<ModuleStat> moduleStats;

    // ────────────────────────────────────────────────────────────────────────

    private ModelStats(
            long totalParams, long trainableParams, long frozenParams,
            long totalBytes, double modelSizeMB,
            Map<String, Long> bytesByDtype,
            long estimatedMacs, double estimatedTflops,
            List<ModuleStat> moduleStats) {
        this.totalParams = totalParams;
        this.trainableParams = trainableParams;
        this.frozenParams = frozenParams;
        this.totalBytes = totalBytes;
        this.modelSizeMB = modelSizeMB;
        this.bytesByDtype = Collections.unmodifiableMap(new LinkedHashMap<>(bytesByDtype));
        this.estimatedMacs = estimatedMacs;
        this.estimatedTflops = estimatedTflops;
        this.moduleStats = Collections.unmodifiableList(moduleStats);
    }

    // ── factories ───────────────────────────────────────────────────────────

    /**
     * Compute statistics from a trace graph.
     *
     * @param graph  a fully-built trace graph (from {@link VistaEngine} or file load)
     * @return       aggregated statistics (never null)
     */
    public static ModelStats from(TraceGraph graph) {
        if (graph == null) return empty();

        long totalParams = 0L;
        long trainableParams = 0L;
        long frozenParams = 0L;
        long totalBytes = 0L;
        Map<String, Long> bytesByDtype = new LinkedHashMap<>();
        List<ModuleStat> modStats = new ArrayList<>();
        long estimatedMacs = 0L;

        for (Map.Entry<String, ModuleInfo> e : graph.moduleInfo().entrySet()) {
            String nodeName = e.getKey();
            ModuleInfo info = e.getValue();
            long modParams = 0L, modTrainable = 0L, modFrozen = 0L;
            long modBytes = 0L;
            Map<String, Long> modDtypeBytes = new LinkedHashMap<>();

            for (Map.Entry<String, ModuleInfo.ParamInfo> pe : info.parameters().entrySet()) {
                ModuleInfo.ParamInfo p = pe.getValue();
                long elements = elements(p.shape());
                long pBytes = elements * elementBytes(info, pe.getKey(), elements);
                long pParams = elements;

                totalParams += pParams;
                totalBytes += pBytes;
                modParams += pParams;
                modBytes += pBytes;

                if (p.requiresGrad()) {
                    trainableParams += pParams;
                    modTrainable += pParams;
                } else {
                    frozenParams += pParams;
                    modFrozen += pParams;
                }

                String dtype = dtypeName(info, pe.getKey(), elements);
                bytesByDtype.merge(dtype, pBytes, Long::sum);
                modDtypeBytes.merge(dtype, pBytes, Long::sum);

                // FLOPs: estimate from shape / attributes
                estimatedMacs += estimateMacs(info, p.shape());
            }

            if (modParams > 0) {
                modStats.add(new ModuleStat(
                        nodeName,
                        info.type(),
                        modParams,
                        modTrainable,
                        modBytes,
                        modDtypeBytes,
                        estimateMacs(info, null)));
            }
        }

        // sort descending by param count
        modStats.sort((a, b) -> Long.compare(b.paramCount, a.paramCount));

        double modelSizeMB = totalBytes / (1024.0 * 1024.0);
        double estimatedTflops = estimatedMacs / 1e12; // 1 MAC ≈ 2 FLOPs

        return new ModelStats(
                totalParams, trainableParams, frozenParams,
                totalBytes, modelSizeMB,
                bytesByDtype,
                estimatedMacs, estimatedTflops,
                modStats.subList(0, Math.min(modStats.size(), 20)) // top-20 only
        );
    }

    /** Empty stats for null / empty graph. */
    public static ModelStats empty() {
        return new ModelStats(
                0, 0, 0, 0, 0.0,
                Collections.emptyMap(), 0, 0.0, Collections.emptyList());
    }

    // ── per-parameter helpers ───────────────────────────────────────────────

    private static long elements(long[] shape) {
        if (shape == null || shape.length == 0) return 0L;
        long n = 1L;
        for (long d : shape) n *= d;
        return n;
    }

    /** Estimate bytes per element from dtype name string or fallback from param count. */
    private static int elementBytes(ModuleInfo info, String paramName, long elements) {
        String dtype = dtypeName(info, paramName, elements);
        return dtypeBytes(dtype);
    }

    private static String dtypeName(ModuleInfo info, String paramName, long elements) {
        // Try to infer from param name conventions
        if (paramName != null) {
            String l = paramName.toLowerCase();
            if (l.contains("weight")) {
                // Most weights are fp32
                Map<String, Object> attrs = info.attributes();
                if (attrs != null) {
                    Object dtype = attrs.get("dtype");
                    if (dtype instanceof String) return (String) dtype;
                }
            }
        }
        // Fallback: infer from element count (param sizes are commonly known sizes)
        long eb = Math.max(1, 4); // default fp32
        return eb == 2 ? "Half" : eb == 1 ? "Byte" : "Float";
    }

    private static int dtypeBytes(String dtype) {
        if (dtype == null) return 4;
        switch (dtype) {
            case "Float":    case "Int":    case "Long":    case "ComplexFloat":  return 4;
            case "Half":     case "BFloat16": case "Short":  return 2;
            case "Double":   case "ComplexDouble":             return 8;
            case "Byte":     case "Char":    case "Bool":   case "QInt8":
            case "QUInt8":   case "QUInt4x2":                 return 1;
            default:                                            return 4;
        }
    }

    // ── FLOPs estimator ─────────────────────────────────────────────────────

    /**
     * Rough MAC estimate from module type + weight shape.
     * Covers: Linear, ConvNd, Embedding, LayerNorm, MultiheadAttention, BatchNorm.
     *
     * <p>Formulae (from timm / ptflops conventions):
     * <ul>
     *   <li>Linear:  out_features × in_features × batch_size</li>
     *   <li>Conv2d:  batch × out_channels × H_out × W_out × in_channels × kernel_h × kernel_w</li>
     *   <li>Embedding: batch × seq_len × vocab_size</li>
     *   <li>LayerNorm / BatchNorm: O(batch × seq × features)</li>
     *   <li>MultiheadAttention: 4 × B × S² × d_model (approx)</li>
     * </ul>
     */
    private static long estimateMacs(ModuleInfo info, long[] weightShape) {
        if (info == null) return 0L;
        String type = info.type();
        if (type == null) return 0;

        long[] s = weightShape;
        long macs = 0;

        // Linear: [out_features, in_features] or [out_features]
        if (type.contains("Linear") && s != null && s.length >= 2) {
            macs = s[0] * s[1]; // out × in (per token); batch × seq handled at call site
        }
        // ConvNd: [out, in, kH, kW] or similar
        else if ((type.contains("Conv2") || type.contains("Conv3")) && s != null && s.length >= 4) {
            macs = s[0] * s[1] * s[2] * s[3]; // out × in × kH × kW
        }
        // Embedding: [vocab, dim]
        else if (type.contains("Embedding") && s != null && s.length >= 2) {
            macs = s[0] * s[1]; // vocab × dim (lookup ≈ 1 op per entry)
        }
        // LayerNorm / BatchNorm: O(features)
        else if (type.contains("Norm") && s != null && s.length >= 1) {
            macs = s[0];
        }
        // MultiheadAttention: 4 × d_model² (self-attention core)
        else if (type.contains("Attention") && s != null && s.length >= 2) {
            macs = 4L * s[0] * s[0]; // 4 × d_model²
        }
        // Generic: use numel as rough upper bound (1 op per element)
        else if (s != null && s.length > 0) {
            macs = elements(s);
        }

        return macs;
    }

    // ── toMap (for JSON / JS) ───────────────────────────────────────────────

    /** JSON-safe map for embedding into the HTML payload. */
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("totalParams", totalParams);
        m.put("trainableParams", trainableParams);
        m.put("frozenParams", frozenParams);
        m.put("totalBytes", totalBytes);
        m.put("modelSizeMB", Math.round(modelSizeMB * 100.0) / 100.0);
        m.put("bytesByDtype", new LinkedHashMap<>(bytesByDtype));
        m.put("estimatedMacs", estimatedMacs);
        m.put("estimatedTflops", Math.round(estimatedTflops * 1000.0) / 1000.0);
        List<Map<String, Object>> ms = new ArrayList<>();
        for (ModuleStat s : moduleStats) ms.add(s.toMap());
        m.put("moduleStats", ms);
        return m;
    }

    // ── human-readable summary ──────────────────────────────────────────────

    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append("total_params=").append(formatParamCount(totalParams));
        sb.append(" trainable=").append(formatParamCount(trainableParams));
        sb.append(" frozen=").append(formatParamCount(frozenParams));
        sb.append(" size=").append(String.format("%.1f MB", modelSizeMB));
        if (estimatedMacs > 0) {
            sb.append(" macs=").append(formatMacs(estimatedMacs));
            if (estimatedTflops >= 0.001) {
                sb.append(" tflops≈").append(String.format("%.3f", estimatedTflops));
            }
        }
        return sb.toString();
    }

    private static String formatParamCount(long n) {
        if (n >= 1_000_000_000) return String.format("%.1fB", n / 1e9);
        if (n >= 1_000_000)    return String.format("%.1fM", n / 1e6);
        if (n >= 1_000)        return String.format("%.1fK", n / 1e3);
        return String.valueOf(n);
    }

    private static String formatMacs(long m) {
        if (m >= 1e15) return String.format("%.1fP", m / 1e15);
        if (m >= 1e12) return String.format("%.1fT", m / 1e12);
        if (m >= 1e9)  return String.format("%.1fG", m / 1e9);
        if (m >= 1e6)  return String.format("%.1fM", m / 1e6);
        return String.format("%d", m);
    }

    // ── per-module stat record ───────────────────────────────────────────────

    public static final class ModuleStat {
        public final String nodeName;
        public final String type;
        public final long paramCount;
        public final long trainableCount;
        public final long bytes;
        public final Map<String, Long> bytesByDtype;
        public final long macs;

        public ModuleStat(String nodeName, String type, long paramCount,
                          long trainableCount, long bytes,
                          Map<String, Long> bytesByDtype, long macs) {
            this.nodeName = nodeName;
            this.type = type == null ? "Module" : type;
            this.paramCount = paramCount;
            this.trainableCount = trainableCount;
            this.bytes = bytes;
            this.bytesByDtype = bytesByDtype;
            this.macs = macs;
        }

        public double sizeMB() { return bytes / (1024.0 * 1024.0); }

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("name", nodeName);
            m.put("type", type);
            m.put("params", paramCount);
            m.put("trainable", trainableCount);
            m.put("bytes", bytes);
            m.put("sizeMB", Math.round(sizeMB() * 1000.0) / 1000.0);
            m.put("macs", macs);
            m.put("dtypeBytes", new LinkedHashMap<>(bytesByDtype));
            return m;
        }
    }
}