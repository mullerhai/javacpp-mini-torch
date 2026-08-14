package org.bytedeco.pytorch.plot.vista;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Enhanced Vista for model visualization with comprehensive format support
 * and rich metadata display.
 *
 * <h2>Supported Formats</h2>
 *
 * <h3>Python PyTorch Formats</h3>
 * <ul>
 *   <li>{@code .pt} / {@code .pth} - Python pickle/zip torch format</li>
 *   <li>{@code .pt2} - PyTorch 2.0+ exported model format</li>
 *   <li>{@code .pte} - PyTorch Model Editor / TorchScript format</li>
 *   <li>{@code TorchScript} - JIT compiled models</li>
 * </ul>
 *
 * <h3>JavaCPP/LibTorch Formats</h3>
 * <ul>
 *   <li>{@code .pt} / {@code .pth} - Native LibTorch archive</li>
 *   <li>{@code .pt2} - AOT Inductor compiled model</li>
 *   <li>{@code .aoti} - AOT Inductor package</li>
 * </ul>
 *
 * <h3>Other Formats</h3>
 * <ul>
 *   <li>{@code .safetensors} - HuggingFace safe tensors</li>
 *   <li>{@code .structure.json} - Structure specification</li>
 * </ul>
 *
 * <h2>Usage</h2>
 *
 * <pre>
 * // Basic usage
 * VistaEnhanced.trace("model.pt");
 * VistaEnhanced.trace("model.safetensors");
 *
 * // With sample input for dynamic shapes
 * Tensor input = torch.randn(1, 10);
 * VistaEnhanced.trace("model.pt2", input);
 *
 * // Get metadata
 * ModelMeta meta = VistaEnhanced.getModelMeta("model.pt");
 * System.out.println("Name: " + meta.getName());
 * System.out.println("Params: " + meta.getParamCount());
 * System.out.println("Layers: " + meta.getLayers());
 *
 * // Export to HTML with metadata panel
 * VistaEnhanced.traceWithMeta("model.safetensors", input);
 * </pre>
 */
public final class VistaEnhanced {

    private VistaEnhanced() {}

    // =========================================================================
    // Format Detection
    // =========================================================================

    /**
     * Supported model file formats.
     */
    public enum ModelFormat {
        // Python PyTorch formats
        PYTHON_PT("Python PyTorch (.pt/.pth)"),
        PYTHON_PT2("PyTorch 2.0 Exported (.pt2)"),
        PYTHON_PTE("PyTorch Model Editor (.pte)"),
        PYTHON_TORCHSCRIPT("TorchScript (.pt)"),

        // JavaCPP/LibTorch formats
        JAVACPP_PT("JavaCPP/LibTorch (.pt)"),
        JAVACPP_PT2("AOT Inductor (.pt2)"),
        JAVACPP_AOTI("AOT Inductor Package (.aoti)"),

        // ONNX format
        ONNX("ONNX Runtime (.onnx)"),

        // Other formats
        SAFETENSORS("HuggingFace SafeTensors (.safetensors)"),
        STRUCTURE_JSON("Structure JSON (.structure.json)"),
        UNKNOWN("Unknown Format");

        public final String displayName;
        ModelFormat(String displayName) {
            this.displayName = displayName;
        }
    }

    /**
     * Detect the format of a model file.
     */
    public static ModelFormat detectFormat(File file) throws IOException {
        if (file == null || !file.exists()) {
            return ModelFormat.UNKNOWN;
        }
        ModelInfoParser.ModelFormat format = ModelInfoParser.detectFormat(file);
        return convertFormat(format);
    }

    private static boolean looksLikeAotiPackage(Path dir) throws IOException {
        return org.bytedeco.pytorch.plot.vista.VistaModelFiles.detect(dir.toFile())
               == org.bytedeco.pytorch.plot.vista.VistaModelFiles.Kind.AOTI_PACKAGE;
    }

    // =========================================================================
    // Model Metadata
    // =========================================================================

    /**
     * Model metadata containing name, parameters, layers, etc.
     */
    public static class ModelMeta {
        private final String name;
        private final ModelFormat format;
        private final long paramCount;
        private final long layerCount;
        private final List<LayerInfo> layers;
        private final Map<String, Object> extra;
        private final long fileSize;
        private final String sourcePath;

        public ModelMeta(String name, ModelFormat format, long paramCount, long layerCount,
                        List<LayerInfo> layers, Map<String, Object> extra, long fileSize, String sourcePath) {
            this.name = name;
            this.format = format;
            this.paramCount = paramCount;
            this.layerCount = layerCount;
            this.layers = layers;
            this.extra = extra;
            this.fileSize = fileSize;
            this.sourcePath = sourcePath;
        }

        public String getName() { return name; }
        public ModelFormat getFormat() { return format; }
        public long getParamCount() { return paramCount; }
        public long getLayerCount() { return layerCount; }
        public List<LayerInfo> getLayers() { return layers; }
        public Map<String, Object> getExtra() { return extra; }
        public long getFileSize() { return fileSize; }
        public String getSourcePath() { return sourcePath; }

        public String getFormattedParamCount() {
            if (paramCount >= 1_000_000_000) {
                return String.format("%.2fB", paramCount / 1_000_000_000.0);
            } else if (paramCount >= 1_000_000) {
                return String.format("%.2fM", paramCount / 1_000_000.0);
            } else if (paramCount >= 1_000) {
                return String.format("%.2fK", paramCount / 1_000.0);
            }
            return String.valueOf(paramCount);
        }

        public String getFormattedFileSize() {
            if (fileSize >= 1_000_000_000) {
                return String.format("%.2fGB", fileSize / 1_000_000_000.0);
            } else if (fileSize >= 1_000_000) {
                return String.format("%.2fMB", fileSize / 1_000_000.0);
            } else if (fileSize >= 1_000) {
                return String.format("%.2fKB", fileSize / 1_000.0);
            }
            return fileSize + "B";
        }
    }

    /**
     * Layer information.
     */
    public static class LayerInfo {
        private final String name;
        private final String type;
        private final long paramCount;
        private final String outputShape;
        private final Map<String, Object> attrs;

        public LayerInfo(String name, String type, long paramCount, String outputShape, Map<String, Object> attrs) {
            this.name = name;
            this.type = type;
            this.paramCount = paramCount;
            this.outputShape = outputShape;
            this.attrs = attrs;
        }

        public String getName() { return name; }
        public String getType() { return type; }
        public long getParamCount() { return paramCount; }
        public String getOutputShape() { return outputShape; }
        public Map<String, Object> getAttrs() { return attrs; }
    }

    /**
     * Get metadata for a model file.
     */
    public static ModelMeta getModelMeta(String path) throws IOException {
        return getModelMeta(new File(path));
    }

    public static ModelMeta getModelMeta(File file) throws IOException {
        if (!file.exists()) {
            throw new IOException("File not found: " + file);
        }

        // Use ModelInfoParser for comprehensive parsing
        ModelInfoParser.ModelInfo info = ModelInfoParser.parse(file);

        // Convert to ModelMeta format
        String name = info.getName();
        ModelFormat format = convertFormat(info.getFormat());
        long fileSize = info.getFileSize();
        long paramCount = info.getTotalParams();

        List<LayerInfo> layers = new ArrayList<>();
        for (ModelInfoParser.LayerInfo li : info.getLayers()) {
            layers.add(new LayerInfo(
                li.getName(),
                li.getType(),
                li.getParamCount(),
                li.getOutputShape(),
                li.getAttrs()
            ));
        }

        Map<String, Object> extra = new LinkedHashMap<>();
        extra.putAll(info.getMetadata());
        extra.put("parseErrors", info.getErrors());
        extra.put("hasErrors", info.hasErrors());

        return new ModelMeta(name, format, paramCount, layers.size(), layers, extra, fileSize, info.getSourcePath());
    }

    private static ModelFormat convertFormat(ModelInfoParser.ModelFormat format) {
        if (format == null) return ModelFormat.UNKNOWN;
        switch (format) {
            case SAFETENSORS: return ModelFormat.SAFETENSORS;
            case PYTHON_PKL:
            case PYTHON_PTH: return ModelFormat.PYTHON_PT;
            case JAVACPP_PT: return ModelFormat.JAVACPP_PT;
            case AOTI_PACKAGE: return ModelFormat.JAVACPP_AOTI;
            case ONNX: return ModelFormat.ONNX;
            case STRUCTURE_JSON: return ModelFormat.STRUCTURE_JSON;
            case TORCHSCRIPT: return ModelFormat.PYTHON_TORCHSCRIPT;
            case PT2_FORMAT: return ModelFormat.PYTHON_PT2;
            case HUGGINGFACE_BIN: return ModelFormat.PYTHON_PT;
            default: return ModelFormat.UNKNOWN;
        }
    }

    // =========================================================================
    // Tracing with Format Support
    // =========================================================================

    /**
     * Trace a model file and render to HTML with metadata.
     */
    public static TraceGraph trace(String path) throws IOException {
        return trace(path, null);
    }

    public static TraceGraph trace(String path, Object inputs) throws IOException {
        return trace(new File(path), inputs);
    }

    public static TraceGraph trace(File file) throws IOException {
        return trace(file, null);
    }

    public static TraceGraph trace(File file, Object inputs) throws IOException {
        return trace(file, inputs, VistaOptions.defaults());
    }

    public static TraceGraph trace(File file, Object inputs, VistaOptions options) throws IOException {
        // First, get metadata
        ModelMeta meta = null;
        try {
            meta = getModelMeta(file);
        } catch (Exception e) {
            System.err.println("[vista] Warning: Could not extract metadata: " + e.getMessage());
        }

        // Use existing Vista for the actual tracing
        TraceGraph graph = Vista.traceFile(file, inputs, options);

        // Attach metadata to graph if available
        if (meta != null) {
            attachMetaToGraph(graph, meta);
        }

        return graph;
    }

    /**
     * Trace with metadata panel in HTML output.
     */
    public static TraceGraph traceWithMeta(String path, Object inputs) throws IOException {
        VistaOptions opts = VistaOptions.builder()
                .showMetadata(true)
                .build();
        return trace(new File(path), inputs, opts);
    }

    public static TraceGraph traceWithMeta(String path) throws IOException {
        return traceWithMeta(path, null);
    }

    private static void attachMetaToGraph(TraceGraph graph, ModelMeta meta) {
        // Add model metadata to nodeMeta
        Map<String, Object> modelMeta = new LinkedHashMap<>();
        modelMeta.put("kind", "model");
        modelMeta.put("name", meta.getName());
        modelMeta.put("format", meta.getFormat().displayName);
        modelMeta.put("formatRaw", meta.getFormat().name());
        modelMeta.put("param_count", meta.getParamCount());
        modelMeta.put("param_count_formatted", meta.getFormattedParamCount());
        modelMeta.put("layer_count", meta.getLayerCount());
        modelMeta.put("file_size", meta.getFileSize());
        modelMeta.put("file_size_formatted", meta.getFormattedFileSize());
        modelMeta.put("source", meta.getSourcePath());

        // Add extra metadata from parser
        Map<String, Object> extra = meta.getExtra();
        if (extra != null) {
            // Add useful extra info
            if (extra.containsKey("totalBytes")) {
                modelMeta.put("total_bytes", extra.get("totalBytes"));
                long bytes = ((Number) extra.get("totalBytes")).longValue();
                if (bytes >= 1_000_000_000) {
                    modelMeta.put("total_bytes_formatted", String.format("%.2fGB", bytes / 1_000_000_000.0));
                } else if (bytes >= 1_000_000) {
                    modelMeta.put("total_bytes_formatted", String.format("%.2fMB", bytes / 1_000_000.0));
                } else if (bytes >= 1_000) {
                    modelMeta.put("total_bytes_formatted", String.format("%.2fKB", bytes / 1_000.0));
                }
            }
            if (extra.containsKey("producerName")) {
                modelMeta.put("producer_name", extra.get("producerName"));
            }
            if (extra.containsKey("graphName")) {
                modelMeta.put("graph_name", extra.get("graphName"));
            }
            if (extra.containsKey("version")) {
                modelMeta.put("version", extra.get("version"));
            }
            if (extra.containsKey("irVersion")) {
                modelMeta.put("ir_version", extra.get("irVersion"));
            }
            if (extra.containsKey("inputs")) {
                modelMeta.put("inputs", extra.get("inputs"));
            }
            if (extra.containsKey("outputs")) {
                modelMeta.put("outputs", extra.get("outputs"));
            }
            if (extra.containsKey("moduleTypes")) {
                modelMeta.put("module_types", extra.get("moduleTypes"));
            }
            if (extra.containsKey("structureVersion")) {
                modelMeta.put("structure_version", extra.get("structureVersion"));
            }
            if (extra.containsKey("hasStructure")) {
                modelMeta.put("has_structure", extra.get("hasStructure"));
            }
        }

        // Add layers info
        List<Map<String, Object>> layerList = new ArrayList<>();
        for (LayerInfo layer : meta.getLayers()) {
            Map<String, Object> layerMap = new LinkedHashMap<>();
            layerMap.put("name", layer.getName());
            layerMap.put("type", layer.getType());
            layerMap.put("params", layer.getParamCount());
            layerMap.put("shape", layer.getOutputShape());
            if (layer.getAttrs() != null) {
                layerMap.put("attrs", layer.getAttrs());
            }
            layerList.add(layerMap);
        }
        modelMeta.put("layers", layerList);

        // Add parse errors if any
        if (extra != null && extra.containsKey("hasErrors") && Boolean.TRUE.equals(extra.get("hasErrors"))) {
            modelMeta.put("parse_errors", extra.get("parseErrors"));
        }

        graph.nodeMeta().put("_model_meta_", modelMeta);
    }

    // =========================================================================
    // Summary Generation
    // =========================================================================

    /**
     * Generate a text summary of the model.
     */
    public static String generateSummary(String path) throws IOException {
        ModelMeta meta = getModelMeta(path);
        return generateSummary(meta);
    }

    public static String generateSummary(ModelMeta meta) {
        StringBuilder sb = new StringBuilder();
        sb.append("═".repeat(60)).append("\n");
        sb.append(" Model Summary\n");
        sb.append("═".repeat(60)).append("\n\n");

        sb.append("  Name:       ").append(meta.getName()).append("\n");
        sb.append("  Format:     ").append(meta.getFormat().displayName).append("\n");
        sb.append("  Size:       ").append(meta.getFormattedFileSize()).append("\n");
        sb.append("  Parameters: ").append(meta.getFormattedParamCount()).append("\n");
        sb.append("  Layers:     ").append(meta.getLayerCount()).append("\n");
        sb.append("  Source:     ").append(meta.getSourcePath()).append("\n");

        sb.append("\n  ").append("─".repeat(40)).append("\n");
        sb.append("  Layer Statistics\n");
        sb.append("  ").append("─".repeat(40)).append("\n");

        long totalParams = meta.getParamCount();
        if (!meta.getLayers().isEmpty()) {
            sb.append("\n  Top 10 layers by parameters:\n");
            meta.getLayers().stream()
                    .sorted((a, b) -> Long.compare(b.getParamCount(), a.getParamCount()))
                    .limit(10)
                    .forEach(layer -> {
                        double pct = totalParams > 0 ? layer.getParamCount() * 100.0 / totalParams : 0;
                        sb.append(String.format("    %-40s %10d (%.1f%%)\n",
                                truncate(layer.getName(), 40), layer.getParamCount(), pct));
                    });
        }

        sb.append("\n  ").append("─".repeat(40)).append("\n");
        sb.append("  All Layers\n");
        sb.append("  ").append("─".repeat(40)).append("\n");

        for (LayerInfo layer : meta.getLayers()) {
            sb.append(String.format("    %-40s %s %s\n",
                    truncate(layer.getName(), 40),
                    layer.getType(),
                    layer.getOutputShape()));
        }

        sb.append("\n").append("═".repeat(60)).append("\n");
        return sb.toString();
    }

    private static String truncate(String s, int maxLen) {
        if (s == null) return "";
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen - 3) + "...";
    }

    // =========================================================================
    // Format Support Info
    // =========================================================================

    /**
     * Get all supported formats.
     */
    public static String getSupportedFormatsHelp() {
        StringBuilder sb = new StringBuilder();
        sb.append("Supported Model Formats:\n\n");

        sb.append("Python PyTorch:\n");
        sb.append("  .pt, .pth    - Standard PyTorch checkpoint\n");
        sb.append("  .pt2         - PyTorch 2.0+ exported model\n");
        sb.append("  .pte         - PyTorch Model Editor format\n");
        sb.append("  TorchScript   - JIT compiled models\n\n");

        sb.append("JavaCPP/LibTorch:\n");
        sb.append("  .pt, .pth    - Native LibTorch archive\n");
        sb.append("  .pt2         - AOT Inductor compiled model\n");
        sb.append("  .aoti        - AOT Inductor package directory\n\n");

        sb.append("Other:\n");
        sb.append("  .safetensors - HuggingFace safe tensors\n");
        sb.append("  .structure.json - Structure specification\n\n");

        return sb.toString();
    }
}
