package org.bytedeco.pytorch.plot.vista;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.serialize.*;
import org.bytedeco.pytorch.inductor.AOTIModelPackageLoader;
import org.bytedeco.pytorch.Module;

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

        String name = file.getName().toLowerCase(Locale.ROOT);

        // Extension-based detection
        if (name.endsWith(".safetensors")) {
            return ModelFormat.SAFETENSORS;
        }
        if (name.endsWith(".structure.json") || (name.endsWith(".json") && name.contains("structure"))) {
            return ModelFormat.STRUCTURE_JSON;
        }
        if (name.endsWith(".aoti")) {
            return ModelFormat.JAVACPP_AOTI;
        }

        // ONNX detection
        if (name.endsWith(".onnx")) {
            return ModelFormat.ONNX;
        }

        // PT2 detection
        if (name.endsWith(".pt2")) {
            // Try to detect if it's Python or JavaCPP
            if (isPythonPklZip(file)) {
                return ModelFormat.PYTHON_PT2;
            }
            return ModelFormat.JAVACPP_PT2;
        }

        // PTE detection
        if (name.endsWith(".pte")) {
            return ModelFormat.PYTHON_PTE;
        }

        // .pt/.pth detection
        if (name.endsWith(".pt") || name.endsWith(".pth")) {
            // Check content
            if (isTorchScript(file)) {
                return ModelFormat.PYTHON_TORCHSCRIPT;
            }
            if (isPythonPklZip(file)) {
                return ModelFormat.PYTHON_PT;
            }
            return ModelFormat.JAVACPP_PT;
        }

        // Directory detection
        if (file.isDirectory()) {
            if (looksLikeAotiPackage(file.toPath())) {
                return ModelFormat.JAVACPP_AOTI;
            }
        }

        return ModelFormat.UNKNOWN;
    }

    private static boolean isPythonPklZip(File file) throws IOException {
        try (var fis = Files.newInputStream(file.toPath())) {
            byte[] header = new byte[8];
            int read = fis.read(header);
            if (read < 2) return false;

            // ZIP magic: PK
            if (header[0] == 0x50 && header[1] == 0x4B) {
                return true;
            }

            // Pickle magic: 0x80-0x8F
            if ((header[0] & 0xFF) >= 0x80 && (header[0] & 0xFF) <= 0x8F) {
                return true;
            }
        }
        return false;
    }

    private static boolean isTorchScript(File file) throws IOException {
        try (var fis = Files.newInputStream(file.toPath())) {
            byte[] header = new byte[12];
            int read = fis.read(header);
            if (read < 8) return false;

            // TorchScript ZIP has 'scripts' entry
            if (header[0] == 0x50 && header[1] == 0x4B) { // PK
                byte[] scripts = "scripts".getBytes();
                byte[] allBytes = Files.readAllBytes(file.toPath());
                for (int i = 0; i < allBytes.length - 7; i++) {
                    if (allBytes[i] == scripts[0]) {
                        boolean match = true;
                        for (int j = 1; j < scripts.length && i + j < allBytes.length; j++) {
                            if (allBytes[i + j] != scripts[j]) {
                                match = false;
                                break;
                            }
                        }
                        if (match) return true;
                    }
                }
            }
        }
        return false;
    }

    private static boolean looksLikeAotiPackage(Path dir) {
        try {
            if (!Files.isDirectory(dir)) return false;
            return Files.exists(dir.resolve("data.pt"))
                    || Files.exists(dir.resolve("data"))
                    || Files.list(dir).anyMatch(p -> {
                        String n = p.getFileName().toString().toLowerCase();
                        return n.endsWith(".so") || n.contains("model") || n.equals("metadata.json");
                    });
        } catch (Exception e) {
            return false;
        }
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

        ModelFormat format = detectFormat(file);
        String name = extractModelName(file);
        long fileSize = file.length();

        Map<String, Tensor> weights = new LinkedHashMap<>();
        StructureSpec spec = null;
        Module module = null;
        List<LayerInfo> layers = new ArrayList<>();
        Map<String, Object> extra = new LinkedHashMap<>();
        long paramCount = 0;

        // Load based on format
        switch (format) {
            case SAFETENSORS:
                weights = SafeTensors.loadAsTensors(file, false);
                break;
            case PYTHON_PT:
            case JAVACPP_PT:
                try {
                    WeightBagModule bag = WeightBagModule.fromPythonPth(file, false);
                    module = bag;
                    for (String key : bag.stateDict().keySet()) {
                        weights.put(key, bag.stateDict().get(key));
                    }
                    try {
                        spec = StructureSpec.fromModule(bag);
                    } catch (Exception ignored) {}
                } catch (Exception e) {
                    // Try native load
                    weights = ModelWeights.load(file, false);
                }
                break;
            case JAVACPP_AOTI:
                // AOTI - load metadata only
                try {
                    AOTIModelPackageLoader loader = new AOTIModelPackageLoader(file.getAbsolutePath());
                    // Extract metadata
                } catch (Exception ignored) {}
                break;
            case ONNX:
                // ONNX - load model info
                try {
                    var onnxSession = org.bytedeco.pytorch.serving.onnxruntime.ONNXSession.load(file.getAbsolutePath());
                    var onnxInfo = onnxSession.getModelInfo();
                    // Build layers from input/output tensors
                    for (var input : onnxInfo.getInputs()) {
                        long size = input.getNumElements();
                        layers.add(new LayerInfo(
                            "input_" + input.getName(),
                            "Input(" + input.getElementTypeString() + ")",
                            size,
                            input.getShapeString(),
                            Map.of("type", input.getElementTypeString())
                        ));
                        paramCount += size;
                    }
                    for (var output : onnxInfo.getOutputs()) {
                        long size = output.getNumElements();
                        layers.add(new LayerInfo(
                            "output_" + output.getName(),
                            "Output(" + output.getElementTypeString() + ")",
                            size,
                            output.getShapeString(),
                            Map.of("type", output.getElementTypeString())
                        ));
                    }
                    extra.put("producerName", onnxInfo.getProducerName());
                    extra.put("graphName", onnxInfo.getGraphName());
                    extra.put("irVersion", onnxInfo.getIrVersion());
                    onnxSession.close();
                } catch (Exception ignored) {}
                break;
            case STRUCTURE_JSON:
                spec = StructureSpec.load(file);
                break;
            default:
                // Try generic load
                try {
                    weights = ModelWeights.load(file, false);
                } catch (Exception ignored) {}
        }

        // Calculate stats (paramCount, layers, extra already initialized above for ONNX branch)
        for (Tensor t : weights.values()) {
            paramCount += t.numel();
        }

        // Build layer info
        for (Map.Entry<String, Tensor> entry : weights.entrySet()) {
            String layerName = entry.getKey();
            Tensor tensor = entry.getValue();
            String layerType = inferLayerType(layerName);
            long layerParams = tensor.numel();
            String shape = formatShape(tensor);

            Map<String, Object> attrs = new LinkedHashMap<>();
            attrs.put("dtype", tensor.dtype().toString());
            attrs.put("shape", shape);

            layers.add(new LayerInfo(layerName, layerType, layerParams, shape, attrs));
        }

        // Extra metadata
        extra.put("detectedFormat", format.displayName);
        extra.put("isZip", isPythonPklZip(file));
        extra.put("hasStructure", spec != null);
        extra.put("weightCount", weights.size());

        return new ModelMeta(name, format, paramCount, layers.size(), layers, extra, fileSize, file.getAbsolutePath());
    }

    private static String extractModelName(File file) {
        String name = file.getName();
        // Remove extension
        int dotIdx = name.lastIndexOf('.');
        if (dotIdx > 0) {
            name = name.substring(0, dotIdx);
        }
        // Clean up
        name = name.replaceAll("[._-]*(state_dict|weights|model|pytorch|torch)", "");
        if (name.isEmpty()) {
            name = file.getName();
        }
        return name;
    }

    private static String inferLayerType(String name) {
        name = name.toLowerCase();
        if (name.contains("weight")) {
            if (name.contains("embed")) return "Embedding";
            if (name.contains("bias")) return "Bias";
            return "Linear/Weight";
        }
        if (name.contains("bias")) return "Bias";
        if (name.contains("running_mean") || name.contains("running_var")) return "BatchNorm/Running";
        if (name.contains("num_batches_tracked")) return "BatchNorm/Tracker";
        return "Parameter";
    }

    private static String formatShape(Tensor t) {
        long[] sizes = new long[(int) t.dim()];
        for (int i = 0; i < sizes.length; i++) {
            sizes[i] = t.size(i);
        }
        return Arrays.toString(sizes);
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
        modelMeta.put("param_count", meta.getParamCount());
        modelMeta.put("param_count_formatted", meta.getFormattedParamCount());
        modelMeta.put("layer_count", meta.getLayerCount());
        modelMeta.put("file_size", meta.getFileSize());
        modelMeta.put("file_size_formatted", meta.getFormattedFileSize());
        modelMeta.put("source", meta.getSourcePath());

        // Add layers info
        List<Map<String, Object>> layerList = new ArrayList<>();
        for (LayerInfo layer : meta.getLayers()) {
            Map<String, Object> layerMap = new LinkedHashMap<>();
            layerMap.put("name", layer.getName());
            layerMap.put("type", layer.getType());
            layerMap.put("params", layer.getParamCount());
            layerMap.put("shape", layer.getOutputShape());
            layerList.add(layerMap);
        }
        modelMeta.put("layers", layerList);

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
