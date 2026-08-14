package org.bytedeco.pytorch.plot.vista;

import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.inductor.*;
import org.bytedeco.pytorch.inductor.AOTIModelPackageLoader;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.data.safetensors.*;
import org.bytedeco.pytorch.data.serialize.*;
import org.bytedeco.pytorch.serving.onnxruntime.*;

/**
 * Unified model information parser supporting multiple formats.
 * Extracts comprehensive metadata for visualization in Vista frontends.
 * 
 * <h2>Supported Formats</h2>
 * <ul>
 *   <li>.pkl/.pickle - Python pickle files</li>
 *   <li>.pth/.pt - PyTorch checkpoint files</li>
 *   <li>.bin - Binary model files (HuggingFace bin format)</li>
 *   <li>.aot/.aoti - AOT Inductor compiled models</li>
 *   <li>.onnx - ONNX Runtime models</li>
 *   <li>.safetensors - HuggingFace SafeTensors</li>
 *   <li>.pt2/.pte - PyTorch 2.0+ formats</li>
 *   <li>.ot - TorchScript serialized models</li>
 *   <li>Structure JSON files</li>
 * </ul>
 */
public final class ModelInfoParser {

    private ModelInfoParser() {}

    // ========================================================================
    // Main Entry Points
    // ========================================================================

    /**
     * Parse model file and extract comprehensive information.
     */
    public static ModelInfo parse(String path) throws IOException {
        return parse(new File(path));
    }

    public static ModelInfo parse(File file) throws IOException {
        if (file == null || !file.exists()) {
            throw new IOException("File not found: " + file);
        }

        ModelFormat format = detectFormat(file);
        String name = extractModelName(file);
        long fileSize = file.length();

        ModelInfo info = new ModelInfo(name, format, fileSize, file.getAbsolutePath());
        
        // Parse based on format
        try {
            switch (format) {
                case SAFETENSORS:
                    parseSafeTensors(file, info);
                    break;
                case PYTHON_PKL:
                case PYTHON_PTH:
                    parsePythonPth(file, info);
                    break;
                case JAVACPP_PT:
                    parseJavacppPt(file, info);
                    break;
                case AOTI_PACKAGE:
                    parseAotiPackage(file, info);
                    break;
                case ONNX:
                    parseOnnx(file, info);
                    break;
                case HUGGINGFACE_BIN:
                    parseHuggingFaceBin(file, info);
                    break;
                case STRUCTURE_JSON:
                    parseStructureJson(file, info);
                    break;
                case TORCHSCRIPT:
                    parseTorchScript(file, info);
                    break;
                case PT2_FORMAT:
                    parsePt2(file, info);
                    break;
                case UNKNOWN:
                    parseUnknown(file, info);
                    break;
            }
        } catch (Exception e) {
            info.addError("Parse error: " + e.getMessage());
            // Try fallback parsing
            parseFallback(file, info);
        }

        // Calculate statistics
        info.calculateStats();

        return info;
    }

    // ========================================================================
    // Format Detection
    // ========================================================================

    public enum ModelFormat {
        SAFETENSORS("SafeTensors (.safetensors)"),
        PYTHON_PKL("Python Pickle (.pkl/.pickle)"),
        PYTHON_PTH("Python PyTorch (.pth/.pt)"),
        JAVACPP_PT("JavaCPP/LibTorch (.pt)"),
        AOTI_PACKAGE("AOT Inductor Package (.aoti)"),
        ONNX("ONNX Runtime (.onnx)"),
        HUGGINGFACE_BIN("HuggingFace Bin (.bin)"),
        STRUCTURE_JSON("Structure JSON (.structure.json)"),
        TORCHSCRIPT("TorchScript (.pt/.ot)"),
        PT2_FORMAT("PyTorch 2.0 Exported (.pt2/.pte)"),
        UNKNOWN("Unknown Format");

        public final String displayName;
        ModelFormat(String displayName) { this.displayName = displayName; }
    }

    public static ModelFormat detectFormat(File file) throws IOException {
        if (file == null || !file.exists()) return ModelFormat.UNKNOWN;

        String name = file.getName().toLowerCase(Locale.ROOT);

        // Extension-based detection
        if (name.endsWith(".safetensors")) {
            return ModelFormat.SAFETENSORS;
        }
        if (name.endsWith(".structure.json") || (name.endsWith(".json") && name.contains("structure"))) {
            return ModelFormat.STRUCTURE_JSON;
        }
        if (name.endsWith(".aoti")) {
            return ModelFormat.AOTI_PACKAGE;
        }
        if (name.endsWith(".onnx")) {
            return ModelFormat.ONNX;
        }
        if (name.endsWith(".pt2") || name.endsWith(".pte")) {
            return ModelFormat.PT2_FORMAT;
        }
        if (name.endsWith(".ot")) {
            return ModelFormat.TORCHSCRIPT;
        }
        if (name.endsWith(".pkl") || name.endsWith(".pickle")) {
            return ModelFormat.PYTHON_PKL;
        }
        if (name.endsWith(".pth") || name.endsWith(".pt")) {
            return detectPthVariant(file);
        }
        if (name.endsWith(".bin")) {
            return ModelFormat.HUGGINGFACE_BIN;
        }

        // Directory detection
        if (file.isDirectory()) {
            if (looksLikeAotiPackage(file.toPath())) {
                return ModelFormat.AOTI_PACKAGE;
            }
        }

        // Content-based detection
        try {
            byte[] header = readHeader(file, 16);
            if (header == null || header.length < 2) return ModelFormat.UNKNOWN;

            // ZIP magic: PK
            if (header[0] == 0x50 && header[1] == 0x4B) {
                // Check if it's a TorchScript ZIP (contains 'scripts' entry)
                if (hasTorchScriptMarker(file)) {
                    return ModelFormat.TORCHSCRIPT;
                }
                // Check if it's Python pickle ZIP
                if (hasPickleContent(file)) {
                    return ModelFormat.PYTHON_PTH;
                }
                return ModelFormat.PYTHON_PKL;
            }

            // Pickle magic: 0x80-0x8F
            if ((header[0] & 0xFF) >= 0x80 && (header[0] & 0xFF) <= 0x8F) {
                return ModelFormat.PYTHON_PKL;
            }

            // ONNX magic
            if (hasOnnxMagic(header)) {
                return ModelFormat.ONNX;
            }
        } catch (Exception ignored) {}

        return ModelFormat.UNKNOWN;
    }

    private static ModelFormat detectPthVariant(File file) throws IOException {
        if (isTorchScript(file)) return ModelFormat.TORCHSCRIPT;
        if (isPythonPklZip(file)) return ModelFormat.PYTHON_PTH;
        return ModelFormat.JAVACPP_PT;
    }

    private static boolean isPythonPklZip(File file) throws IOException {
        byte[] header = readHeader(file, 8);
        if (header == null || header.length < 2) return false;
        // ZIP magic
        if (header[0] == 0x50 && header[1] == 0x4B) return true;
        // Pickle magic
        return (header[0] & 0xFF) >= 0x80 && (header[0] & 0xFF) <= 0x8F;
    }

    private static boolean isTorchScript(File file) throws IOException {
        byte[] header = readHeader(file, 12);
        if (header == null || header.length < 8) return false;
        if (header[0] == 0x50 && header[1] == 0x4B) { // ZIP
            byte[] allBytes = Files.readAllBytes(file.toPath());
            byte[] scripts = "scripts".getBytes();
            return containsPattern(allBytes, scripts);
        }
        return false;
    }

    private static boolean hasTorchScriptMarker(File file) throws IOException {
        try {
            byte[] data = Files.readAllBytes(file.toPath());
            return containsPattern(data, "scripts".getBytes());
        } catch (Exception e) {
            return false;
        }
    }

    private static boolean hasPickleContent(File file) throws IOException {
        try {
            byte[] data = Files.readAllBytes(file.toPath());
            // Look for PyTorch model keys
            return containsPattern(data, "state_dict".getBytes())
                || containsPattern(data, "module".getBytes())
                || containsPattern(data, "_metadata".getBytes());
        } catch (Exception e) {
            return false;
        }
    }

    private static boolean hasOnnxMagic(byte[] header) {
        if (header.length < 8) return false;
        // ONNX files start with 'ONNX' or protobuf header
        return (header[0] == 'O' && header[1] == 'N' && header[2] == 'N' && header[3] == 'X')
            || (header[0] == 0x08); // protobuf field 1, wire_type 0
    }

    private static boolean looksLikeAotiPackage(Path dir) {
        try {
            if (!Files.isDirectory(dir)) return false;
            return Files.exists(dir.resolve("data.pt"))
                || Files.exists(dir.resolve("data"))
                || Files.exists(dir.resolve("metadata.json"))
                || Files.list(dir).anyMatch(p -> {
                    String n = p.getFileName().toString().toLowerCase();
                    return n.endsWith(".so") || n.contains("model");
                });
        } catch (Exception e) {
            return false;
        }
    }

    // ========================================================================
    // Format-Specific Parsers
    // ========================================================================

    private static void parseSafeTensors(File file, ModelInfo info) throws IOException {
        try {
            Map<String, Tensor> tensors = SafeTensors.loadAsTensors(file, false);
            info.tensors.putAll(tensors);
            info.setFormat(ModelFormat.SAFETENSORS);

            // Try to load metadata
            try {
                Map<String, String> metadata = SafeTensors.readMetadata(file);
                if (metadata != null) {
                    info.metadata.putAll(metadata);
                }
            } catch (Exception ignored) {}

            // Add layer info from tensors
            for (Map.Entry<String, Tensor> e : tensors.entrySet()) {
                String layerName = e.getKey();
                Tensor t = e.getValue();
                LayerInfo layer = new LayerInfo(
                    layerName,
                    inferLayerType(layerName),
                    t.numel(),
                    formatShape(t),
                    inferLayerAttrs(layerName, t)
                );
                info.layers.add(layer);
                info.totalParams += t.numel();
            }

            // Look for sibling structure.json
            File structureFile = findSibling(file, ".structure.json");
            if (structureFile != null && structureFile.exists()) {
                try {
                    StructureSpec spec = StructureSpec.load(structureFile);
                    enrichFromStructure(info, spec);
                } catch (Exception ignored) {}
            }

        } catch (Exception e) {
            info.addError("SafeTensors parse failed: " + e.getMessage());
        }
    }

    private static void parsePythonPth(File file, ModelInfo info) throws IOException {
        try {
            WeightBagModule bag = WeightBagModule.fromPythonPth(file, false);
            info.module = bag;

            Map<String, Tensor> stateDict = bag.stateDict();
            if (stateDict != null) {
                info.tensors.putAll(stateDict);
                for (Map.Entry<String, Tensor> e : stateDict.entrySet()) {
                    String layerName = e.getKey();
                    Tensor t = e.getValue();
                    LayerInfo layer = new LayerInfo(
                        layerName,
                        inferLayerType(layerName),
                        t.numel(),
                        formatShape(t),
                        inferLayerAttrs(layerName, t)
                    );
                    info.layers.add(layer);
                    info.totalParams += t.numel();
                }
            }

            // Try structure extraction
            try {
                StructureSpec spec = StructureSpec.fromModule(bag);
                enrichFromStructure(info, spec);
            } catch (Exception ignored) {}

            info.setFormat(ModelFormat.PYTHON_PTH);

        } catch (Exception e) {
            info.addError("Python PTH parse failed: " + e.getMessage());
            // Fallback: try as pickle
            parsePickleContent(file, info);
        }
    }

    private static void parseJavacppPt(File file, ModelInfo info) throws IOException {
        try {
            // Try native load first
            Map<String, Tensor> weights = ModelWeights.load(file, false);
            info.tensors.putAll(weights);

            for (Map.Entry<String, Tensor> e : weights.entrySet()) {
                Tensor t = e.getValue();
                LayerInfo layer = new LayerInfo(
                    e.getKey(),
                    inferLayerType(e.getKey()),
                    t.numel(),
                    formatShape(t),
                    inferLayerAttrs(e.getKey(), t)
                );
                info.layers.add(layer);
                info.totalParams += t.numel();
            }

            info.setFormat(ModelFormat.JAVACPP_PT);

        } catch (Exception e1) {
            // Fallback: try Python path
            try {
                parsePythonPth(file, info);
            } catch (Exception e2) {
                info.addError("JavaCPP PT parse failed: " + e1.getMessage());
            }
        }
    }

    private static void parseAotiPackage(File file, ModelInfo info) throws IOException {
        try {
            AOTIModelPackageLoader loader = new AOTIModelPackageLoader(file.getAbsolutePath());
            info.aotiLoader = loader;

            // Extract metadata using iterator
            try {
                ExtraFilesMap metadata = loader.get_metadata();
                if (metadata != null && !metadata.isNull()) {
                    ExtraFilesMap.Iterator iter = metadata.begin();
                    ExtraFilesMap.Iterator end = metadata.end();
                    while (!iter.equals(end)) {
                        try {
                            BytePointer keyPtr = iter.first();
                            BytePointer valPtr = iter.second();
                            if (keyPtr != null && valPtr != null) {
                                String key = keyPtr.getString();
                                String value = valPtr.getString();
                                if (key != null && value != null) {
                                    info.metadata.put(key, value);
                                }
                            }
                        } catch (Exception ignored) {}
                        iter.increment();
                    }
                }
            } catch (Exception ignored) {}

            // Get constants
            try {
                StringVector constFqns = loader.get_constant_fqns();
                if (constFqns != null && !constFqns.isNull()) {
                    long size = constFqns.size();
                    for (long i = 0; i < size; i++) {
                        try {
                            BytePointer bp = constFqns.get(i);
                            if (bp != null) {
                                String name = bp.getString();
                                if (name != null && !name.isEmpty()) {
                                    LayerInfo layer = new LayerInfo(
                                        name,
                                        "Constant",
                                        0,
                                        "[]",
                                        Map.<String, Object>of("category", "constant")
                                    );
                                    info.layers.add(layer);
                                }
                            }
                        } catch (Exception ignored) {}
                    }
                }
            } catch (Exception ignored) {}

            // Get call spec
            try {
                StringVector callSpec = loader.get_call_spec();
                if (callSpec != null && !callSpec.isNull()) {
                    List<String> inputs = new ArrayList<>();
                    List<String> outputs = new ArrayList<>();
                    boolean readingInput = true;
                    long specSize = callSpec.size();
                    for (long i = 0; i < specSize; i++) {
                        try {
                            BytePointer bp = callSpec.get(i);
                            if (bp != null) {
                                String s = bp.getString();
                                if (s != null) {
                                    if (s.startsWith("output:")) readingInput = false;
                                    else if (!s.startsWith("input:") && !s.startsWith("output:")) {
                                        if (readingInput) inputs.add(s);
                                        else outputs.add(s);
                                    }
                                }
                            }
                        } catch (Exception ignored) {}
                    }
                    if (!inputs.isEmpty()) {
                        info.metadata.put("inputs", String.join(", ", inputs));
                    }
                    if (!outputs.isEmpty()) {
                        info.metadata.put("outputs", String.join(", ", outputs));
                    }
                }
            } catch (Exception ignored) {}

            info.setFormat(ModelFormat.AOTI_PACKAGE);

        } catch (Exception e) {
            info.addError("AOTI parse failed: " + e.getMessage());
        }
    }

    private static void parseOnnx(File file, ModelInfo info) throws IOException {
        try {
            ONNXSession session = ONNXSession.load(file.getAbsolutePath());
            ONNXModelInfo modelInfo = session.getModelInfo();

            info.metadata.put("producerName", modelInfo.getProducerName());
            info.metadata.put("graphName", modelInfo.getGraphName());
            info.metadata.put("version", modelInfo.getVersion());
            info.metadata.put("irVersion", modelInfo.getIrVersion());

            // Input layers
            List<ONNXTensorInfo> inputs = modelInfo.getInputs();
            if (inputs != null) {
                for (ONNXTensorInfo input : inputs) {
                    long size = input.getNumElements();
                    LayerInfo layer = new LayerInfo(
                        "input_" + input.getName(),
                        "Input(" + input.getElementTypeString() + ")",
                        size,
                        input.getShapeString(),
                        Map.<String, Object>of(
                            "type", input.getElementTypeString(),
                            "category", "input"
                        )
                    );
                    info.layers.add(layer);
                    info.totalParams += size;
                }
            }

            // Output layers
            List<ONNXTensorInfo> outputs = modelInfo.getOutputs();
            if (outputs != null) {
                for (ONNXTensorInfo output : outputs) {
                    LayerInfo layer = new LayerInfo(
                        "output_" + output.getName(),
                        "Output(" + output.getElementTypeString() + ")",
                        output.getNumElements(),
                        output.getShapeString(),
                        Map.<String, Object>of(
                            "type", output.getElementTypeString(),
                            "category", "output"
                        )
                    );
                    info.layers.add(layer);
                }
            }

            info.setFormat(ModelFormat.ONNX);
            session.close();

        } catch (Exception e) {
            info.addError("ONNX parse failed: " + e.getMessage());
        }
    }

    private static void parseHuggingFaceBin(File file, ModelInfo info) throws IOException {
        // HuggingFace .bin files are typically PyTorch tensors
        try {
            parsePythonPth(file, info);
            info.setFormat(ModelFormat.HUGGINGFACE_BIN);
        } catch (Exception e) {
            info.addError("HuggingFace bin parse failed: " + e.getMessage());
        }
    }

    private static void parseStructureJson(File file, ModelInfo info) throws IOException {
        try {
            StructureSpec spec = StructureSpec.load(file);
            enrichFromStructure(info, spec);
            info.setFormat(ModelFormat.STRUCTURE_JSON);
        } catch (Exception e) {
            info.addError("Structure JSON parse failed: " + e.getMessage());
        }
    }

    private static void parseTorchScript(File file, ModelInfo info) throws IOException {
        // TorchScript - try to load as module
        try {
            parsePythonPth(file, info);
            info.setFormat(ModelFormat.TORCHSCRIPT);
        } catch (Exception e) {
            info.addError("TorchScript parse failed: " + e.getMessage());
            // Basic parsing from file
            parseBinaryMetadata(file, info);
        }
    }

    private static void parsePt2(File file, ModelInfo info) throws IOException {
        // PT2 format - can be Python or AOT
        if (isPythonPklZip(file)) {
            try {
                parsePythonPth(file, info);
                info.setFormat(ModelFormat.PT2_FORMAT);
                info.metadata.put("variant", "Python PT2");
            } catch (Exception e) {
                info.addError("PT2 parse failed: " + e.getMessage());
            }
        } else {
            try {
                parseAotiPackage(file, info);
                info.metadata.put("variant", "AOT Inductor PT2");
            } catch (Exception e) {
                info.addError("PT2 AOT parse failed: " + e.getMessage());
            }
        }
    }

    private static void parseUnknown(File file, ModelInfo info) throws IOException {
        // Try various parsers
        parseBinaryMetadata(file, info);

        // Try as safetensors
        try {
            Map<String, Tensor> tensors = SafeTensors.loadAsTensors(file, false);
            if (!tensors.isEmpty()) {
                info.tensors.putAll(tensors);
                for (Map.Entry<String, Tensor> e : tensors.entrySet()) {
                    Tensor t = e.getValue();
                    LayerInfo layer = new LayerInfo(e.getKey(), inferLayerType(e.getKey()),
                        t.numel(), formatShape(t), inferLayerAttrs(e.getKey(), t));
                    info.layers.add(layer);
                    info.totalParams += t.numel();
                }
                info.setFormat(ModelFormat.SAFETENSORS);
                return;
            }
        } catch (Exception ignored) {}

        // Try as pickle
        try {
            parsePickleContent(file, info);
            return;
        } catch (Exception ignored) {}

        info.addError("Could not determine model format");
    }

    private static void parseFallback(File file, ModelInfo info) throws IOException {
        // Last resort: binary metadata extraction
        parseBinaryMetadata(file, info);

        // Extract any readable strings that look like layer names
        try {
            byte[] data = Files.readAllBytes(file.toPath());
            extractReadableKeys(data, info);
        } catch (Exception ignored) {}
    }

    private static void parseBinaryMetadata(File file, ModelInfo info) throws IOException {
        try {
            long size = file.length();
            if (size > 0 && size < 10 * 1024 * 1024) { // Only for files < 10MB
                byte[] header = readHeader(file, Math.min((int)size, 4096));
                if (header != null) {
                    info.metadata.put("magic", String.format("0x%02X", header[0] & 0xFF));
                    info.metadata.put("headerPreview", new String(header, 0, Math.min(header.length, 64), StandardCharsets.UTF_8)
                        .replaceAll("[^\\x20-\\x7E\n]", "?"));
                }
            }
        } catch (Exception ignored) {}
    }

    private static void parsePickleContent(File file, ModelInfo info) throws IOException {
        try {
            WeightBagModule bag = WeightBagModule.fromPythonPth(file, false);
            Map<String, Tensor> stateDict = bag.stateDict();
            if (stateDict != null && !stateDict.isEmpty()) {
                info.tensors.putAll(stateDict);
                for (Map.Entry<String, Tensor> e : stateDict.entrySet()) {
                    Tensor t = e.getValue();
                    LayerInfo layer = new LayerInfo(e.getKey(), inferLayerType(e.getKey()),
                        t.numel(), formatShape(t), inferLayerAttrs(e.getKey(), t));
                    info.layers.add(layer);
                    info.totalParams += t.numel();
                }
            }
        } catch (Exception e) {
            info.addError("Pickle parse failed: " + e.getMessage());
        }
    }

    private static void extractReadableKeys(byte[] data, ModelInfo info) {
        String text = new String(data, StandardCharsets.ISO_8859_1);
        String[] lines = text.split("[\r\n]+");
        for (String line : lines) {
            // Look for patterns like "layer_name" or 'layer_name'
            java.util.regex.Pattern p = java.util.regex.Pattern.compile("[\"'](\\w+[\"']\\s*[=:])");
            java.util.regex.Matcher m = p.matcher(line);
            while (m.find()) {
                String key = m.group(1).replaceAll("[\"':=]", "").trim();
                if (key.length() > 3 && key.length() < 100 && !key.matches(".*[^a-zA-Z0-9_\\.].*")) {
                    info.metadata.put("foundKey_" + info.metadata.size(), key);
                }
            }
        }
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    private static byte[] readHeader(File file, int size) throws IOException {
        try (InputStream is = Files.newInputStream(file.toPath())) {
            byte[] header = new byte[size];
            int read = is.read(header);
            if (read < 0) return null;
            if (read < size) {
                return Arrays.copyOf(header, read);
            }
            return header;
        }
    }

    private static boolean containsPattern(byte[] data, byte[] pattern) {
        if (data == null || pattern == null || data.length < pattern.length) return false;
        outer:
        for (int i = 0; i <= data.length - pattern.length; i++) {
            for (int j = 0; j < pattern.length; j++) {
                if (data[i + j] != pattern[j]) continue outer;
            }
            return true;
        }
        return false;
    }

    private static String extractModelName(File file) {
        String name = file.getName();
        int dotIdx = name.lastIndexOf('.');
        if (dotIdx > 0) {
            name = name.substring(0, dotIdx);
        }
        // Clean up common suffixes
        name = name.replaceAll("[._-]*(state_dict|weights|model|pytorch|torch|safetensors|bin)", "");
        if (name.isEmpty()) {
            name = file.getName();
            dotIdx = name.lastIndexOf('.');
            if (dotIdx > 0) name = name.substring(0, dotIdx);
        }
        return name;
    }

    private static void enrichFromStructure(ModelInfo info, StructureSpec spec) {
        if (spec == null || spec.nodes == null) return;

        info.metadata.put("structureVersion", String.valueOf(spec.version));

        // Count module types
        Map<String, Integer> typeCounts = new LinkedHashMap<>();
        for (Map.Entry<String, StructureSpec.Node> e : spec.nodes.entrySet()) {
            StructureSpec.Node node = e.getValue();
            if (node != null && node.kind != null) {
                String kind = node.kind.toUpperCase();
                typeCounts.merge(kind, 1, Integer::sum);

                // Add as layer if not already present
                boolean found = info.layers.stream().anyMatch(l -> l.name.equals(e.getKey()));
                if (!found) {
                    LayerInfo layer = new LayerInfo(
                        e.getKey(),
                        kind,
                        node.ownParameters != null ? node.ownParameters.size() : 0,
                        "[]",
                        node.hyper != null ? new LinkedHashMap<>(node.hyper) : new LinkedHashMap<String, Object>()
                    );
                    info.layers.add(layer);
                }
            }
        }

        info.metadata.put("moduleTypes", typeCounts);
        info.metadata.put("hasStructure", true);
    }

    private static String inferLayerType(String name) {
        if (name == null) return "Parameter";
        name = name.toLowerCase();
        if (name.contains("weight")) {
            if (name.contains("embed")) return "Embedding";
            if (name.contains("layer_norm")) return "LayerNorm";
            if (name.contains("batch_norm")) return "BatchNorm";
            if (name.contains("conv")) return "Conv2d";
            if (name.contains("linear") || name.contains("fc")) return "Linear";
            return "Weight";
        }
        if (name.contains("bias")) return "Bias";
        if (name.contains("running_mean") || name.contains("running_var")) return "BatchNorm/Running";
        if (name.contains("num_batches_tracked")) return "BatchNorm/Tracker";
        if (name.contains("gru") || name.contains("lstm")) return "RNN";
        if (name.contains("attention") || name.contains("attn")) return "Attention";
        if (name.contains("embeddings")) return "Embedding";
        if (name.contains("pos_embed") || name.contains("cls_token")) return "PositionalEmbedding";
        return "Parameter";
    }

    private static Map<String, Object> inferLayerAttrs(String name, Tensor t) {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("dtype", t.dtype().toString());
        attrs.put("shape", formatShape(t));
        attrs.put("numBytes", t.nbytes());

        if (name != null) {
            String lower = name.toLowerCase();
            if (lower.contains("weight") && t.dim() >= 2) {
                attrs.put("category", "weight");
                long[] shape = t.shape();
                if (shape.length >= 2) {
                    attrs.put("outFeatures", shape[0]);
                    attrs.put("inFeatures", shape[1]);
                }
            } else if (lower.contains("bias")) {
                attrs.put("category", "bias");
            } else if (lower.contains("running")) {
                attrs.put("category", "running_stats");
            }
        }
        return attrs;
    }

    private static String formatShape(Tensor t) {
        if (t == null || t.isNull()) return "[]";
        long dim = t.dim();
        if (dim <= 0) return "[]";
        long[] sizes = new long[(int) dim];
        for (int i = 0; i < dim; i++) {
            sizes[i] = t.size(i);
        }
        return Arrays.toString(sizes);
    }

    private static File findSibling(File file, String suffix) {
        String base = file.getName();
        int dotIdx = base.lastIndexOf('.');
        if (dotIdx > 0) {
            base = base.substring(0, dotIdx);
        }
        File parent = file.getParentFile();
        if (parent != null) {
            File sibling = new File(parent, base + suffix);
            if (sibling.exists()) return sibling;
        }
        return null;
    }

    // ========================================================================
    // ModelInfo Data Class
    // ========================================================================

    public static class ModelInfo {
        private String name;
        private ModelFormat format;
        private long fileSize;
        private String sourcePath;
        private long totalParams;
        private long trainableParams;
        private long totalBytes;
        private List<LayerInfo> layers = new ArrayList<>();
        private Map<String, Object> metadata = new LinkedHashMap<>();
        private Map<String, Tensor> tensors = new LinkedHashMap<>();
        private List<String> errors = new ArrayList<>();
        private Module module;
        private AOTIModelPackageLoader aotiLoader;

        public ModelInfo(String name, ModelFormat format, long fileSize, String sourcePath) {
            this.name = name;
            this.format = format;
            this.fileSize = fileSize;
            this.sourcePath = sourcePath;
        }

        public void setFormat(ModelFormat format) { this.format = format; }

        public void addError(String error) { this.errors.add(error); }

        public void calculateStats() {
            totalBytes = 0;
            totalParams = 0;
            trainableParams = 0;

            for (Tensor t : tensors.values()) {
                if (t != null && !t.isNull()) {
                    totalParams += t.numel();
                    totalBytes += t.nbytes();
                }
            }

            for (LayerInfo layer : layers) {
                if (layer.paramCount > 0) {
                    totalParams += layer.paramCount;
                }
            }

            metadata.put("totalParams", totalParams);
            metadata.put("trainableParams", trainableParams);
            metadata.put("totalBytes", totalBytes);
            metadata.put("fileSize", fileSize);
            metadata.put("layerCount", layers.size());
            metadata.put("format", format.displayName);
        }

        // Getters
        public String getName() { return name; }
        public ModelFormat getFormat() { return format; }
        public long getFileSize() { return fileSize; }
        public String getSourcePath() { return sourcePath; }
        public long getTotalParams() { return totalParams; }
        public long getTrainableParams() { return trainableParams; }
        public long getTotalBytes() { return totalBytes; }
        public List<LayerInfo> getLayers() { return layers; }
        public Map<String, Object> getMetadata() { return metadata; }
        public Map<String, Tensor> getTensors() { return tensors; }
        public List<String> getErrors() { return errors; }
        public boolean hasErrors() { return !errors.isEmpty(); }

        public String getFormattedParamCount() {
            if (totalParams >= 1_000_000_000) {
                return String.format("%.2fB", totalParams / 1_000_000_000.0);
            } else if (totalParams >= 1_000_000) {
                return String.format("%.2fM", totalParams / 1_000_000.0);
            } else if (totalParams >= 1_000) {
                return String.format("%.2fK", totalParams / 1_000.0);
            }
            return String.valueOf(totalParams);
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

        /** Convert to JSON-safe map for frontend */
        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("name", name);
            map.put("format", format != null ? format.displayName : "Unknown");
            map.put("formatRaw", format != null ? format.name() : "UNKNOWN");
            map.put("fileSize", fileSize);
            map.put("fileSizeFormatted", getFormattedFileSize());
            map.put("sourcePath", sourcePath);
            map.put("totalParams", totalParams);
            map.put("trainableParams", trainableParams);
            map.put("totalBytes", totalBytes);
            map.put("paramCountFormatted", getFormattedParamCount());
            map.put("layerCount", layers.size());

            // Layers as list
            List<Map<String, Object>> layerList = new ArrayList<>();
            for (LayerInfo layer : layers) {
                layerList.add(layer.toMap());
            }
            map.put("layers", layerList);

            // Clean metadata for JSON
            Map<String, Object> cleanMeta = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : metadata.entrySet()) {
                Object v = e.getValue();
                if (v instanceof Map) {
                    cleanMeta.put(e.getKey(), v);
                } else if (v instanceof List) {
                    cleanMeta.put(e.getKey(), v);
                } else if (v instanceof Number || v instanceof String || v instanceof Boolean) {
                    cleanMeta.put(e.getKey(), v);
                } else if (v != null) {
                    cleanMeta.put(e.getKey(), v.toString());
                }
            }
            map.put("metadata", cleanMeta);

            // Errors
            map.put("hasErrors", hasErrors());
            if (hasErrors()) {
                map.put("errors", errors);
            }

            return map;
        }
    }

    // ========================================================================
    // LayerInfo Data Class
    // ========================================================================

    public static class LayerInfo {
        private String name;
        private String type;
        private long paramCount;
        private String outputShape;
        private Map<String, Object> attrs;

        public LayerInfo(String name, String type, long paramCount, String outputShape, Map<String, Object> attrs) {
            this.name = name;
            this.type = type;
            this.paramCount = paramCount;
            this.outputShape = outputShape;
            this.attrs = attrs != null ? attrs : new LinkedHashMap<>();
        }

        public String getName() { return name; }
        public String getType() { return type; }
        public long getParamCount() { return paramCount; }
        public String getOutputShape() { return outputShape; }
        public Map<String, Object> getAttrs() { return attrs; }

        public Map<String, Object> toMap() {
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("name", name);
            map.put("type", type);
            map.put("paramCount", paramCount);
            map.put("outputShape", outputShape);
            map.put("attrs", attrs);
            return map;
        }
    }
}
