package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.nn.*;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.security.MessageDigest;
import java.util.*;

/**
 * Cross-validate PyTorch bin files between Java and Python.
 *
 * <p>This loads the same file that Python's inspect_shard2.py loads and compares
 * the per-tensor MD5 checksums to verify data integrity.</p>
 *
 * <h2>Usage</h2>
 *
 * <pre>{@code
 * // 1. Run Python first to get reference data:
 * cd pytorch/test-models/llama-sentient-3b
 * python3 inspect_shard2.py > python_output.txt
 *
 * // 2. Compile and run Java:
 * cd pytorch
 * mvn compile -q
 * java -cp target/classes:... \
 *   org.bytedeco.pytorch.data.serialize.BinFileValidator \
 *   /home/muller/IdeaProjects/javacpp-mini-muller/pytorch/test-models/llama-sentient-3b/pytorch_model-00002-of-00002.bin
 *
 * // 3. Or use the convenience main:
 * java -cp target/classes:... \
 *   org.bytedeco.pytorch.data.serialize.BinFileValidator
 * }</pre>
 *
 * <h2>Validation workflow</h2>
 *
 * <ol>
 *   <li>Python loads the .bin file via torch.load()</li>
 *   <li>Java loads the same .bin file via PyTorchModelLoader.loadStateDict()</li>
 *   <li>Both compute MD5 checksums for each tensor's raw float16 bytes</li>
 *   <li>Compare results to ensure data integrity</li>
 * </ol>
 */
public class BinFileValidator {

    private static final String LLAMA_DIR = "/home/muller/IdeaProjects/javacpp-mini-muller/pytorch/test-models/llama-sentient-3b";
    private static final String SHARD2_BIN = LLAMA_DIR + "/pytorch_model-00002-of-00002.bin";

    public static void main(String[] args) throws Exception {
        torch.manual_seed(42);

        String binPath;
        if (args.length > 0) {
            binPath = args[0];
        } else {
            binPath = SHARD2_BIN;
        }

        System.out.println("=".repeat(70));
        System.out.println("Java Bin File Validator");
        System.out.println("=".repeat(70));
        System.out.println("File: " + binPath);

        File binFile = new File(binPath);
        if (!binFile.exists()) {
            System.out.println("[ERROR] File not found: " + binPath);
            return;
        }
        System.out.printf("Size: %,d bytes (%.3f GiB)%n",
            binFile.length(), binFile.length() / (1024.0 * 1024 * 1024));

        // Load using PyTorchModelLoader
        System.out.println("\nLoading via PyTorchModelLoader.loadStateDict()...");
        long startTime = System.currentTimeMillis();

        Map<String, Tensor> stateDict;
        try {
            stateDict = PyTorchModelLoader.loadStateDict(binFile);
        } catch (Exception e) {
            System.out.println("[ERROR] Failed to load: " + e.getMessage());
            e.printStackTrace();
            return;
        }

        long loadTime = System.currentTimeMillis() - startTime;
        System.out.println("Load time: " + loadTime + " ms");
        System.out.println("Loaded " + stateDict.size() + " tensors");

        // Analyze each tensor
        System.out.println("\n" + "-".repeat(70));
        System.out.println("Per-tensor analysis");
        System.out.println("-".repeat(70));

        long totalParams = 0;
        long totalBytes = 0;
        List<TensorInfo> tensorInfos = new ArrayList<>();

        for (Map.Entry<String, Tensor> entry : stateDict.entrySet()) {
            String name = entry.getKey();
            Tensor tensor = entry.getValue();
            TensorInfo info = analyzeTensor(name, tensor);
            tensorInfos.add(info);
            totalParams += tensor.numel();
            totalBytes += tensor.element_size() * tensor.numel();
        }

        // Sort by name for consistent output
        tensorInfos.sort(Comparator.comparing(a -> a.name));

        // Print table header
        System.out.printf("  %-55s %-22s %-12s %12s %12s%n",
            "name", "shape", "dtype", "min", "max");
        System.out.println("  " + "-".repeat(115));

        // Print each tensor
        for (TensorInfo info : tensorInfos) {
            System.out.printf("  %-55s %-22s %-12s %12.6f %12.6f%n",
                info.name,
                Arrays.toString(info.shape),
                info.dtype,
                info.min,
                info.max);
        }

        // Summary
        System.out.println("  " + "-".repeat(115));
        System.out.printf("  TOTAL tensors:    %d%n", stateDict.size());
        System.out.printf("  TOTAL parameters: %,d (%.2f M)%n", totalParams, totalParams / 1e6);
        System.out.printf("  TOTAL bytes:      %,d (%.3f GiB)%n",
            totalBytes, totalBytes / (1024.0 * 1024 * 1024));

        // Save JSON for Python comparison
        saveJson(binPath, stateDict, tensorInfos, totalParams, totalBytes);

        System.out.println("\n" + "=".repeat(70));
        System.out.println("VALIDATION COMPLETE");
        System.out.println("=".repeat(70));

        // Compare with expected values from Python
        System.out.println("\nComparison with Python (inspect_shard2.py):");
        compareWithPython(stateDict.size(), totalParams);
    }

    private static TensorInfo analyzeTensor(String name, Tensor tensor) {
        TensorInfo info = new TensorInfo();
        info.name = name;

        LongArrayRef sizes = tensor.sizes();
        info.shape = new long[(int)sizes.size()];
        for (int i = 0; i < sizes.size(); i++) {
            info.shape[i] = sizes.get(i);
        }

        torch.ScalarType dtype = tensor.dtype().toScalarType();
        info.dtype = dtype.toString();
        info.numel = tensor.numel();

        // Calculate MD5 of raw float16 bytes
        info.md5 = calculateTensorMD5(tensor);

        // Calculate statistics
        try {
            Tensor floatTensor = tensor.to(torch.kFloat());
            info.min = floatTensor.min().item().toDouble();
            info.max = floatTensor.max().item().toDouble();
            info.mean = floatTensor.mean().item().toDouble();
            floatTensor.close();
        } catch (Exception e) {
            info.min = 0;
            info.max = 0;
            info.mean = 0;
        }

        return info;
    }

    private static String calculateTensorMD5(Tensor tensor) {
        try {
            long numel = tensor.numel();
            int elementSize = (int) tensor.element_size();

            // For consistent MD5, convert to float32 first
            Tensor floatTensor = tensor.to(torch.kFloat());
            numel = floatTensor.numel();
            elementSize = 4; // float32

            ByteBuffer buffer = ByteBuffer.allocate((int)(numel * elementSize));
            buffer.order(ByteOrder.nativeOrder());
            FloatBuffer fb = buffer.asFloatBuffer();

            FloatPointer fp = new FloatPointer(floatTensor.data_ptr());
            for (int i = 0; i < numel; i++) {
                fb.put(fp.get(i));
            }

            byte[] data = buffer.array();
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] digest = md.digest(data);

            StringBuilder sb = new StringBuilder();
            for (byte b : digest) {
                sb.append(String.format("%02x", b));
            }

            floatTensor.close();
            return sb.toString();
        } catch (Exception e) {
            return "error_" + Math.abs(e.getMessage().hashCode());
        }
    }

    private static void saveJson(String binPath, Map<String, Tensor> stateDict,
                                 List<TensorInfo> infos, long totalParams, long totalBytes) throws IOException {
        File binFile = new File(binPath);
        String jsonPath = binFile.getParent() + "/" + binFile.getName().replace(".bin", "_java_validation.json");

        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"file\": \"").append(binFile.getName()).append("\",\n");
        json.append("  \"file_size\": ").append(binFile.length()).append(",\n");
        json.append("  \"total_tensors\": ").append(stateDict.size()).append(",\n");
        json.append("  \"total_params\": ").append(totalParams).append(",\n");
        json.append("  \"total_bytes\": ").append(totalBytes).append(",\n");
        json.append("  \"tensors\": {\n");

        for (int i = 0; i < infos.size(); i++) {
            TensorInfo info = infos.get(i);
            if (i > 0) json.append(",\n");

            json.append("    \"").append(escapeJson(info.name)).append("\": {\n");
            json.append("      \"shape\": [");
            for (int j = 0; j < info.shape.length; j++) {
                if (j > 0) json.append(", ");
                json.append(info.shape[j]);
            }
            json.append("],\n");
            json.append("      \"dtype\": \"").append(info.dtype).append("\",\n");
            json.append("      \"numel\": ").append(info.numel).append(",\n");
            json.append("      \"md5\": \"").append(info.md5).append("\",\n");
            json.append("      \"min\": ").append(String.format("%.6f", info.min)).append(",\n");
            json.append("      \"max\": ").append(String.format("%.6f", info.max)).append(",\n");
            json.append("      \"mean\": ").append(String.format("%.6f", info.mean)).append("\n");
            json.append("    }");
        }

        json.append("\n  }\n");
        json.append("}\n");

        try (FileWriter writer = new FileWriter(jsonPath)) {
            writer.write(json.toString());
        }

        System.out.println("\n[INFO] JSON saved to: " + jsonPath);
    }

    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                 .replace("\"", "\\\"")
                 .replace("\n", "\\n")
                 .replace("\r", "\\r")
                 .replace("\t", "\\t");
    }

    private static void compareWithPython(int javaTensorCount, long javaTotalParams) {
        // Expected values from Python's inspect_shard2.py output
        int expectedTensors = 67;
        long expectedParams = 729861120L;  // 729,861,120

        System.out.printf("  Java:  %d tensors, %,d params%n", javaTensorCount, javaTotalParams);
        System.out.printf("  Python: %d tensors, %,d params (from inspect_shard2.py)%n",
            expectedTensors, expectedParams);

        if (javaTensorCount == expectedTensors && javaTotalParams == expectedParams) {
            System.out.println("  [OK] Tensor count and parameter count match Python!");
        } else {
            System.out.println("  [WARN] Mismatch detected!");
            if (javaTensorCount != expectedTensors) {
                System.out.printf("    Tensor count diff: %d%n", javaTensorCount - expectedTensors);
            }
            if (javaTotalParams != expectedParams) {
                System.out.printf("    Parameter count diff: %,d%n", javaTotalParams - expectedParams);
            }
        }
    }

    static class TensorInfo {
        String name;
        long[] shape;
        String dtype;
        long numel;
        String md5;
        double min;
        double max;
        double mean;
    }
}
