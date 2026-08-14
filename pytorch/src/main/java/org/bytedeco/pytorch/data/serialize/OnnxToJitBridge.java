package org.bytedeco.pytorch.data.serialize;

import org.bytedeco.pytorch.jit.JitModule;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.serving.onnxruntime.ONNXSession;
import org.bytedeco.pytorch.serving.onnxruntime.ONNXModelInfo;
import org.bytedeco.pytorch.serving.onnxruntime.ONNXTensorInfo;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Bridge utilities for ONNX to PyTorch JitModule conversion.
 *
 * <p>This class provides utilities for converting ONNX models to PyTorch JitModule:</p>
 *
 * <ul>
 *   <li>Direct conversion using {@link OnnxToJitConverter} (pure Java)</li>
 *   <li>Detection: Check if an ONNX model is TorchScript-compatible</li>
 *   <li>ONNX to nn.Module: Use ONNXModuleWrapper for inference-only scenarios</li>
 * </ul>
 *
 * <h2>Usage</h2>
 *
 * <pre>
 * {@code
 * // Method 1: Direct conversion (recommended)
 * JitModule jit = OnnxToJitBridge.convertToJitModule(Path.of("model.onnx"));
 *
 * // Method 2: Use ONNXModuleWrapper for inference only
 * Module wrapper = OnnxToJitBridge.loadAsModule(Path.of("model.onnx"));
 * }</pre>
 *
 * @see OnnxToJitConverter For direct ONNX to JitModule conversion
 * @see ONNXSession
 * @see org.bytedeco.pytorch.serving.onnxruntime.ONNXModuleWrapper
 * @see PyTorchModelLoader#loadJitScript(String)
 */
public class OnnxToJitBridge {

    private OnnxToJitBridge() {}

    /**
     * Convert ONNX model directly to JitModule.
     *
     * <p>This is the recommended method for converting ONNX models to PyTorch.</p>
     *
     * @param onnxPath Path to ONNX model file
     * @return JitModule ready for inference and training
     * @throws IOException if conversion fails
     */
    public static JitModule convertToJitModule(Path onnxPath) throws IOException {
        return OnnxToJitConverter.convert(onnxPath);
    }

    /**
     * Convert ONNX model directly to JitModule.
     */
    public static JitModule convertToJitModule(String onnxPath) throws IOException {
        return convertToJitModule(Path.of(onnxPath));
    }

    /**
     * Convert ONNX model bytes to JitModule.
     */
    public static JitModule convertToJitModule(byte[] onnxBytes) throws IOException {
        return OnnxToJitConverter.convert(onnxBytes);
    }

    /**
     * Check if an ONNX file might be convertible to TorchScript.
     *
     * <p>This is a heuristic check based on common patterns. It cannot guarantee
     * compatibility since ONNX opset and PyTorch JIT support differ.</p>
     *
     * @param onnxPath Path to ONNX model
     * @return true if the model appears potentially convertible
     */
    public static boolean isPotentiallyConvertible(Path onnxPath) {
        try {
            ONNXSession session = ONNXSession.load(onnxPath);
            ONNXModelInfo info = session.getModelInfo();

            // Check for IR version (higher is generally better supported)
            long irVersion = info.getIrVersion();
            if (irVersion < 6) {
                session.close();
                return false; // Too old
            }

            // Check input/output types (only tensor I/O is convertible)
            boolean allTensorIO = true;
            for (ONNXTensorInfo input : info.getInputs()) {
                if (input.getElementType() == null) {
                    allTensorIO = false;
                    break;
                }
            }
            if (allTensorIO) {
                for (ONNXTensorInfo output : info.getOutputs()) {
                    if (output.getElementType() == null) {
                        allTensorIO = false;
                        break;
                    }
                }
            }

            session.close();
            return allTensorIO;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Load ONNX model as a PyTorch-compatible Module wrapper.
     *
     * <p>This does NOT create a true JitModule, but provides a similar interface
     * for inference. Training is NOT supported for ONNX models.</p>
     *
     * @param onnxPath Path to ONNX model
     * @return Module wrapper for ONNX inference (NOT a JitModule)
     * @throws IOException if loading fails
     */
    public static org.bytedeco.pytorch.nn.Module loadAsModule(Path onnxPath) throws IOException {
        try {
            ONNXSession session = ONNXSession.load(onnxPath.toString());
            return new org.bytedeco.pytorch.serving.onnxruntime.ONNXModuleWrapper(session);
        } catch (Exception e) {
            throw new IOException("Failed to load ONNX model: " + e.getMessage(), e);
        }
    }

    /**
     * Load ONNX model as a PyTorch-compatible Module wrapper.
     */
    public static org.bytedeco.pytorch.nn.Module loadAsModule(String onnxPath) throws IOException {
        return loadAsModule(Path.of(onnxPath));
    }

    /**
     * Generate a Python script for ONNX to TorchScript conversion.
     *
     * <p>This is the recommended approach since ONNX cannot be directly converted
     * to TorchScript without Python's torch.onnx module.</p>
     *
     * @param onnxPath Path to the ONNX model
     * @return Python code that converts ONNX to TorchScript
     */
    public static String generatePythonConversionScript(Path onnxPath) {
        String modelName = onnxPath.getFileName().toString().replace(".onnx", "");
        String onnxPathStr = onnxPath.toAbsolutePath().toString();
        String outputPath = modelName + "_from_onnx.pt";

        StringBuilder sb = new StringBuilder();
        sb.append("#!/usr/bin/env python\n");
        sb.append("# ONNX to TorchScript Conversion Script\n");
        sb.append("# Generated by JavaCPP PyTorch\n\n");
        sb.append("import torch\n");
        sb.append("import sys\n\n");
        sb.append("ONNX_MODEL = \"").append(onnxPathStr).append("\"\n");
        sb.append("OUTPUT_PATH = \"").append(outputPath).append("\"\n\n");
        sb.append("def main():\n");
        sb.append("    print(f\"Loading ONNX model: {ONNX_MODEL}\")\n");
        sb.append("    \n");
        sb.append("    # Method 1: torch.jit.trace via ONNX Runtime wrapper\n");
        sb.append("    try:\n");
        sb.append("        import onnxruntime as ort\n");
        sb.append("        session = ort.InferenceSession(ONNX_MODEL, providers=['CPUExecutionProvider'])\n");
        sb.append("        \n");
        sb.append("        # Create example input based on model spec\n");
        sb.append("        inputs_meta = session.get_inputs()\n");
        sb.append("        example_input = []\n");
        sb.append("        for inp in inputs_meta:\n");
        sb.append("            shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in inp.shape)\n");
        sb.append("            example_input.append(torch.randn(*shape))\n");
        sb.append("        \n");
        sb.append("        # Wrap ONNX model as a torch.nn.Module\n");
        sb.append("        class OnnxWrapper(torch.nn.Module):\n");
        sb.append("            def __init__(self, onnx_path):\n");
        sb.append("                super().__init__()\n");
        sb.append("                import onnxruntime as ort\n");
        sb.append("                self.session = ort.InferenceSession(\n");
        sb.append("                    onnx_path,\n");
        sb.append("                    providers=['CPUExecutionProvider']\n");
        sb.append("                )\n");
        sb.append("                self.input_names = [i.name for i in self.session.get_inputs()]\n");
        sb.append("            \n");
        sb.append("            def forward(self, *args):\n");
        sb.append("                inputs = {self.input_names[i]: args[i].detach().cpu().numpy()\n");
        sb.append("                         for i in range(len(args))}\n");
        sb.append("                outputs = self.session.run(None, inputs)\n");
        sb.append("                return tuple(torch.from_numpy(o) for o in outputs)\n");
        sb.append("        \n");
        sb.append("        model = OnnxWrapper(ONNX_MODEL)\n");
        sb.append("        example = tuple(example_input)\n");
        sb.append("        \n");
        sb.append("        # Trace the wrapped model\n");
        sb.append("        with torch.no_grad():\n");
        sb.append("            traced = torch.jit.trace(model, example)\n");
        sb.append("        traced.save(OUTPUT_PATH)\n");
        sb.append("        print(f\"SUCCESS: TorchScript saved to {OUTPUT_PATH}\")\n");
        sb.append("        return 0\n");
        sb.append("    except ImportError:\n");
        sb.append("        print(\"onnxruntime not available, trying torch.onnx methods...\")\n");
        sb.append("    except Exception as e:\n");
        sb.append("        print(f\"Method 1 failed: {e}\")\n");
        sb.append("    \n");
        sb.append("    # Method 2: torch.onnx.export with TORCH_SCRIPT (if model is already a torch.nn.Module)\n");
        sb.append("    print(\"Note: If the ONNX model originated from PyTorch and you have the source,\")\n");
        sb.append("    print(\"you can directly save it as TorchScript with:\")\n");
        sb.append("    print(\"    scripted = torch.jit.script(model)\")\n");
        sb.append("    print(f\"    scripted.save('{outputPath}')\")\n");
        sb.append("    return 1\n\n");
        sb.append("if __name__ == \"__main__\":\n");
        sb.append("    sys.exit(main())\n");

        return sb.toString();
    }

    /**
     * Check if ONNX ops used by the model are supported by PyTorch JIT.
     *
     * @param onnxPath Path to ONNX model
     * @return Report of potential compatibility issues
     */
    public static String checkCompatibility(Path onnxPath) {
        StringBuilder report = new StringBuilder();
        report.append("ONNX to TorchScript Compatibility Report\n");
        report.append("=".repeat(50));
        report.append("\n\n");

        try {
            ONNXSession session = ONNXSession.load(onnxPath);
            ONNXModelInfo info = session.getModelInfo();

            report.append("Model: ").append(onnxPath.getFileName()).append("\n");
            report.append("IR Version: ").append(info.getIrVersion()).append("\n");
            report.append("Producer Version: ").append(info.getVersion()).append("\n\n");

            // Check inputs
            report.append("Inputs:\n");
            for (ONNXTensorInfo input : info.getInputs()) {
                report.append("  - ").append(input.getName())
                       .append(" [").append(input.getShapeString()).append("]\n");
            }

            // Check outputs
            report.append("\nOutputs:\n");
            for (ONNXTensorInfo output : info.getOutputs()) {
                report.append("  - ").append(output.getName())
                       .append(" [").append(output.getShapeString()).append("]\n");
            }

            session.close();

            report.append("\n");
            report.append("-".repeat(50));
            report.append("\n");
            report.append("Compatibility Assessment:\n\n");
            report.append("WARNING: ONNX models CANNOT be directly converted to TorchScript\n");
            report.append("from Java. Use the Python conversion script generated by:\n\n");
            report.append("    String script = OnnxToJitBridge.generatePythonConversionScript(path);\n\n");
            report.append("Then load the resulting .pt file with:\n");
            report.append("    JitModule jit = PyTorchModelLoader.loadJitScript(\"model_from_onnx.pt\");\n");

        } catch (Exception e) {
            report.append("Error analyzing model: ").append(e.getMessage()).append("\n");
        }

        return report.toString();
    }

    /**
     * Temporary file holder for ONNX to JIT conversion workflow.
     */
    public static class ConversionWorkspace implements AutoCloseable {
        private final Path tempDir;

        public ConversionWorkspace() throws IOException {
            this.tempDir = Files.createTempDirectory("onnx2jit");
        }

        public Path getTempDir() {
            return tempDir;
        }

        public Path getPythonScriptPath() {
            return tempDir.resolve("convert_onnx_to_jit.py");
        }

        public Path getOutputPath(String onnxFileName) {
            String base = onnxFileName.replace(".onnx", "");
            return tempDir.resolve(base + "_jit.pt");
        }

        @Override
        public void close() {
            try {
                if (Files.exists(tempDir)) {
                    try (java.util.stream.Stream<Path> stream = Files.walk(tempDir)) {
                        stream.sorted(java.util.Comparator.reverseOrder())
                            .forEach(p -> {
                                try {
                                    Files.deleteIfExists(p);
                                } catch (IOException ignored) {
                                    // best effort cleanup
                                }
                            });
                    }
                }
            } catch (IOException ignored) {
                // best effort cleanup
            }
        }
    }
}
