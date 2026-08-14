package org.bytedeco.pytorch.data.serialize;

import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.DeviceType;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.nn.Module;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Bridge layer between nn.Module (trainable) and JitModule (inference-only).
 *
 * <p>This class provides utilities to convert between different module representations:</p>
 * <ul>
 *   <li><b>WeightBagModule</b> → Trainable nn.Module with typed layers</li>
 *   <li><b>JitModule</b> → TorchScript model for inference (pre-compiled)</li>
 *   <li><b>StateDict</b> → Raw Map&lt;String, Tensor&gt; weights</li>
 * </ul>
 *
 * <p>Note: Full nn.Module → JitModule conversion requires native JNI.
 * Use {@link #loadJitScript(Path)} for loading pre-compiled TorchScript models.</p>
 *
 * <pre>{@code
 * // Example: Load TorchScript for inference
 * JitModule model = JitBridge.loadJitScript(Path.of("model.pt"));
 * IValueVector inputs = new IValueVector();
 * inputs.add(IValue.from(tensor));
 * IValue output = model.forward(inputs);
 *
 * // Example: Load weights as trainable module
 * WeightBagModule bag = WeightBagModule.fromPythonPth("model.pth");
 * Adam optimizer = new Adam(bag.parameters(), new AdamOptions(1e-4));
 * }</pre>
 */
public final class JitBridge {

    private JitBridge() {}

    // ---- TorchScript Loading ----

    /**
     * Load a TorchScript model from file.
     *
     * @param path Path to the TorchScript .pt file
     * @return Loaded JitModule ready for inference
     * @throws IOException if file cannot be read
     */
    public static JitModule loadJitScript(Path path) throws IOException {
        return torch.load(path.toString());
    }

    /**
     * Load a TorchScript model from file with device specification.
     *
     * @param path   Path to the TorchScript .pt file
     * @param device Target device (e.g., "cpu" or "cuda:0")
     * @return Loaded JitModule on specified device
     * @throws IOException if file cannot be read
     */
    public static JitModule loadJitScript(Path path, Device device) throws IOException {
        ExtraFilesMap extras = new ExtraFilesMap();
        return torch.load(path.toString(), new DeviceOptional(device), extras);
    }

    /**
     * Load a TorchScript model from file.
     *
     * @param file File object for the TorchScript .pt file
     * @return Loaded JitModule ready for inference
     * @throws IOException if file cannot be read
     */
    public static JitModule loadJitScript(File file) throws IOException {
        return loadJitScript(file.toPath());
    }

    /**
     * Load a TorchScript model with device specification.
     *
     * @param file   File object for the TorchScript .pt file
     * @param device Target device
     * @return Loaded JitModule on specified device
     * @throws IOException if file cannot be read
     */
    public static JitModule loadJitScript(File file, Device device) throws IOException {
        return loadJitScript(file.toPath(), device);
    }

    /**
     * Load a TorchScript model from InputStream (saved to temp file).
     *
     * @param in     Input stream containing TorchScript data
     * @param device Target device
     * @return Loaded JitModule
     * @throws IOException if stream cannot be read
     */
    public static JitModule loadJitScript(InputStream in, Device device) throws IOException {
        Path tmp = Files.createTempFile("jit_model", ".pt");
        try {
            Files.copy(in, tmp, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            return loadJitScript(tmp, device);
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    // ---- Type Detection ----

    /**
     * Detect if a file is a TorchScript model.
     *
     * @param path Path to check
     * @return true if the file appears to be a TorchScript model
     */
    public static boolean isTorchScript(Path path) {
        return isTorchScript(path.toFile());
    }

    /**
     * Detect if a file is a TorchScript model.
     *
     * @param file File to check
     * @return true if the file appears to be a TorchScript model
     */
    public static boolean isTorchScript(File file) {
        String name = file.getName().toLowerCase();
        if (name.endsWith(".pt") || name.endsWith(".pth")) {
            return isTorchScriptMagic(file);
        }
        return false;
    }

    private static boolean isTorchScriptMagic(File file) {
        try (FileInputStream fis = new FileInputStream(file)) {
            byte[] magic = new byte[8];
            int read = fis.read(magic);
            if (read >= 8) {
                // TorchScript files typically start with "PYTORCH" or similar
                String header = new String(magic, 0, 8);
                return header.startsWith("PYTORCH");
            }
        } catch (IOException e) {
            // Ignore and return false
        }
        return false;
    }

    // ---- Inference Helpers ----

    /**
     * Run inference with a single tensor input.
     *
     * @param model   The JitModule
     * @param input   Input tensor
     * @return Output tensor
     */
    public static Tensor infer(JitModule model, Tensor input) {
        IValueVector inputs = new IValueVector();
        inputs.push_back(new IValue(input));
        IValue output = model.forward(inputs);
        return output.toTensor();
    }

    /**
     * Run inference with multiple tensor inputs.
     *
     * @param model   The JitModule
     * @param inputs  Input tensors
     * @return Output tensor
     */
    public static Tensor infer(JitModule model, Tensor... inputs) {
        IValueVector inputsVec = new IValueVector();
        for (Tensor t : inputs) {
            inputsVec.push_back(new IValue(t));
        }
        IValue output = model.forward(inputsVec);
        return output.toTensor();
    }

    // ---- Module to JitModule (Requires JNI) ----

    /**
     * Convert an nn.Module to TorchScript format.
     *
     * <p><b>Note:</b> This requires native JNI implementation to wrap the Python
     * torch.jit.script() or torch.jit.trace() functionality. Currently throws
     * UnsupportedOperationException.</p>
     *
     * <p>Alternative approaches:</p>
     * <ul>
     *   <li>Export the model to TorchScript from Python before loading</li>
     *   <li>Use {@link WeightBagModule#saveNative(File)} for native checkpoint format</li>
     *   <li>Create a custom JitModule subclass and populate it manually</li>
     * </ul>
     *
     * @param module The nn.Module to convert
     * @param method "script" or "trace"
     * @return A JitModule (when JNI is implemented)
     * @throws UnsupportedOperationException always (JNI not implemented)
     */
    public static JitModule moduleToJitScript(Module module, String method) {
        throw new UnsupportedOperationException(
            "nn.Module to TorchScript conversion requires native JNI implementation. " +
            "Export the model to TorchScript from Python using:\n" +
            "  model = torch.jit.script(your_model)\n" +
            "  model.save('model.pt')"
        );
    }

    /**
     * Convert using torch.jit.script.
     */
    public static JitModule script(Module module) {
        return moduleToJitScript(module, "script");
    }

    /**
     * Convert using torch.jit.trace with example input.
     */
    public static JitModule trace(Module module, Tensor exampleInput) {
        throw new UnsupportedOperationException(
            "torch.jit.trace requires an example input tensor. " +
            "This overload is not yet implemented."
        );
    }

    // ---- Saving (Requires JNI) ----

    /**
     * Save a JitModule to file.
     *
     * @param model  The JitModule to save
     * @param path   Output path
     * @param extras Extra files (can be empty)
     */
    public static void save(JitModule model, Path path, ExtraFilesMap extras) {
        model.save(path.toString(), extras);
    }

    /**
     * Save a JitModule to file.
     *
     * @param model  The JitModule to save
     * @param path   Output path
     */
    public static void save(JitModule model, Path path) {
        model.save(path.toString());
    }

    /**
     * Save an nn.Module as TorchScript.
     *
     * <p><b>Note:</b> Requires native JNI for nn.Module → TorchScript conversion.</p>
     *
     * @param module The module to save
     * @param path   Output path
     * @throws UnsupportedOperationException if JNI is not available
     */
    public static void saveAsTorchScript(Module module, Path path) {
        throw new UnsupportedOperationException(
            "Saving nn.Module as TorchScript requires native JNI implementation. " +
            "Use WeightBagModule.saveNative() for native checkpoint format."
        );
    }
}
