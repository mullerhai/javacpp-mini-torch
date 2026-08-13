package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.data.numpy.NP;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.optim.options.*;

import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.jit.JitModule;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.data.safetensors.LoadOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.DeviceType;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.zip.ZipFile;

/**
 * Universal PyTorch model loader supporting all formats.
 *
 * <p>Supported formats:
 * <ul>
 *   <li><b>TorchScript (.pt/.pth)</b> — {@code torch.jit.load()} → {@link JitModule}</li>
 *   <li><b>StateDict (.pt/.pth/.safetensors)</b> — tensor map → {@link WeightBagModule}</li>
 *   <li><b>Pickle (.pkl)</b> — Python pickle objects</li>
 *   <li><b>HuggingFace directories</b> — config.json + sharded weights</li>
 * </ul>
 *
 * <p>Usage:
 * <pre>{@code
 * // 1. Load TorchScript model directly for inference
 * JitModule model = PyTorchModelLoader.loadJitScript("model.pt");
 * IValueVector inputs = new IValueVector();
 * inputs.add(IValue.from(tensor));
 * IValue output = model.forward(inputs);
 *
 * // 2. Load state_dict and convert to trainable Module
 * WeightBagModule model = PyTorchModelLoader.loadAsModule("model.safetensors");
 * Adam optimizer = new Adam(model.parameters(), new AdamOptions(1e-4));
 * // ... training loop
 *
 * // 3. Universal load — auto-detect format
 * PyTorchModelLoader.ModelType type = PyTorchModelLoader.detectModelType(file);
 * if (type == PyTorchModelLoader.ModelType.TORCH_SCRIPT) {
 *     JitModule jit = PyTorchModelLoader.loadJitScript(file);
 * } else {
 *     WeightBagModule mod = PyTorchModelLoader.loadAsModule(file);
 * }
 * }</pre>
 */
public final class PyTorchModelLoader {

    public enum ModelType {
        /** TorchScript model (.pt saved with torch.jit.script(...).save()) */
        TORCH_SCRIPT,
        /** State-dict checkpoint (.pt/.pth/.safetensors with tensor map) */
        STATE_DICT,
        /** Pickle file with arbitrary Python objects */
        PICKLE,
        /** HuggingFace model directory (config.json + sharded weights) */
        HUGGINGFACE,
        /** HDF5 binary format */
        HDF5,
        /** Custom binary format (MicroLens .bin) */
        BIN_MICROLENS,
        /** Generic binary format */
        BIN_GENERIC,
        /** NumPy array format */
        NUMPY,
        /** Unknown format */
        UNKNOWN
    }

    public enum LoadMode {
        /** Load as TorchScript JitModule (for inference) */
        JitModule,
        /** Load as trainable nn.Module via WeightBagModule */
        Module,
        /** Load as Map<String, Tensor> (raw weights) */
        StateDict,
        /** Auto-detect and load appropriately */
        Auto
    }

    private PyTorchModelLoader() {}

    // ---- Model Type Detection ----

    /**
     * Detect the type of a PyTorch model file.
     */
    public static ModelType detectModelType(File file) throws IOException {
        if (file == null || !file.isFile()) {
            return ModelType.UNKNOWN;
        }

        String name = file.getName().toLowerCase();
        byte[] magic = new byte[8];
        try (InputStream in = Files.newInputStream(file.toPath())) {
            int read = in.read(magic);
            if (read < 4) return ModelType.UNKNOWN;

            // Check ZIP magic (modern torch.save format)
            if (magic[0] == 'P' && magic[1] == 'K' && magic[2] == 3 && magic[3] == 4) {
                return detectZipContents(file);
            }

            // Check safetensors magic (8-byte header)
            if (read >= 8 && magic[0] == 0x82 && magic[1] == 0x00 && magic[2] == 0x00 && magic[3] == 0x00) {
                return ModelType.STATE_DICT;
            }

            // Check pickle protocol
            if (magic[0] >= 0x80 && magic[0] <= 0x8F) {
                return ModelType.PICKLE;
            }

            // Check .bin format - could be:
            // 1. PyTorch legacy pickle (BINPUT, BINGET, etc.)
            // 2. Custom binary format with tensor data
            // 3. HDF5 binary format
            if (name.endsWith(".bin")) {
                return detectBinFormat(file, magic);
            }
        }

        // Check if it's a directory
        if (file.isDirectory()) {
            if (new File(file, "config.json").exists()) {
                return ModelType.HUGGINGFACE;
            }
            return ModelType.STATE_DICT;
        }

        // Check extension
        if (name.endsWith(".pt") || name.endsWith(".pth")) {
            return ModelType.STATE_DICT;
        }
        if (name.endsWith(".safetensors")) {
            return ModelType.STATE_DICT;
        }
        if (name.endsWith(".pkl") || name.endsWith(".pickle")) {
            return ModelType.PICKLE;
        }
        if (name.endsWith(".bin")) {
            return ModelType.BIN_GENERIC; // .bin files default to BIN format
        }

        return ModelType.UNKNOWN;
    }

    /**
     * Detect specific .bin format based on file structure.
     */
    private static ModelType detectBinFormat(File file, byte[] magic) throws IOException {
        // Check if it's a torch.save ZIP with .bin extension
        if (magic[0] == 'P' && magic[1] == 'K') {
            return detectZipContents(file);
        }

        // Check for HDF5 magic
        if (magic[0] == (byte)0x89 && magic[1] == 'H' && magic[2] == 'D' && magic[3] == 'F') {
            return ModelType.HDF5;
        }

        // Check for our custom binary format (MicroLens)
        if (magic[0] == 'M' && magic[1] == 'L' && magic[2] == 'N' && magic[3] == 'S') {
            return ModelType.BIN_MICROLENS;
        }

        // Check for numpy .npy format
        if (magic[0] == (byte)0x93 && magic[1] == 'N' && magic[2] == 'U' && magic[3] == 'M') {
            return ModelType.NUMPY;
        }

        // Check for pickle protocol in binary file
        if (magic[0] >= 0x80 && magic[0] <= 0x8F) {
            return ModelType.PICKLE;
        }

        // Default: treat as binary state-dict
        return ModelType.BIN_GENERIC;
    }

    public static ModelType detectModelType(Path path) throws IOException {
        return detectModelType(path.toFile());
    }

    public static ModelType detectModelType(String path) throws IOException {
        return detectModelType(new File(path));
    }

    private static ModelType detectZipContents(File file) throws IOException {
        try (ZipFile zip = new ZipFile(file)) {
            // Check for TorchScript markers
            if (zip.getEntry("data/code/__torch__.py") != null ||
                zip.getEntry("data/torch.jit.JitModule") != null ||
                zip.getEntry("model.jit") != null) {
                return ModelType.TORCH_SCRIPT;
            }

            // Check for state-dict markers
            if (zip.getEntry("data.pkl") != null && zip.getEntry("byteorder") != null) {
                return ModelType.STATE_DICT;
            }

            // Default to state-dict for torch.save ZIP
            return ModelType.STATE_DICT;
        } catch (IOException e) {
            return ModelType.UNKNOWN;
        }
    }

    // ---- TorchScript Loading (JitModule) ----

    /**
     * Load a TorchScript model from file.
     * This is the fastest path for inference as it uses LibTorch's native JIT compiler.
     *
     * @param file Path to TorchScript .pt file
     * @return Loaded JitModule ready for inference
     */
    public static JitModule loadJitScript(File file) {
        return torch.load(file.getAbsolutePath());
    }

    /**
     * Load a TorchScript model to specific device.
     */
    public static JitModule loadJitScript(File file, Device device) {
        return torch.load(file.getAbsolutePath(), new DeviceOptional(device), new ExtraFilesMap());
    }

    /**
     * Load TorchScript model from Path.
     */
    public static JitModule loadJitScript(Path path) {
        return loadJitScript(path.toFile());
    }

    /**
     * Load TorchScript model from String path.
     */
    public static JitModule loadJitScript(String path) {
        return loadJitScript(new File(path));
    }

    /**
     * Load TorchScript model with custom extras.
     */
    public static JitModule loadJitScript(File file, Device device, ExtraFilesMap extras) {
        return torch.load(file.getAbsolutePath(), new DeviceOptional(device), extras);
    }

    /**
     * Load TorchScript model from InputStream.
     */
    public static JitModule loadJitScript(InputStream in) {
        return loadJitScript(in, new Device(DeviceType.CPU));
    }

    /**
     * Load TorchScript model from InputStream with device.
     */
    public static JitModule loadJitScript(InputStream in, Device device) {
        try {
            Path tmp = Files.createTempFile("jit_model", ".pt");
            Files.copy(in, tmp, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            JitModule result = torch.load(tmp.toString(), new DeviceOptional(device), new ExtraFilesMap());
            Files.deleteIfExists(tmp);
            return result;
        } catch (IOException e) {
            throw new RuntimeException("Failed to load TorchScript from stream", e);
        }
    }

    // ---- StateDict Loading ----

    /**
     * Load model weights as Map&lt;String, Tensor&gt;.
     * This is the most flexible loading mode.
     */
    public static Map<String, Tensor> loadStateDict(File file) throws IOException {
        return ModelWeights.load(file);
    }

    public static Map<String, Tensor> loadStateDict(Path path) throws IOException {
        return ModelWeights.load(path);
    }

    public static Map<String, Tensor> loadStateDict(String path) throws IOException {
        return ModelWeights.load(path);
    }

    public static Map<String, Tensor> loadStateDict(File file, LoadOptions opts) throws IOException {
        return ModelWeights.load(file, opts);
    }

    // ---- Module Loading (WeightBagModule) ----

    /**
     * Load weights and convert to a trainable Module.
     * The module structure mirrors the original Python model.
     *
     * @param file Weight file (.safetensors, .pt, .pth)
     * @return Trainable WeightBagModule
     */
    public static WeightBagModule loadAsModule(File file) throws IOException {
        return ModelWeights.toModule(file);
    }

    public static WeightBagModule loadAsModule(Path path) throws IOException {
        return loadAsModule(path.toFile());
    }

    public static WeightBagModule loadAsModule(String path) throws IOException {
        return loadAsModule(new File(path));
    }

    public static WeightBagModule loadAsModule(File file, boolean requiresGrad) throws IOException {
        return ModelWeights.toModule(file, requiresGrad);
    }

    /**
     * Load weights into an existing Module.
     *
     * @param module Target module to load weights into
     * @param file Weight file
     * @param strict Whether to require all keys to match
     * @return Number of parameters loaded
     */
    public static int loadIntoModule(Module module, File file, boolean strict) throws IOException {
        return ModelWeights.loadIntoModule(module, file, strict);
    }

    public static int loadIntoModule(Module module, File file, LoadOptions opts) throws IOException {
        return ModelWeights.loadIntoModule(module, file, opts);
    }

    // ---- Binary Format Loading ----

    /**
     * Load binary model files (.bin).
     * Supports multiple binary formats based on magic bytes detection.
     */
    public static Map<String, Tensor> loadBin(File file) throws IOException {
        return loadBin(file, LoadOptions.defaults());
    }

    public static Map<String, Tensor> loadBin(File file, LoadOptions opts) throws IOException {
        ModelType type = detectModelType(file);
        switch (type) {
            case BIN_MICROLENS:
                return loadMicroLensBin(file, opts);
            case HDF5:
                return loadHdf5Bin(file);
            case NUMPY:
                return loadNumpyBin(file);
            case PICKLE:
                return ModelWeights.load(file, opts);
            default:
                // Try to treat as generic binary tensor format
                return loadGenericBin(file);
        }
    }

    /**
     * Load MicroLens .bin format (custom binary format).
     * Format: MLNS + version(4) + num_tensors(4) + [name_len(4) + name + dtype(4) + ndims(4) + dims[n] + data]
     */
    private static Map<String, Tensor> loadMicroLensBin(File file, LoadOptions opts) throws IOException {
        Map<String, Tensor> result = new LinkedHashMap<>();
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            byte[] magic = new byte[4];
            raf.readFully(magic);
            if (magic[0] != 'M' || magic[1] != 'L' || magic[2] != 'N' || magic[3] != 'S') {
                throw new IOException("Not a MicroLens binary file: " + file);
            }

            int version = raf.readInt();
            int numTensors = raf.readInt();

            for (int i = 0; i < numTensors; i++) {
                int nameLen = raf.readInt();
                byte[] nameBytes = new byte[nameLen];
                raf.readFully(nameBytes);
                String name = new String(nameBytes, StandardCharsets.UTF_8);

                int dtype = raf.readInt();
                int ndims = raf.readInt();
                long[] dims = new long[ndims];
                for (int d = 0; d < ndims; d++) {
                    dims[d] = raf.readLong();
                }

                ScalarType scalarType = ScalarType.values()[dtype];
                long numel = 1;
                for (long d : dims) numel *= d;

                FileChannel channel = raf.getChannel();
                long pos = channel.position();
                long dataSize = numel * elementSize(scalarType);
                java.nio.MappedByteBuffer buf = channel.map(FileChannel.MapMode.READ_ONLY, pos, dataSize);
                org.bytedeco.javacpp.BytePointer ptr = new org.bytedeco.javacpp.BytePointer(buf);
                TensorOptions topts = new TensorOptions().dtype(new ScalarTypeOptional(scalarType));
                if (opts.mapLocation() != null) {
                    topts = topts.device(new DeviceOptional(opts.mapLocation()));
                }
                Tensor tensor = torch.from_blob(ptr, dims, topts);
                channel.position(pos + dataSize);

                result.put(name, tensor);
            }
        }
        return SafeTensors.applyMapLocation(result, opts);
    }

    /**
     * Load HDF5 binary format.
     */
    private static Map<String, Tensor> loadHdf5Bin(File file) throws IOException {
        try {
            // Use existing HDF5 reader if available
            Class<?> hdf5ReaderClass = Class.forName("org.bytedeco.pytorch.dataframe.hdf5.Hdf5Reader");
            java.lang.reflect.Method readMethod = hdf5ReaderClass.getMethod("read", File.class);
            @SuppressWarnings("unchecked")
            Map<String, Tensor> result = (Map<String, Tensor>) readMethod.invoke(null, file);
            return result;
        } catch (Exception e) {
            throw new IOException("Failed to load HDF5 file: " + file, e);
        }
    }

    /**
     * Load NumPy .npy format.
     */
    private static Map<String, Tensor> loadNumpyBin(File file) throws IOException {
        Tensor tensor = NP.load(file.getPath()).toTensor();
        Map<String, Tensor> result = new LinkedHashMap<>();
        result.put(file.getName().replace(".npy", ""), tensor);
        return result;
    }

    /**
     * Load generic binary tensor format.
     * Assumes: dtype(4) + ndims(4) + dims[n] + data
     */
    private static Map<String, Tensor> loadGenericBin(File file) throws IOException {
        Map<String, Tensor> result = new LinkedHashMap<>();
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            long fileLen = raf.length();
            long pos = 0;

            while (pos < fileLen - 8) {
                raf.seek(pos);
                int dtype = raf.readInt();
                int ndims = raf.readInt();

                long[] dims = new long[ndims];
                for (int d = 0; d < ndims; d++) {
                    dims[d] = raf.readLong();
                }

                ScalarType scalarType = dtype < ScalarType.values().length ?
                    ScalarType.values()[dtype] : ScalarType.Float;

                long numel = 1;
                for (long d : dims) numel *= d;
                long dataSize = numel * elementSize(scalarType);

                FileChannel channel = raf.getChannel();
                long dataPos = channel.position();
                java.nio.MappedByteBuffer buf = channel.map(FileChannel.MapMode.READ_ONLY, dataPos, dataSize);
                org.bytedeco.javacpp.BytePointer ptr = new org.bytedeco.javacpp.BytePointer(buf);
                TensorOptions topts = new TensorOptions().dtype(new ScalarTypeOptional(scalarType));
                Tensor tensor = torch.from_blob(ptr, dims, topts);

                result.put("tensor_" + result.size(), tensor);

                pos = dataPos + dataSize;
            }
        }
        return result;
    }

    private static long elementSize(ScalarType dtype) {
        if (dtype == null) return 4;
        switch (dtype) {
            case ScalarType.Float:   return 4;
            case ScalarType.Double:  return 8;
            case ScalarType.Int:     return 4;
            case ScalarType.UInt8:   return 4;
            case ScalarType.Short:   return 2;
            case ScalarType.Long:    return 8;
            case ScalarType.Byte:    return 1;
            case ScalarType.Char:    return 1;
            case ScalarType.Bool:    return 1;
            case ScalarType.Half:   return 2;
            case ScalarType.BFloat16: return 2;
            default:                return 4;
        }
    }

    // ---- Universal Loading ----

    /**
     * Load model using the specified mode.
     */
    public static Object load(File file, LoadMode mode) throws IOException {
        switch (mode) {
            case JitModule:
                return loadJitScript(file);
            case Module:
                return loadAsModule(file);
            case StateDict:
                return loadStateDict(file);
            case Auto:
            default:
                ModelType type = detectModelType(file);
                switch (type) {
                    case TORCH_SCRIPT:
                        return loadJitScript(file);
                    case STATE_DICT:
                        return loadAsModule(file);
                    case HUGGINGFACE:
                        return loadHuggingFaceModel(file);
                    case PICKLE:
                    case BIN_MICROLENS:
                    case HDF5:
                    case NUMPY:
                    case BIN_GENERIC:
                        return loadBin(file);
                    default:
                        throw new IOException("Cannot determine model type for: " + file);
                }
        }
    }

    public static Object load(Path path, LoadMode mode) throws IOException {
        return load(path.toFile(), mode);
    }

    public static Object load(String path, LoadMode mode) throws IOException {
        return load(new File(path), mode);
    }

    /**
     * Universal load with options.
     */
    public static Object load(File file, LoadMode mode, LoadOptions opts) throws IOException {
        if (mode == LoadMode.StateDict) {
            return ModelWeights.load(file, opts);
        }
        if (mode == LoadMode.Module) {
            return ModelWeights.toModule(file, opts.weightsOnly, opts);
        }
        return load(file, mode);
    }

    // ---- HuggingFace Specific ----

    /**
     * Load a HuggingFace model directory.
     * Requires config.json and weight files in the directory.
     */
    public static WeightBagModule loadHuggingFaceModel(File dir) throws IOException {
        if (!dir.isDirectory()) {
            throw new IOException("Not a directory: " + dir);
        }
        File configFile = new File(dir, "config.json");
        if (!configFile.exists()) {
            throw new IOException("Missing config.json in: " + dir);
        }
        return LLMModuleBuilder.fromHuggingFace(dir.toPath());
    }

    public static WeightBagModule loadHuggingFaceModel(Path dir) throws IOException {
        return loadHuggingFaceModel(dir.toFile());
    }

    public static WeightBagModule loadHuggingFaceModel(String dir) throws IOException {
        return loadHuggingFaceModel(new File(dir));
    }

    // ---- Inference Helpers ----

    /**
     * Run inference on a JitModule with tensor inputs.
     */
    public static Tensor infer(JitModule model, Tensor... inputs) {
        IValueVector inputIValues = new IValueVector();
        for (Tensor t : inputs) {
            inputIValues.push_back(new IValue(t));
        }
        IValue output = model.forward(inputIValues);
        if (output.isTensor()) {
            return output.toTensor();
        }
        throw new RuntimeException("Model output is not a tensor: " + output.tagKind());
    }

    /**
     * Run inference with List[Tensor] input.
     */
    public static Tensor infer(JitModule model, List<Tensor> inputs) {
        IValueVector inputIValues = new IValueVector();
        for (Tensor t : inputs) {
            inputIValues.push_back(new IValue(t));
        }
        IValue output = model.forward(inputIValues);
        if (output.isTensor()) {
            return output.toTensor();
        }
        throw new RuntimeException("Model output is not a tensor: " + output.tagKind());
    }

    /**
     * Run inference and return IValue (for complex outputs).
     */
    public static IValue inferRaw(JitModule model, Tensor... inputs) {
        IValueVector inputIValues = new IValueVector();
        for (Tensor t : inputs) {
            inputIValues.push_back(new IValue(t));
        }
        return model.forward(inputIValues);
    }

    /**
     * Run inference on a Module (WeightBagModule or custom).
     */
    public static Tensor infer(Module model, Tensor... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("At least one input tensor required");
        }
        if (inputs.length == 1) {
            return model.forward(inputs[0]);
        }
        // Multi-input: pass as varargs — nn.Module.forward(Tensor, long...) or single Tensor
        if (model instanceof WeightBagModule) {
            return model.forward(inputs[0]);
        }
        // For multi-input modules, try first tensor
        return model.forward(inputs[0]);
    }

    /**
     * Convenience: load and infer in one call.
     */
    public static Tensor loadAndInfer(File file, Tensor... inputs) throws IOException {
        ModelType type = detectModelType(file);
        if (type == ModelType.TORCH_SCRIPT) {
            JitModule jit = loadJitScript(file);
            return infer(jit, inputs);
        } else {
            WeightBagModule mod = loadAsModule(file);
            return infer(mod, inputs);
        }
    }

    // ---- Training Helpers ----

    /**
     * Prepare a loaded model for training.
     */
    public static void prepareTraining(Module module) {
        module.train(true);
    }

    /**
     * Prepare a loaded model for evaluation.
     */
    public static void prepareEval(Module module) {
        module.eval();
    }

    /**
     * Freeze model parameters (set requiresGrad=false).
     */
    public static void freeze(Module module) {
        TensorVector params = module.parameters();
        for (TensorVector.Iterator it = params.begin(), end = params.end(); !it.equals(end); it.increment()) {
            it.get().set_requires_grad(false);
        }
    }

    /**
     * Freeze parameters with prefix match.
     */
    public static void freezePrefix(Module module, String prefix) {
        if (module instanceof WeightBagModule) {
            ((WeightBagModule) module).freezePrefix(prefix);
        } else {
            // Manual freeze for custom modules
            StringTensorDict params = module.named_parameters();
            for (StringTensorDictItemVector.Iterator it = params.begin(), end = params.end(); !it.equals(end); it.increment()) {
                StringTensorDictItem item = it.get();
                if (item.key().getString().startsWith(prefix)) {
                    item.value().set_requires_grad(false);
                }
            }
        }
    }

    // ---- Model Info ----

    /**
     * Print model structure to stdout.
     */
    public static void printStructure(File file) throws IOException {
        ModelType type = detectModelType(file);
        System.out.println("Model: " + file.getName());
        System.out.println("Type: " + type);

        if (type == ModelType.TORCH_SCRIPT) {
            System.out.println("\n--- TorchScript Model ---");
            JitModule model = loadJitScript(file);
            printJitModuleStructure(model);
        } else {
            System.out.println("\n--- State-Dict ---");
            Map<String, Tensor> sd = loadStateDict(file);
            printStateDict(sd);
        }
    }

    public static void printStructure(Path path) throws IOException {
        printStructure(path.toFile());
    }

    public static void printStructure(String path) throws IOException {
        printStructure(new File(path));
    }

    private static void printJitModuleStructure(JitModule model) {
        System.out.println("TorchScript module methods:");
        // Note: accessing method names requires more jit introspection
        // This is a basic implementation
    }

    private static void printStateDict(Map<String, Tensor> sd) {
        long totalParams = 0;
        long totalBytes = 0;
        for (Map.Entry<String, Tensor> entry : sd.entrySet()) {
            Tensor t = entry.getValue();
            System.out.printf("  %-60s %-15s %s%n",
                entry.getKey(),
                formatShape(t.sizes()),
                t.dtype().toString());
            totalParams += t.numel();
            totalBytes += t.element_size() * t.numel();
        }
        System.out.printf("%nTotal parameters: %d (%.2f M)%n", totalParams, totalParams / 1e6);
        System.out.printf("Total size: %.2f MB%n", totalBytes / (1024.0 * 1024.0));
    }

    private static String formatShape(LongArrayRef sizes) {
        if (sizes == null || sizes.size() == 0) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < sizes.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(sizes.get(i));
        }
        sb.append("]");
        return sb.toString();
    }

    // ---- Serialization ----

    /**
     * Save a Module to TorchScript format.
     */
    public static void saveAsTorchScript(Module module, File file) {
        JitModule jit = module_to_jit(module);
        jit.save(file.getAbsolutePath());
    }

    /**
     * Save a Module to TorchScript with extras.
     */
    public static void saveAsTorchScript(Module module, File file, ExtraFilesMap extras) {
        JitModule jit = module_to_jit(module);
        jit.save(file.getAbsolutePath(), extras);
    }

    private static native JitModule module_to_jit(Module module);

    // ---- Compatibility ----

    /**
     * Convert a WeightBagModule to a standard nn.Module with registered parameters.
     * This allows using the model with standard PyTorch Java APIs.
     */
    public static Module toModule(WeightBagModule bag) {
        return bag;
    }

    /**
     * Get module type description.
     */
    public static String getModuleType(Object model) {
        if (model instanceof JitModule) return "JitModule (TorchScript)";
        if (model instanceof WeightBagModule) return "WeightBagModule (nn.Module)";
        if (model instanceof Module) return "nn.Module";
        if (model instanceof Map) return "StateDict";
        return "Unknown";
    }
}
