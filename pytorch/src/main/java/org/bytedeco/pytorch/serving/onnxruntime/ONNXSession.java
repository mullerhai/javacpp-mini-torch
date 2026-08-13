package org.bytedeco.pytorch.serving.onnxruntime;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.global.torch.ScalarType;
import ai.onnxruntime.*;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtSession.SessionOptions;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * ONNX Runtime session wrapper for JavaCPP PyTorch.
 *
 * <h2>Supported Operations</h2>
 *
 * <ul>
 *   <li>Model loading from file/path</li>
 *   <li>Session creation with configurable providers</li>
 *   <li>Inference with OrtValue / Tensor conversion</li>
 *   <li>Tensor conversion between ONNX and PyTorch formats</li>
 *   <li>Input/output metadata extraction</li>
 * </ul>
 *
 * <h2>Usage</h2>
 *
 * <pre>{@code
 * // Load model
 * ONNXSession session = ONNXSession.load("model.onnx");
 *
 * // Get metadata
 * ONNXModelInfo info = session.getModelInfo();
 * System.out.println("Inputs: " + info.getInputNames());
 * System.out.println("Outputs: " + info.getOutputNames());
 *
 * // Run inference
 * Map<String, Tensor> inputs = Map.of("input", torch.randn(1, 10));
 * Map<String, Tensor> outputs = session.run(inputs);
 *
 * // Convert to PyTorch Module wrapper
 * Module module = session.toModule();
 *
 * session.close();
 * }</pre>
 *
 * @see <a href="https://onnxruntime.ai/docs/">ONNX Runtime Documentation</a>
 */
public class ONNXSession implements AutoCloseable {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final ONNXModelInfo modelInfo;
    private final ONNXOptions options;
    private boolean closed;

    private ONNXSession(OrtEnvironment env, OrtSession session, ONNXModelInfo modelInfo, ONNXOptions options) {
        this.env = env;
        this.session = session;
        this.modelInfo = modelInfo;
        this.options = options;
        this.closed = false;
    }

    /**
     * Load an ONNX model from file.
     */
    public static ONNXSession load(String path) throws ONNXException {
        return load(Path.of(path), new ONNXOptions());
    }

    public static ONNXSession load(Path path) throws ONNXException {
        return load(path, new ONNXOptions());
    }

    public static ONNXSession load(String path, ONNXOptions options) throws ONNXException {
        return load(Path.of(path), options);
    }

    public static ONNXSession load(Path path, ONNXOptions options) throws ONNXException {
        Objects.requireNonNull(path, "path");
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("Model file not found: " + path);
        }

        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();

            OrtSession.SessionOptions sessionOpts = new OrtSession.SessionOptions();
            configureSessionOptions(sessionOpts, options);

            OrtSession sess = env.createSession(path.toString(), sessionOpts);
            ONNXModelInfo info = extractModelInfo(env, sess);

            return new ONNXSession(env, sess, info, options);
        } catch (OrtException e) {
            throw new ONNXException("Failed to load ONNX model: " + e.getMessage(), e);
        }
    }

    /**
     * Load an ONNX model from bytes.
     */
    public static ONNXSession load(byte[] modelBytes) throws ONNXException {
        return load(modelBytes, new ONNXOptions());
    }

    public static ONNXSession load(byte[] modelBytes, ONNXOptions options) throws ONNXException {
        Objects.requireNonNull(modelBytes, "modelBytes");

        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();

            SessionOptions sessionOpts = new SessionOptions();
            configureSessionOptions(sessionOpts, options);

            OrtSession sess = env.createSessionFromArray(modelBytes, sessionOpts);
            ONNXModelInfo info = extractModelInfo(env, sess);

            return new ONNXSession(env, sess, info, options);
        } catch (OrtException e) {
            throw new ONNXException("Failed to load ONNX model from bytes: " + e.getMessage(), e);
        }
    }

    private static void configureSessionOptions(SessionOptions opts, ONNXOptions options) throws OrtException {
        if (options == null) return;

        // Configure providers
        List<String> providers = options.getProviders();
        if (providers.isEmpty()) {
            // Default providers order
            providers = List.of(
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            );
        }

        for (String provider : providers) {
            try {
                switch (provider.toUpperCase()) {
                    case "CUDA":
                    case "CUDAEXECUTIONPROVIDER":
                        opts.registerCUDA();
                        opts.addConfigEntry("device_id", String.valueOf(options.getDeviceId()));
                        if (options.getGpuMemLimit() > 0) {
                            opts.addConfigEntry("gpu_mem_limit", String.valueOf(options.getGpuMemLimit()));
                        }
                        break;
                    case "CPU":
                    case "CPUEXECUTIONPROVIDER":
                        opts.registerCPU();
                        break;
                    case "COREML":
                    case "COREMLEXECUTIONPROVIDER":
                        opts.registerCoreML();
                        break;
                    case "NNAPI":
                    case "NNAPIEXECUTIONPROVIDER":
                        opts.registerNnapi();
                        break;
                    case "TENSORRT":
                    case "TENSORRTEXECUTIONPROVIDER":
                        opts.registerTensorrt();
                        break;
                    default:
                        // Skip unknown providers
                        break;
                }
            } catch (OrtException ignored) {
                // Provider not available
            }
        }

        // Configure threading
        if (options.getInterOpNumThreads() > 0) {
            opts.setInterOpNumThreads(options.getInterOpNumThreads());
        }
        if (options.getIntraOpNumThreads() > 0) {
            opts.setIntraOpNumThreads(options.getIntraOpNumThreads());
        }

        // Graph optimization level
        opts.setGraphOptimizationLevel(options.getGraphOptimizationLevel().toORTLevel());

        // Execution mode
        if (options.isParallelExecution()) {
            opts.setExecutionMode(SessionOptions.ExecutionMode.ORT_PARALLEL);
        } else {
            opts.setExecutionMode(SessionOptions.ExecutionMode.ORT_SEQUENTIAL);
        }
    }

    private static ONNXModelInfo extractModelInfo(OrtEnvironment env, OrtSession session) throws ONNXException {
        ONNXModelInfo.Builder builder = new ONNXModelInfo.Builder();

        try {
            // Get input metadata
            try (var inputs = session.getInputs()) {
                List<String> inputNames = new ArrayList<>();
                for (var input : inputs) {
                    inputNames.add(input.getName());
                    builder.addInput(new ONNXTensorInfo(
                        input.getName(),
                        input.getType(),
                        input.getShape(),
                        input.getElementType()
                    ));
                }
                builder.inputNames(inputNames);
            }

            // Get output metadata
            try (var outputs = session.getOutputs()) {
                List<String> outputNames = new ArrayList<>();
                for (var output : outputs) {
                    outputNames.add(output.getName());
                    builder.addOutput(new ONNXTensorInfo(
                        output.getName(),
                        output.getType(),
                        output.getShape(),
                        output.getElementType()
                    ));
                }
                builder.outputNames(outputNames);
            }

            // Get model metadata
            try (var meta = session.getModelMetadata()) {
                builder.producerName(meta.getProducerName());
                builder.graphName(meta.getGraphName());
                builder.domain(meta.getDomain());
                builder.description(meta.getDescription());
                builder.version(String.valueOf(meta.getModelVersion()));
                try {
                    builder.irVersion(meta.getIrVersion());
                } catch (Exception ignored) {}
            }
        } catch (OrtException e) {
            throw new ONNXException("Failed to extract model info: " + e.getMessage(), e);
        }

        return builder.build();
    }

    /**
     * Get model metadata.
     */
    public ONNXModelInfo getModelInfo() {
        return modelInfo;
    }

    /**
     * Run inference with OrtValue inputs.
     */
    public OrtValue[] run(OrtValue[] inputs) throws ONNXException {
        checkNotClosed();
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("inputs cannot be null or empty");
        }

        try {
            String[] inputNames = new String[inputs.length];
            for (int i = 0; i < inputs.length; i++) {
                inputNames[i] = modelInfo.getInputNames().get(i);
            }
            try (var outputHandles = session.run(inputNames, inputs)) {
                OrtValue[] outputs = new OrtValue[(int) outputHandles.size()];
                for (int i = 0; i < outputHandles.size(); i++) {
                    outputs[i] = outputHandles.get(i);
                }
                return outputs;
            }
        } catch (OrtException e) {
            throw new ONNXException("ONNX inference failed: " + e.getMessage(), e);
        }
    }

    /**
     * Run inference with named inputs.
     */
    public Map<String, Tensor> run(Map<String, Tensor> inputs) throws ONNXException {
        checkNotClosed();
        if (inputs == null || inputs.isEmpty()) {
            throw new IllegalArgumentException("inputs cannot be null or empty");
        }

        // Convert Tensor to OrtValue
        List<String> inputNames = new ArrayList<>(inputs.keySet());
        OrtValue[] inputValues = new OrtValue[inputNames.size()];

        int idx = 0;
        for (String name : inputNames) {
            inputValues[idx++] = tensorToOrtValue(inputs.get(name));
        }

        // Run inference
        try (var outputHandles = session.run(inputNames.toArray(new String[0]), inputValues)) {
            Map<String, Tensor> outputs = new LinkedHashMap<>();
            for (int i = 0; i < outputHandles.size(); i++) {
                String outputName = modelInfo.getOutputNames().get(i);
                outputs.put(outputName, ortValueToTensor(outputHandles.get(i)));
            }
            return outputs;
        } catch (OrtException e) {
            throw new ONNXException("ONNX inference failed: " + e.getMessage(), e);
        }
    }

    /**
     * Run inference with single tensor input.
     */
    public Tensor run(Tensor input) throws ONNXException {
        String inputName = modelInfo.getInputNames().get(0);
        return run(Map.of(inputName, input)).values().iterator().next();
    }

    /**
     * Run inference with single tensor and return specified output.
     */
    public Tensor run(Tensor input, String outputName) throws ONNXException {
        String inputName = modelInfo.getInputNames().get(0);
        return run(Map.of(inputName, input)).get(outputName);
    }

    /**
     * Convert JavaCPP Tensor to ONNX OrtValue.
     */
    public static OrtValue tensorToOrtValue(Tensor tensor) throws OrtException {
        if (tensor == null || tensor.isNull()) {
            throw new IllegalArgumentException("tensor cannot be null");
        }

        long[] shape = new long[(int) tensor.dim()];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = tensor.size(i);
        }

        OnnxJavaType onnxType = toOnnxType(tensor.dtype());

        // Get contiguous tensor
        try (Tensor cont = tensor.is_contiguous() ? tensor : tensor.contiguous()) {
            // Create ONNX tensor based on data type
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            return createOrtValue(env, cont, shape, onnxType);
        }
    }

    private static OrtValue createOrtValue(OrtEnvironment env, Tensor tensor, long[] shape, OnnxJavaType onnxType) throws OrtException {
        long numel = 1;
        for (long s : shape) {
            numel *= s;
        }

        switch (onnxType) {
            case FLOAT: {
                float[] data = new float[(int) numel];
                if (tensor.dtype() == ScalarType.kFloat) {
                    // Direct copy from float tensor
                    org.bytedeco.pytorch.FloatTensor floatTensor = new org.bytedeco.pytorch.FloatTensor(tensor);
                    floatTensor.copyTo(data);
                } else {
                    // Convert from other types
                    copyAsFloat(tensor, data);
                }
                return OnnxTensor.createTensor(env, data, shape);
            }
            case DOUBLE: {
                double[] data = new double[(int) numel];
                copyAsDouble(tensor, data);
                return OnnxTensor.createTensor(env, data, shape);
            }
            case INT64: {
                long[] data = new long[(int) numel];
                if (tensor.dtype() =ScalarType.kLong) {
                    Tensor longTensor = new LongTensor(tensor);
                    longTensor.copyTo(data);
                } else {
                    copyAsLong(tensor, data);
                }
                return OnnxTensor.createTensor(env, data, shape);
            }
            case INT32: {
                int[] data = new int[(int) numel];
                copyAsInt(tensor, data);
                return OnnxTensor.createTensor(env, data, shape);
            }
            case BOOL: {
                byte[] data = new byte[(int) numel];
                copyAsByte(tensor, data);
                return OnnxTensor.createTensor(env, data, shape);
            }
            case STRING: {
                String[] data = new String[(int) numel];
                return OnnxTensor.createTensor(env, data, shape);
            }
            default:
                throw new OrtException("Unsupported ONNX type: " + onnxType);
        }
    }

    private static void copyAsFloat(Tensor tensor, float[] data) {
        long numel = tensor.numel();
        for (int i = 0; i < numel; i++) {
            data[i] = tensor.item_float();
        }
    }

    private static void copyAsDouble(Tensor tensor, double[] data) {
        long numel = tensor.numel();
        for (int i = 0; i < numel; i++) {
            data[i] = tensor.item_double();
        }
    }

    private static void copyAsLong(Tensor tensor, long[] data) {
        long numel = tensor.numel();
        for (int i = 0; i < numel; i++) {
            data[i] = tensor.item_long();
        }
    }

    private static void copyAsInt(Tensor tensor, int[] data) {
        long numel = tensor.numel();
        for (int i = 0; i < numel; i++) {
            data[i] = (int) tensor.item_long();
        }
    }

    private static void copyAsByte(Tensor tensor, byte[] data) {
        long numel = tensor.numel();
        for (int i = 0; i < numel; i++) {
            data[i] = (byte) tensor.item_long();
        }
    }

    /**
     * Convert ONNX OrtValue to JavaCPP Tensor.
     */
    public static Tensor ortValueToTensor(OrtValue ortValue) throws OrtException {
        if (ortValue == null) {
            throw new IllegalArgumentException("ortValue cannot be null");
        }

        if (!ortValue.isTensor()) {
            throw new IllegalArgumentException("Only tensor OrtValue is supported");
        }

        try (OnnxTensor onnxTensor = ortValue.getValue()) {
            long[] shape = onnxTensor.getShape();
            OnnxJavaType type = onnxTensor.getType();
            return createTensorFromOnnxTensor(onnxTensor, shape, type);
        }
    }

    private static Tensor createTensorFromOnnxTensor(OnnxTensor onnxTensor, long[] shape, OnnxJavaType type) throws OrtException {
        torch torch = new org.bytedeco.pytorch.global.torch();
        Tensor tensor;

        switch (type) {
            case FLOAT: {
                float[] data = onnxTensor.getFloatBuffer();
                tensor = torch.from_blob(data, shape, torch.kCPU()).clone();
                break;
            }
            case DOUBLE: {
                double[] data = onnxTensor.getDoubleBuffer();
                // Convert to float
                float[] floatData = new float[data.length];
                for (int i = 0; i < data.length; i++) {
                    floatData[i] = (float) data[i];
                }
                tensor = torch.from_blob(floatData, shape, torch.kCPU()).clone();
                break;
            }
            case INT64: {
                long[] data = onnxTensor.getLongBuffer();
                tensor = torch.from_blob(data, shape, torch.kCPU()).clone();
                break;
            }
            case INT32: {
                int[] data = onnxTensor.getIntBuffer();
                // Convert to long
                long[] longData = new long[data.length];
                for (int i = 0; i < data.length; i++) {
                    longData[i] = data[i];
                }
                tensor = torch.from_blob(longData, shape, torch.kCPU()).clone();
                break;
            }
            case BOOL: {
                byte[] data = onnxTensor.getBooleanBuffer();
                tensor = torch.from_blob(data, shape, torch.kCPU()).clone();
                break;
            }
            default:
                throw new OrtException("Unsupported ONNX type: " + type);
        }

        return tensor;
    }

    private static OnnxJavaType toOnnxType(org.bytedeco.pytorch.ScalarType dtype) {
        switch (dtype) {
            case kFloat:
            case kHalf:
                return OnnxJavaType.FLOAT;
            case kDouble:
                return OnnxJavaType.DOUBLE;
            case kInt:
            case kShort:
            case kByte:
            case kChar:
                return OnnxJavaType.INT32;
            case kLong:
                return OnnxJavaType.INT64;
            case kBool:
                return OnnxJavaType.BOOL;
            default:
                return OnnxJavaType.FLOAT;
        }
    }

    /**
     * Convert ONNX session to PyTorch nn.Module wrapper.
     *
     * <p>This creates a wrapper module that delegates to the ONNX session.
     * The wrapper handles input/output mapping and tensor conversions.
     */
    public org.bytedeco.pytorch.nn.Module toModule() {
        checkNotClosed();
        return new ONNXModuleWrapper(this);
    }

    /**
     * Get input names.
     */
    public List<String> getInputNames() {
        return modelInfo.getInputNames();
    }

    /**
     * Get output names.
     */
    public List<String> getOutputNames() {
        return modelInfo.getOutputNames();
    }

    /**
     * Check if session is closed.
     */
    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("ONNXSession has been closed");
        }
    }

    /**
     * Get the underlying OrtSession.
     */
    public OrtSession getOrtSession() {
        return session;
    }

    /**
     * Get the OrtEnvironment.
     */
    public OrtEnvironment getOrtEnvironment() {
        return env;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            try {
                session.close();
                env.close();
            } catch (Exception e) {
                System.err.println("Error closing ONNX session: " + e.getMessage());
            }
        }
    }
}
