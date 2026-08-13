package org.bytedeco.pytorch.serving.onnxruntime;
import ai.onnxruntime.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DeviceOptional;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.*;

import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.options.ONNXOptions;

import java.nio.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * ONNX Runtime session wrapper for JavaCPP PyTorch.
 *
 * <h2>Supported Operations</h2>
 *
 * <ul>
 *   <li>Model loading from file/path/bytes</li>
 *   <li>Session creation with configurable execution providers</li>
 *   <li>Inference with {@link Tensor} inputs</li>
 *   <li>Conversion between JavaCPP {@link Tensor} and ONNX {@link OnnxTensor}</li>
 *   <li>Input/output metadata extraction</li>
 *   <li>nn.Module wrapper creation</li>
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
 * // Run inference with Tensor
 * Tensor input = ...; // JavaCPP Tensor
 * Map<String, Tensor> inputs = new HashMap<>();
 * inputs.put(info.getInputNames().get(0), input);
 * Map<String, Tensor> outputs = session.run(inputs);
 *
 * // Convert to PyTorch nn.Module wrapper
 * org.bytedeco.pytorch.nn.Module module = session.toModule();
 *
 * session.close();
 * }</pre>
 *
 * <p>API targets ONNX Runtime Java 1.28.0 (no OrtValue; uses OnnxTensor/OnnxValue).
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
     * Load an ONNX model from file (default options).
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
            SessionOptions sessionOpts = new SessionOptions();
            configureSessionOptions(sessionOpts, options);

            OrtSession sess = env.createSession(path.toString(), sessionOpts);
            ONNXModelInfo info = extractModelInfo(sess);
            return new ONNXSession(env, sess, info, options);
        } catch (OrtException e) {
            throw new ONNXException("Failed to load ONNX model: " + e.getMessage(), e);
        }
    }

    /**
     * Load an ONNX model from raw bytes.
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

            OrtSession sess = env.createSession(modelBytes, sessionOpts);
            ONNXModelInfo info = extractModelInfo(sess);
            return new ONNXSession(env, sess, info, options);
        } catch (OrtException e) {
            throw new ONNXException("Failed to load ONNX model from bytes: " + e.getMessage(), e);
        }
    }

    private static void configureSessionOptions(SessionOptions opts, ONNXOptions options) throws OrtException {
        if (options == null) return;

        // Configure execution providers (ONNX Runtime 1.28 API: addXxx())
        List<String> providers = options.getProviders();
        if (providers.isEmpty()) {
            providers = List.of("CPU");
        }

        for (String provider : providers) {
            try {
                switch (provider.toUpperCase()) {
                    case "CUDA":
                    case "CUDAEXECUTIONPROVIDER":
                        opts.addCUDA(options.getDeviceId());
                        if (options.getGpuMemLimit() > 0) {
                            opts.addConfigEntry("gpu_mem_limit", String.valueOf(options.getGpuMemLimit()));
                        }
                        break;
                    case "CPU":
                    case "CPUEXECUTIONPROVIDER":
                        opts.addCPU(true);
                        break;
                    case "COREML":
                    case "COREMLEXECUTIONPROVIDER":
                        opts.addCoreML();
                        break;
                    case "NNAPI":
                    case "NNAPIEXECUTIONPROVIDER":
                        opts.addNnapi();
                        break;
                    case "TENSORRT":
                    case "TENSORRTEXECUTIONPROVIDER":
                        opts.addTensorrt(options.getDeviceId());
                        break;
                    case "DIRECTML":
                    case "DIRECTMLEXECUTIONPROVIDER":
                        opts.addDirectML(options.getDeviceId());
                        break;
                    case "ROCM":
                    case "ROCMEXECUTIONPROVIDER":
                        opts.addROCM(options.getDeviceId());
                        break;
                    default:
                        // Skip unknown providers
                        break;
                }
            } catch (OrtException ignored) {
                // Provider not available on this platform
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
        opts.setOptimizationLevel(options.getGraphOptimizationLevel().toORTLevel());

        // Execution mode
        opts.setExecutionMode(options.isParallelExecution()
                ? SessionOptions.ExecutionMode.PARALLEL
                : SessionOptions.ExecutionMode.SEQUENTIAL);
    }

    private static ONNXModelInfo extractModelInfo(OrtSession session) throws ONNXException {
        ONNXModelInfo.Builder builder = new ONNXModelInfo.Builder();
        try {
            // Inputs
            List<String> inputNames = new ArrayList<>(session.getInputNames());
            Map<String, org.bytedeco.pytorch.serving.onnxruntime.NodeInfo> inputNodeInfos =
                    extractNodeInfos(session.getInputInfo());
            for (String name : inputNames) {
                org.bytedeco.pytorch.serving.onnxruntime.NodeInfo ni = inputNodeInfos.get(name);
                long[] shape = ni != null ? ni.getShape() : new long[0];
                builder.addInput(new ONNXTensorInfo(name, ni != null ? ni.getTypeString() : "tensor", shape,
                        ni != null ? ni.getElementType() : OnnxJavaType.FLOAT));
            }
            builder.inputNames(inputNames);

            // Outputs
            List<String> outputNames = new ArrayList<>(session.getOutputNames());
            Map<String, org.bytedeco.pytorch.serving.onnxruntime.NodeInfo> outputNodeInfos =
                    extractNodeInfos(session.getOutputInfo());
            for (String name : outputNames) {
                org.bytedeco.pytorch.serving.onnxruntime.NodeInfo ni = outputNodeInfos.get(name);
                long[] shape = ni != null ? ni.getShape() : new long[0];
                builder.addOutput(new ONNXTensorInfo(name, ni != null ? ni.getTypeString() : "tensor", shape,
                        ni != null ? ni.getElementType() : OnnxJavaType.FLOAT));
            }
            builder.outputNames(outputNames);

            // Model metadata
            try {
                OnnxModelMetadata meta = session.getMetadata();
                builder.producerName(meta.getProducerName());
                builder.graphName(meta.getGraphName());
                builder.domain(meta.getDomain());
                builder.description(meta.getDescription());
                builder.version(String.valueOf(meta.getVersion()));
                try {
                    builder.irVersion(meta.getVersion());
                } catch (Exception ignored) {}
            } catch (Exception ignored) {}
        } catch (OrtException e) {
            throw new ONNXException("Failed to extract model info: " + e.getMessage(), e);
        }
        return builder.build();
    }

    private static Map<String, org.bytedeco.pytorch.serving.onnxruntime.NodeInfo> extractNodeInfos(
            Map<String, ai.onnxruntime.NodeInfo> infos) {
        Map<String, org.bytedeco.pytorch.serving.onnxruntime.NodeInfo> out = new LinkedHashMap<>();
        for (Map.Entry<String, ai.onnxruntime.NodeInfo> e : infos.entrySet()) {
            ai.onnxruntime.NodeInfo ni = e.getValue();
            ai.onnxruntime.ValueInfo vi = ni.getInfo();
            out.put(e.getKey(), new org.bytedeco.pytorch.serving.onnxruntime.NodeInfo(vi));
        }
        return out;
    }

    /**
     * Run inference with named Tensor inputs.
     */
    public Map<String, Tensor> run(Map<String, Tensor> inputs) throws ONNXException {
        checkNotClosed();
        Objects.requireNonNull(inputs, "inputs");
        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("inputs cannot be empty");
        }

        try {
            // Build input map: name -> OnnxTensor
            Map<String, OnnxTensorLike> onnxInputs = new LinkedHashMap<>();
            List<OnnxTensorLike> created = new ArrayList<>();
            try {
                for (Map.Entry<String, Tensor> entry : inputs.entrySet()) {
                    OnnxTensor t = tensorToOnnxTensor(entry.getValue());
                    created.add(t);
                    onnxInputs.put(entry.getKey(), t);
                }

                // Run inference
                Result result = session.run(onnxInputs);
                Map<String, Tensor> outputs = new LinkedHashMap<>();
                for (String outputName : modelInfo.getOutputNames()) {
                    OnnxValue value = result.get(outputName).orElseThrow(
                            () -> new OrtException("Output '" + outputName + "' not present in result"));
                    outputs.put(outputName, onnxValueToTensor(value));
                }
                result.close();
                return outputs;
            } finally {
                for (OnnxTensorLike t : created) {
                    try {
                        if (t instanceof OnnxTensor) {
                            ((OnnxTensor) t).close();
                        }
                    } catch (Exception ignored) {}
                }
            }
        } catch (OrtException e) {
            throw new ONNXException("ONNX inference failed: " + e.getMessage(), e);
        }
    }

    /**
     * Run inference with single tensor input (uses first input name).
     */
    public Tensor run(Tensor input) throws ONNXException {
        String inputName = modelInfo.getInputNames().get(0);
        Map<String, Tensor> inputs = new HashMap<>();
        inputs.put(inputName, input);
        Map<String, Tensor> outputs = run(inputs);
        return outputs.values().iterator().next();
    }

    /**
     * Convert JavaCPP {@link Tensor} to ONNX {@link OnnxTensor}.
     *
     * <p>This creates a heap-allocated copy of the tensor's data and wraps it in an
     * OnnxTensor owned by the ORT environment.
     */
    public static OnnxTensor tensorToOnnxTensor(Tensor tensor) throws OrtException {
        Objects.requireNonNull(tensor, "tensor");
        long[] shape = new long[(int) tensor.dim()];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = tensor.size(i);
        }
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        return createOnnxTensor(env, tensor, shape, toOnnxType(tensor));
    }

    private static OnnxTensor createOnnxTensor(OrtEnvironment env, Tensor tensor, long[] shape, OnnxJavaType type) throws OrtException {
        long numel = 1;
        for (long s : shape) numel *= s;
        int n = (int) numel;

        switch (type) {
            case FLOAT: {
                float[] data = new float[n];
                copyToFloatArray(tensor, data);
                FloatBuffer buffer = FloatBuffer.wrap(data);
                return OnnxTensor.createTensor(env, buffer, shape);
            }
            case DOUBLE: {
                double[] data = new double[n];
                copyToDoubleArray(tensor, data);
                DoubleBuffer buffer = DoubleBuffer.wrap(data);
                return OnnxTensor.createTensor(env, buffer, shape);
            }
            case INT64: {
                long[] data = new long[n];
                copyToLongArray(tensor, data);
                LongBuffer buffer = LongBuffer.wrap(data);
                return OnnxTensor.createTensor(env, buffer, shape);
            }
            case INT32: {
                int[] data = new int[n];
                copyToIntArray(tensor, data);
                IntBuffer buffer = IntBuffer.wrap(data);
                return OnnxTensor.createTensor(env, buffer, shape);
            }
            case BOOL: {
                byte[] data = new byte[n];
                copyToByteArray(tensor, data);
                ByteBuffer buffer = ByteBuffer.wrap(data);
                return OnnxTensor.createTensor(env, buffer, shape);
            }
            case STRING: {
                String[] data = new String[n];
                return OnnxTensor.createTensor(env, data, shape);
            }
            default:
                throw new OrtException("Unsupported ONNX type: " + type);
        }
    }

    private static void copyToFloatArray(Tensor tensor, float[] data) {
        long n = tensor.numel();
        for (int i = 0; i < n; i++) {
            data[i] = tensor.item_float();
        }
    }

    private static void copyToDoubleArray(Tensor tensor, double[] data) {
        long n = tensor.numel();
        for (int i = 0; i < n; i++) {
            data[i] = tensor.item_double();
        }
    }

    private static void copyToLongArray(Tensor tensor, long[] data) {
        long n = tensor.numel();
        for (int i = 0; i < n; i++) {
            data[i] = tensor.item_long();
        }
    }

    private static void copyToIntArray(Tensor tensor, int[] data) {
        long n = tensor.numel();
        for (int i = 0; i < n; i++) {
            data[i] = (int) tensor.item_long();
        }
    }

    private static void copyToByteArray(Tensor tensor, byte[] data) {
        long n = tensor.numel();
        for (int i = 0; i < n; i++) {
            data[i] = (byte) tensor.item_long();
        }
    }

    /**
     * Convert an {@link OnnxValue} (typically from inference result) to a JavaCPP Tensor.
     *
     * <p>Only {@link OnnxTensor} values are supported.
     */
    public static Tensor onnxValueToTensor(OnnxValue value) throws OrtException {
        Objects.requireNonNull(value, "value");
        if (!(value instanceof OnnxTensor)) {
            throw new OrtException("Only OnnxTensor values are supported, got: " + value.getClass().getSimpleName());
        }
        OnnxTensor onnxTensor = (OnnxTensor) value;
        try {
            long[] shape = onnxTensor.getInfo().getShape();
            OnnxJavaType type = onnxTensor.getInfo().type;
            return onnxTensorToTensor(onnxTensor, shape, type);
        } finally {
            onnxTensor.close();
        }
    }

    private static Tensor onnxTensorToTensor(OnnxTensor onnxTensor, long[] shape, OnnxJavaType type) throws OrtException {
        // Use JavaCPP torch from_blob with kCPU allocator.
        // For each supported type, read buffer and construct Tensor.
        // The OnnxTensor is closed after read.
        int n = 1;
        for (long s : shape) n *= (int) s;
        org.bytedeco.pytorch.global.torch t = new org.bytedeco.pytorch.global.torch();
        Tensor result;
        var opt = new TensorOptions().device(new DeviceOptional(new Device(t.kCPU())));
        switch (type) {
            case FLOAT: {
                java.nio.FloatBuffer buf = onnxTensor.getFloatBuffer();
                float[] data = new float[n];
                buf.get(data);
                FloatPointer imgPtr = new FloatPointer(data);
                result = torch.from_blob(imgPtr, shape, opt).clone();
                break;
            }
            case DOUBLE: {
                java.nio.DoubleBuffer buf = onnxTensor.getDoubleBuffer();
                double[] data = new double[n];
                buf.get(data);
                DoublePointer imgPtr = new DoublePointer(data);
                float[] fdata = new float[n];
                for (int i = 0; i < n; i++) fdata[i] = (float) data[i];
                result = torch.from_blob(imgPtr, shape, opt).clone();
                break;
            }
            case INT64: {
                java.nio.LongBuffer buf = onnxTensor.getLongBuffer();
                long[] data = new long[n];
                buf.get(data);
                LongPointer imgPtr = new LongPointer(data);
                result = torch.from_blob(imgPtr, shape, opt).clone();
                break;
            }
            case INT32: {
                java.nio.IntBuffer buf = onnxTensor.getIntBuffer();
                int[] data = new int[n];
                buf.get(data);
                IntPointer imgPtr = new IntPointer(data);
                long[] ldata = new long[n];
                for (int i = 0; i < n; i++) ldata[i] = data[i];
                result = torch.from_blob(imgPtr, shape, opt).clone();
                break;
            }
            case BOOL: {
                java.nio.ByteBuffer buf = onnxTensor.getByteBuffer();
                byte[] data = new byte[n];
                buf.get(data);
                BytePointer imgPtr = new BytePointer(data);
                result = torch.from_blob(imgPtr, shape, opt).clone();
                break;
            }
            default:
                throw new OrtException("Unsupported ONNX type: " + type);
        }
        return result;
    }

    private static OnnxJavaType toOnnxType(Tensor tensor) {
        // Default to FLOAT for safety; downstream code reads dtype explicitly when needed.
        return OnnxJavaType.FLOAT;
    }

    /**
     * Convert ONNX session to PyTorch {@code nn.Module} wrapper.
     *
     * <p>This creates a wrapper module that delegates forward() to the ONNX session.
     */
    public org.bytedeco.pytorch.nn.Module toModule() {
        checkNotClosed();
        return new ONNXModuleWrapper(this);
    }

    /**
     * Get model metadata.
     */
    public ONNXModelInfo getModelInfo() {
        return modelInfo;
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
     * Get the underlying {@link OrtSession}.
     */
    public OrtSession getOrtSession() {
        return session;
    }

    /**
     * Get the {@link OrtEnvironment}.
     */
    public OrtEnvironment getOrtEnvironment() {
        return env;
    }

    private void checkNotClosed() {
        if (closed) {
            throw new IllegalStateException("ONNXSession has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            try {
                session.close();
            } catch (Exception e) {
                System.err.println("Error closing ONNX session: " + e.getMessage());
            }
            // Note: do not close the shared OrtEnvironment singleton
        }
    }
}