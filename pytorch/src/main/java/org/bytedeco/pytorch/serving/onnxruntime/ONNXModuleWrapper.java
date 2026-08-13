package org.bytedeco.pytorch.serving.onnxruntime;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.jit.IValue;

import java.util.Map;

/**
 * Wrapper module that delegates to ONNX Runtime session.
 *
 * <p>This class provides a PyTorch-like interface to ONNX models, enabling:
 * <ul>
 *   <li>Use of ONNX models through PyTorch APIs</li>
 *   <li>Easy conversion from ONNX to PyTorch workflows</li>
 *   <li>Model composition with other PyTorch modules</li>
 * </ul>
 *
 * <h2>Usage</h2>
 *
 * <pre>{@code
 * // Load ONNX model
 * ONNXSession session = ONNXSession.load("model.onnx");
 *
 * // Wrap as PyTorch module
 * Module module = session.toModule();
 *
 * // Use like PyTorch module
 * Tensor input = torch.randn(1, 10);
 * Tensor output = module.forward(input);
 *
 * // Multi-input via Tensor map
 * Map<String, Tensor> inputs = Map.of("input1", t1, "input2", t2);
 * Map<String, Tensor> outputs = module.forward(inputs);
 *
 * session.close();
 * }</pre>
 */
public class ONNXModuleWrapper extends org.bytedeco.pytorch.nn.Module {

    private final ONNXSession session;

    public ONNXModuleWrapper(ONNXSession session) {
        super();
        this.session = session;
    }

    /**
     * Forward pass with single Tensor input.
     *
     * @param input single input tensor (uses first ONNX input name)
     * @return output tensor
     */
    public Tensor forward(Tensor input) {
        try {
            return session.run(input);
        } catch (ONNXException e) {
            throw new RuntimeException("ONNX inference failed: " + e.getMessage(), e);
        }
    }

    /**
     * Forward pass with named Tensor inputs.
     *
     * @param inputs map of input name to tensor
     * @return map of output name to tensor
     */
    public Map<String, Tensor> forward(Map<String, Tensor> inputs) {
        try {
            return session.run(inputs);
        } catch (ONNXException e) {
            throw new RuntimeException("ONNX inference failed: " + e.getMessage(), e);
        }
    }

    /**
     * Forward pass with multiple Tensor inputs (positional).
     *
     * <p>Each tensor is mapped to the corresponding ONNX input name in order.
     */
    public Map<String, Tensor> forward(Tensor... inputs) {
        try {
            Map<String, Tensor> inputMap = new java.util.LinkedHashMap<>();
            var inputNames = session.getInputNames();
            for (int i = 0; i < inputs.length && i < inputNames.size(); i++) {
                inputMap.put(inputNames.get(i), inputs[i]);
            }
            return session.run(inputMap);
        } catch (ONNXException e) {
            throw new RuntimeException("ONNX inference failed: " + e.getMessage(), e);
        }
    }

    /**
     * Forward pass wrapping Tensor in IValue (for PyTorch JIT compatibility).
     */
    public IValue forwardAsIValue(Tensor input) {
        try {
            Tensor output = session.run(input);
            return new IValue(output);
        } catch (ONNXException e) {
            throw new RuntimeException("ONNX inference failed: " + e.getMessage(), e);
        }
    }

    /**
     * Get the underlying ONNX session.
     */
    public ONNXSession getSession() {
        return session;
    }

    /**
     * Get model info.
     */
    public ONNXModelInfo getModelInfo() {
        return session.getModelInfo();
    }

    /**
     * Get input names.
     */
    public java.util.List<String> getInputNames() {
        return session.getInputNames();
    }

    /**
     * Get output names.
     */
    public java.util.List<String> getOutputNames() {
        return session.getOutputNames();
    }

    @Override
    public String toString() {
        return "ONNXModuleWrapper{" +
                "inputs=" + session.getInputNames() +
                ", outputs=" + session.getOutputNames() +
                '}';
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }

    @Override
    public void close() {
        super.close();
        if (session != null) {
            session.close();
        }
    }
}