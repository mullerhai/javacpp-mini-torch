/**
 * ONNX Runtime integration for JavaCPP PyTorch.
 *
 * <h2>Overview</h2>
 *
 * This package provides seamless integration between ONNX Runtime and JavaCPP PyTorch,
 * enabling:
 *
 * <ul>
 *   <li>ONNX model loading and inference</li>
 *   <li>Zero-copy tensor conversion between ONNX and PyTorch</li>
 *   <li>Conversion of ONNX models to PyTorch nn.Module</li>
 *   <li>Model visualization in Vista</li>
 * </ul>
 *
 * <h2>Supported Formats</h2>
 *
 * <ul>
 *   <li>.onnx - Standard ONNX model files</li>
 * </ul>
 *
 * <h2>Quick Start</h2>
 *
 * <pre>{@code
 * // Load and run inference
 * try (ONNXSession session = ONNXSession.load("model.onnx")) {
 *     Map<String, Tensor> inputs = Map.of("input", torch.randn(1, 10));
 *     Map<String, Tensor> outputs = session.run(inputs);
 * }
 *
 * // Convert to PyTorch module
 * try (ONNXSession session = ONNXSession.load("model.onnx")) {
 *     Module module = session.toModule();
 *     // Use like any PyTorch module
 * }
 * }</pre>
 *
 * @see ONNXSession
 * @see ONNXOptions
 * @see ONNXModelInfo
 * @see <a href="https://onnxruntime.ai/docs/">ONNX Runtime Documentation</a>
 */
package org.bytedeco.pytorch.serving.onnxruntime;
import org.bytedeco.pytorch.nn.options.*;
