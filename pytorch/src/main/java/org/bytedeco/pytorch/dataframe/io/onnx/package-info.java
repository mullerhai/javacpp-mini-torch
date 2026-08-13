/**
 * ONNX model file reader for DataFrame integration.
 *
 * <p>This package provides schema inference for ONNX models, allowing them
 * to be queried like other DataFrame sources.
 *
 * <h2>Usage</h2>
 *
 * <pre>{@code
 * ONNXReader reader = new ONNXReader();
 * reader.load("model.onnx");
 * reader.printSchema();
 * reader.show();
 * reader.close();
 * }</pre>
 *
 * @see ONNXReader
 * @see org.bytedeco.pytorch.serving.onnxruntime.ONNXSession
 */
package org.bytedeco.pytorch.dataframe.io.onnx;