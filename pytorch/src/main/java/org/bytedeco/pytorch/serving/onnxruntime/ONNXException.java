package org.bytedeco.pytorch.serving.onnxruntime;

/**
 * Exception class for ONNX operations.
 */
public class ONNXException extends RuntimeException {

    public ONNXException(String message) {
        super(message);
    }

    public ONNXException(String message, Throwable cause) {
        super(message, cause);
    }

    public ONNXException(Throwable cause) {
        super(cause);
    }
}
