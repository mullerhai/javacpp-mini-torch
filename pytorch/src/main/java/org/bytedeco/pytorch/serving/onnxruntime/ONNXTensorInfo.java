package org.bytedeco.pytorch.serving.onnxruntime;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.TensorInfo;
import java.util.Objects;

/**
 * Information about a single tensor in the ONNX model.
 */
public final class ONNXTensorInfo {

    private final String name;
    private final String type;
    private final long[] shape;
    private final OnnxJavaType elementType;

    public ONNXTensorInfo(String name, String type, long[] shape, OnnxJavaType elementType) {
        this.name = Objects.requireNonNull(name, "name");
        this.type = type;
        this.shape = shape != null ? shape : new long[0];
        this.elementType = elementType;
    }

    /**
     * Construct from ONNX Runtime {@link TensorInfo} (onnxruntime 1.28 API).
     */
    public ONNXTensorInfo(String name, TensorInfo info, long[] shape) {
        this.name = Objects.requireNonNull(name, "name");
        this.type = info != null ? info.toString() : "";
        this.shape = shape != null ? shape : new long[0];
        this.elementType = info != null ? info.type : OnnxJavaType.FLOAT;
    }

    /**
     * Get tensor name.
     */
    public String getName() {
        return name;
    }

    /**
     * Get tensor type.
     */
    public String getType() {
        return type;
    }

    /**
     * Get tensor shape.
     */
    public long[] getShape() {
        return shape;
    }

    /**
     * Get element type.
     */
    public OnnxJavaType getElementType() {
        return elementType;
    }

    /**
     * Get element type as string.
     */
    public String getElementTypeString() {
        if (elementType == null) return "unknown";
        switch (elementType) {
            case FLOAT: return "float32";
            case DOUBLE: return "float64";
            case INT64: return "int64";
            case INT32: return "int32";
            case INT16: return "int16";
            case UINT8: return "uint8";
            case INT8: return "int8";
            case BOOL: return "bool";
            case STRING: return "string";
            default: return "unknown";
        }
    }

    /**
     * Get total number of elements.
     */
    public long getNumElements() {
        long count = 1;
        for (long dim : shape) {
            if (dim > 0) {
                count *= dim;
            }
        }
        return count;
    }

    /**
     * Check if shape is dynamic (contains unknown dimensions).
     */
    public boolean hasDynamicShape() {
        for (long dim : shape) {
            if (dim < 0) return true;
        }
        return false;
    }

    /**
     * Get shape as string.
     */
    public String getShapeString() {
        if (shape == null || shape.length == 0) {
            return "[]";
        }
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            if (i > 0) sb.append(", ");
            if (shape[i] < 0) {
                sb.append("?").append(-shape[i]);
            } else {
                sb.append(shape[i]);
            }
        }
        sb.append("]");
        return sb.toString();
    }

    @Override
    public String toString() {
        return name + ": " + getElementTypeString() + " " + getShapeString();
    }
}
