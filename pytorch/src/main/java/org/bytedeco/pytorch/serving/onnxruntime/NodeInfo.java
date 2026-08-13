package org.bytedeco.pytorch.serving.onnxruntime;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.ValueInfo;

/**
 * Internal helper to extract tensor metadata from ORT {@link ValueInfo}.
 *
 * <p>The ONNX Runtime 1.28 Java API exposes {@link ValueInfo} (interface with two impls:
 * {@link TensorInfo} and a map/sequence info). For model-IO purposes we only need the
 * tensor case, so this wrapper normalizes it into {@code (shape, OnnxJavaType)}.
 *
 * <p>This is a local type in the serving package to avoid colliding with
 * {@code ai.onnxruntime.NodeInfo}.
 */
public final class NodeInfo {

    private final String typeString;
    private final long[] shape;
    private final OnnxJavaType elementType;

    public NodeInfo(ValueInfo vi) {
        if (vi instanceof TensorInfo) {
            TensorInfo ti = (TensorInfo) vi;
            this.typeString = ti.toString();
            this.shape = ti.getShape();
            this.elementType = ti.type;
        } else {
            // Non-tensor IO (sequence/map). Shape unknown; type string captures it.
            this.typeString = vi != null ? vi.toString() : "";
            this.shape = new long[0];
            this.elementType = OnnxJavaType.UNKNOWN;
        }
    }

    public String getTypeString() {
        return typeString;
    }

    public long[] getShape() {
        return shape;
    }

    public OnnxJavaType getElementType() {
        return elementType;
    }
}