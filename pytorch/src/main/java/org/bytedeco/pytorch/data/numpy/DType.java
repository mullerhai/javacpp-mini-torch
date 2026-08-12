package org.bytedeco.pytorch.data.numpy;

import org.bytedeco.pytorch.global.torch.ScalarType;

/**
 * NumPy dtype descriptors used by {@code .npy}/{@code .npz} I/O and {@link NDArray}.
 * Little-endian only (standard for modern NumPy files).
 */
public enum DType {
    FLOAT64("<f8", 8, ScalarType.Double, false),
    FLOAT32("<f4", 4, ScalarType.Float, false),
    FLOAT16("<f2", 2, ScalarType.Half, false),
    INT64("<i8", 8, ScalarType.Long, false),
    INT32("<i4", 4, ScalarType.Int, false),
    INT16("<i2", 2, ScalarType.Short, false),
    INT8("|i1", 1, ScalarType.Char, false),
    UINT8("|u1", 1, ScalarType.Byte, false),
    BOOL("|b1", 1, ScalarType.Bool, false),
    /** Interleaved float32 complex — stored as double pairs in {@link NDArray}. */
    COMPLEX64("<c8", 8, ScalarType.ComplexFloat, true),
    /** Interleaved float64 complex. */
    COMPLEX128("<c16", 16, ScalarType.ComplexDouble, true);

    private final String descriptor;
    private final int byteSize;
    private final ScalarType torchType;
    private final boolean complex;

    DType(String descriptor, int byteSize, ScalarType torchType, boolean complex) {
        this.descriptor = descriptor;
        this.byteSize = byteSize;
        this.torchType = torchType;
        this.complex = complex;
    }

    public String getDescriptor() { return descriptor; }
    public int getByteSize() { return byteSize; }
    public ScalarType toTorch() { return torchType; }
    public boolean isComplex() { return complex; }

    public static DType fromDescriptor(String desc) {
        if (desc == null) return FLOAT64;
        String d = desc.trim();

        // First check exact match
        for (DType t : values()) {
            if (t.descriptor.equals(d)) return t;
        }

        // Normalize: strip byte order prefix (< or >) and unicode type
        String norm = d.startsWith("<") || d.startsWith(">") ? d.substring(1) : d;

        // Handle complex types first (longer strings)
        if (norm.startsWith("c") || d.contains("complex")) {
            if (norm.equals("c8") || norm.equals("c0") || d.contains("complex64")) return COMPLEX64;
            if (norm.equals("c16") || d.contains("complex128")) return COMPLEX128;
        }

        // Handle float types
        if (norm.equals("f8") || norm.equals("f0") || d.contains("float64")) return FLOAT64;
        if (norm.equals("f4") || norm.equals("f") || d.contains("float32")) return FLOAT32;
        if (norm.equals("f2") || d.contains("float16")) return FLOAT16;

        // Handle signed int types
        if (norm.equals("i8") || norm.startsWith("i") && norm.length() > 2 || d.contains("int64")) return INT64;
        if (norm.equals("i4") || d.contains("int32")) return INT32;
        if (norm.equals("i2") || d.contains("int16")) return INT16;
        if (norm.equals("i1") || d.contains("int8")) return INT8;

        // Handle unsigned int types
        if (norm.equals("u1") || d.contains("uint8")) return UINT8;
        if (norm.equals("u2") || d.contains("uint16")) return INT16;
        if (norm.equals("u4") || d.contains("uint32")) return INT32;
        if (norm.equals("u8") || d.contains("uint64")) return INT64;

        // Handle bool
        if (norm.equals("b1") || d.contains("bool")) return BOOL;

        return FLOAT64; // default
    }

    public static DType fromTorch(ScalarType st) {
        if (st == null) return FLOAT32;
        // JavaCPP: Tensor.scalar_type() returns a non-canonical proxy — intern first
        // or switch falls through to Byte (ordinal 0).
        ScalarType s = st.intern();
        switch (s) {
            case Double: return FLOAT64;
            case Float: return FLOAT32;
            case Half: return FLOAT16;
            case Long: return INT64;
            case Int: return INT32;
            case Short: return INT16;
            case Char: return INT8;
            case Byte: return UINT8;
            case Bool: return BOOL;
            case ComplexDouble: return COMPLEX128;
            case ComplexFloat: return COMPLEX64;
            default: return FLOAT32;
        }
    }
}
