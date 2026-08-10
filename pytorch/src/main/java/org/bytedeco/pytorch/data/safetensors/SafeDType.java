package org.bytedeco.pytorch.data.safetensors;

import org.bytedeco.pytorch.global.torch.ScalarType;

import java.util.Locale;

/**
 * Dtypes used by the safetensors format (little-endian on disk).
 *
 * <p>Supports:
 * <ul>
 *   <li>Standard dtypes: F64, F32, F16, BF16, FP8</li>
 *   <li>Integer types: I64, I32, I16, I8, U8, BOOL</li>
 *   <li>Quantized dtypes: NF4, FP4, Q80, Q40 (via {@code format.SafeDType})</li>
 * </ul>
 */
public enum SafeDType {
    F64("F64", 8, ScalarType.Double, true),
    F32("F32", 4, ScalarType.Float, true),
    F16("F16", 2, ScalarType.Half, true),
    BF16("BF16", 2, ScalarType.BFloat16, true),
    /** FP8 E4M3 (safetensors name {@code F8_E4M3}); maps to torch Float8_e4m3fn. */
    F8_E4M3("F8_E4M3", 1, ScalarType.Float8_e4m3fn, true),
    /** FP8 E5M2 (safetensors name {@code F8_E5M2}). */
    F8_E5M2("F8_E5M2", 1, ScalarType.Float8_e5m2, true),
    I64("I64", 8, ScalarType.Long, true),
    I32("I32", 4, ScalarType.Int, true),
    I16("I16", 2, ScalarType.Short, true),
    I8("I8", 1, ScalarType.Char, true),
    U8("U8", 1, ScalarType.Byte, true),
    BOOL("BOOL", 1, ScalarType.Bool, true);

    private final String name;
    private final int bytes;
    private final ScalarType torch;
    private final boolean nativeLayout;

    SafeDType(String name, int bytes, ScalarType torch, boolean nativeLayout) {
        this.name = name;
        this.bytes = bytes;
        this.torch = torch;
        this.nativeLayout = nativeLayout;
    }

    public String typeName() { return name; }
    public int sizeBytes() { return bytes; }
    public ScalarType toTorch() { return torch; }

    /**
     * Whether on-disk little-endian layout matches torch storage so
     * {@code from_blob} can share the mapping without conversion.
     * F16/BF16/FP8 are native 1:1; BOOL is not (torch may pack differently).
     */
    public boolean isNativeLayout() {
        return nativeLayout;
    }

    public static SafeDType fromString(String s) {
        if (s == null) return null;
        switch (s.toUpperCase(Locale.ROOT)) {
            case "F64": case "FLOAT64": case "DOUBLE": return F64;
            case "F32": case "FLOAT32": case "FLOAT": return F32;
            case "F16": case "FLOAT16": case "HALF": return F16;
            case "BF16": case "BFLOAT16": return BF16;
            case "F8_E4M3": case "F8E4M3": case "FLOAT8_E4M3FN": case "FLOAT8_E4M3": return F8_E4M3;
            case "F8_E5M2": case "F8E5M2": case "FLOAT8_E5M2": return F8_E5M2;
            case "I64": case "INT64": case "LONG": return I64;
            case "I32": case "INT32": case "INT": return I32;
            case "I16": case "INT16": case "SHORT": return I16;
            case "I8": case "INT8": return I8;
            case "U8": case "UINT8": return U8;
            case "BOOL": case "BOOLEAN": return BOOL;
            default: return null;
        }
    }

    /**
     * Map a torch ScalarType to safetensors dtype.
     *
     * <p><b>JavaCPP pitfall:</b> {@code Tensor.scalar_type()} often returns a
     * non-canonical enum proxy ({@code name=null}, {@code ordinal=0}) whose
     * {@code switch} identity matches {@link ScalarType#Byte}. Always
     * {@link ScalarType#intern()} first so case labels resolve by real value
     * ({@code Float.value=6}, etc.). Matching on {@code st.value} is an
     * equivalent alternative.
     */
    public static SafeDType fromTorch(ScalarType st) {
        if (st == null) return F32;
        // intern() maps the native-backed proxy onto the canonical enum constant
        ScalarType s = st.intern();
        switch (s) {
            case Double: return F64;
            case Float: return F32;
            case Half: return F16;
            case BFloat16: return BF16;
            case Float8_e4m3fn: return F8_E4M3;
            case Float8_e5m2: return F8_E5M2;
            case Long: return I64;
            case Int: return I32;
            case Short: return I16;
            case Char: return I8;
            case Byte: return U8;
            case Bool: return BOOL;
            default: return F32;
        }
    }
}
