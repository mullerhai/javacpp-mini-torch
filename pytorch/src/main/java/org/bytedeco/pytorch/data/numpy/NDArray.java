package org.bytedeco.pytorch.data.numpy;

import java.util.Arrays;

/**
 * Dense multi-dimensional array backed by a contiguous (or strided) Java primitive buffer.
 * Float family and complex use {@code double[]} (complex is interleaved re,im pairs).
 * Integer/bool family use {@code long[]}.
 *
 * <p>Supports optional non-contiguous views via {@link #asStrided(long[], long[], long)} —
 * element access goes through strides; mutators write through to the shared base buffer.
 */
public final class NDArray {
    public final long[] shape;
    public final long size;
    public final DType dtype;

    /** Element strides (in elements for real; in complex-elements for complex). Null → C-contiguous. */
    public final long[] strides;
    /** Base offset into storage (in real elements, or complex-elements*2 for complex pair index base). */
    public final long offset;

    /** True when this array shares storage and may be non-contiguous. */
    public final boolean isView;

    /** Element storage as doubles (float + complex) or longs (integer/bool). */
    private final double[] fdata;
    private final long[] idata;

    // ---- constructors: owning contiguous ------------------------------------

    public NDArray(DType dtype, long... shape) {
        this.dtype = dtype != null ? dtype : DType.FLOAT64;
        this.shape = shape != null ? shape.clone() : new long[0];
        long n = 1;
        for (long s : this.shape) n *= s;
        this.size = n;
        this.strides = contigStrides(this.shape, this.dtype);
        this.offset = 0;
        this.isView = false;
        if (isFloatOrComplex(this.dtype)) {
            int cap = this.dtype.isComplex() ? (int) (n * 2) : (int) n;
            this.fdata = new double[cap];
            this.idata = null;
        } else {
            this.fdata = null;
            this.idata = new long[(int) n];
        }
    }

    public NDArray(double[] data, long... shape) {
        this.dtype = DType.FLOAT64;
        this.shape = normalizeDataShape(shape, data.length);
        this.size = numel(this.shape);
        this.strides = contigStrides(this.shape, this.dtype);
        this.offset = 0;
        this.isView = false;
        this.fdata = data;
        this.idata = null;
    }

    public NDArray(float[] data, long... shape) {
        this.dtype = DType.FLOAT32;
        this.shape = normalizeDataShape(shape, data.length);
        this.size = numel(this.shape);
        this.strides = contigStrides(this.shape, this.dtype);
        this.offset = 0;
        this.isView = false;
        this.fdata = new double[data.length];
        for (int i = 0; i < data.length; i++) this.fdata[i] = data[i];
        this.idata = null;
    }

    public NDArray(long[] data, DType dtype, long... shape) {
        this.dtype = dtype != null ? dtype : DType.INT64;
        this.shape = normalizeDataShape(shape, data.length);
        this.size = numel(this.shape);
        this.strides = contigStrides(this.shape, this.dtype);
        this.offset = 0;
        this.isView = false;
        this.fdata = null;
        this.idata = data;
    }

    /**
     * Complex from interleaved re/im doubles: data length must be {@code 2 * numel(shape)}.
     */
    public NDArray(double[] interleavedComplex, DType complexDtype, long... shape) {
        if (complexDtype == null || !complexDtype.isComplex()) {
            throw new IllegalArgumentException("complex dtype required");
        }
        this.dtype = complexDtype;
        this.shape = normalizeDataShape(shape, interleavedComplex.length / 2);
        this.size = numel(this.shape);
        if (interleavedComplex.length < this.size * 2) {
            throw new IllegalArgumentException("complex buffer too short");
        }
        this.strides = contigStrides(this.shape, this.dtype);
        this.offset = 0;
        this.isView = false;
        this.fdata = interleavedComplex;
        this.idata = null;
    }

    /** Empty or null varargs → 1D {@code [len]}; otherwise clone shape. */
    private static long[] normalizeDataShape(long[] shape, long len) {
        if (shape == null || shape.length == 0) return new long[]{len};
        return shape.clone();
    }

    /** Internal: shared-storage view. */
    NDArray(DType dtype, long[] shape, long[] strides, long offset,
            double[] fdata, long[] idata, boolean isView) {
        this.dtype = dtype;
        this.shape = shape;
        this.size = numel(shape);
        this.strides = strides;
        this.offset = offset;
        this.isView = isView;
        this.fdata = fdata;
        this.idata = idata;
    }

    public static boolean isFloatFamily(DType d) {
        return d == DType.FLOAT64 || d == DType.FLOAT32 || d == DType.FLOAT16;
    }

    public static boolean isFloatOrComplex(DType d) {
        return isFloatFamily(d) || d.isComplex();
    }

    public static long numel(long[] shape) {
        long n = 1;
        for (long s : shape) n *= s;
        return n;
    }

    public static long[] contigStrides(long[] shape, DType dtype) {
        long[] st = new long[shape.length];
        long acc = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            st[i] = acc;
            acc *= shape[i];
        }
        return st;
    }

    public boolean isContiguous() {
        if (offset != 0) return false;
        return hasContiguousStrides();
    }

    public boolean hasContiguousStrides() {
        return Arrays.equals(strides, contigStrides(shape, dtype));
    }

    public boolean isComplex() { return dtype.isComplex(); }

    // ---- flat element access (logical C-order index 0..size-1) --------------

    private int storageIndex(int flatLogical) {
        if (shape.length == 0) return (int) offset;
        if (hasContiguousStrides()) return (int) (offset + flatLogical);
        long idx = offset;
        long left = flatLogical;
        for (int d = 0; d < shape.length; d++) {
            long strideC = 1;
            for (int k = d + 1; k < shape.length; k++) strideC *= shape[k];
            long coord = shape[d] == 0 ? 0 : (left / strideC) % shape[d];
            idx += coord * strides[d];
        }
        return (int) idx;
    }

    public double getDouble(int flatIndex) {
        if (dtype.isComplex()) {
            // return real part for scalar-double views of complex
            int si = storageIndex(flatIndex);
            return fdata[si * 2];
        }
        int si = storageIndex(flatIndex);
        if (fdata != null) return fdata[si];
        return idata[si];
    }

    public long getLong(int flatIndex) {
        if (dtype.isComplex()) return (long) getDouble(flatIndex);
        int si = storageIndex(flatIndex);
        if (idata != null) return idata[si];
        return (long) fdata[si];
    }

    public void setDouble(int flatIndex, double v) {
        if (dtype.isComplex()) {
            int si = storageIndex(flatIndex);
            fdata[si * 2] = v;
            // leave imag unchanged
            return;
        }
        int si = storageIndex(flatIndex);
        if (fdata != null) fdata[si] = v;
        else idata[si] = (long) v;
    }

    public void setLong(int flatIndex, long v) {
        if (dtype.isComplex()) {
            setDouble(flatIndex, v);
            return;
        }
        int si = storageIndex(flatIndex);
        if (idata != null) idata[si] = v;
        else fdata[si] = v;
    }

    /** Complex element access. */
    public double getReal(int flatIndex) {
        if (!dtype.isComplex()) return getDouble(flatIndex);
        return fdata[storageIndex(flatIndex) * 2];
    }

    public double getImag(int flatIndex) {
        if (!dtype.isComplex()) return 0.0;
        return fdata[storageIndex(flatIndex) * 2 + 1];
    }

    public void setComplex(int flatIndex, double re, double im) {
        if (!dtype.isComplex()) {
            setDouble(flatIndex, re);
            return;
        }
        int si = storageIndex(flatIndex) * 2;
        fdata[si] = re;
        fdata[si + 1] = im;
    }

    public double[] asDoubleArray() {
        if (dtype.isComplex()) {
            double[] o = new double[(int) size];
            for (int i = 0; i < size; i++) o[i] = getReal(i);
            return o;
        }
        if (isContiguous() && fdata != null && offset == 0 && fdata.length >= size) {
            if (fdata.length == size) return fdata;
        }
        double[] o = new double[(int) size];
        for (int i = 0; i < size; i++) o[i] = getDouble(i);
        return o;
    }

    public float[] asFloatArray() {
        float[] o = new float[(int) size];
        for (int i = 0; i < size; i++) o[i] = (float) getDouble(i);
        return o;
    }

    public long[] asLongArray() {
        if (isContiguous() && idata != null && offset == 0 && idata.length == size) return idata;
        long[] o = new long[(int) size];
        for (int i = 0; i < size; i++) o[i] = getLong(i);
        return o;
    }

    public int[] asIntArray() {
        int[] o = new int[(int) size];
        for (int i = 0; i < size; i++) o[i] = (int) getLong(i);
        return o;
    }

    /** Interleaved re,im of length {@code 2*size}. Copies. */
    public double[] asInterleavedComplex() {
        double[] o = new double[(int) size * 2];
        if (dtype.isComplex()) {
            for (int i = 0; i < size; i++) {
                o[i * 2] = getReal(i);
                o[i * 2 + 1] = getImag(i);
            }
        } else {
            for (int i = 0; i < size; i++) {
                o[i * 2] = getDouble(i);
                o[i * 2 + 1] = 0;
            }
        }
        return o;
    }

    public long numel() { return size; }

    public int ndim() { return shape.length; }

    /**
     * NumPy {@code as_strided}: create a view with given shape/strides sharing storage.
     * Strides are in <em>elements</em> (complex-elements for complex dtypes).
     */
    public NDArray asStrided(long[] newShape, long[] newStrides, long newOffset) {
        if (newShape.length != newStrides.length) {
            throw new IllegalArgumentException("shape/strides rank mismatch");
        }
        return new NDArray(dtype, newShape.clone(), newStrides.clone(), offset + newOffset,
                fdata, idata, true);
    }

    public NDArray asStrided(long[] newShape, long[] newStrides) {
        return asStrided(newShape, newStrides, 0);
    }

    /** Deep copy to contiguous owning array. */
    public NDArray copy() {
        if (dtype.isComplex()) {
            return new NDArray(asInterleavedComplex(), dtype, shape.clone());
        }
        if (isFloatFamily(dtype)) {
            NDArray out = new NDArray(dtype, shape.clone());
            for (int i = 0; i < size; i++) out.setDouble(i, getDouble(i));
            return out;
        }
        NDArray out = new NDArray(dtype, shape.clone());
        for (int i = 0; i < size; i++) out.setLong(i, getLong(i));
        return out;
    }

    /** Write this array's logical contents into {@code out} (must match size). */
    public void copyTo(NDArray out) {
        if (out.size != this.size) throw new IllegalArgumentException("copyTo size mismatch");
        if (out.dtype.isComplex() && dtype.isComplex()) {
            for (int i = 0; i < size; i++) out.setComplex(i, getReal(i), getImag(i));
        } else if (out.dtype.isComplex()) {
            for (int i = 0; i < size; i++) out.setComplex(i, getDouble(i), 0);
        } else {
            for (int i = 0; i < size; i++) out.setDouble(i, getDouble(i));
        }
    }

    double[] rawFData() { return fdata; }
    long[] rawIData() { return idata; }

    /** Convert to a javacpp-pytorch {@link org.bytedeco.pytorch.Tensor} (delegates to {@link NP#toTensor}). */
    public org.bytedeco.pytorch.Tensor toTensor() {
        return NP.toTensor(this);
    }

    /** Build from a torch Tensor (delegates to {@link NP#fromTensor}). */
    public static NDArray fromTensor(org.bytedeco.pytorch.Tensor t) {
        return NP.fromTensor(t);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("NDArray(dtype=").append(dtype)
                .append(", shape=").append(Arrays.toString(shape))
                .append(", size=").append(size);
        if (isView) sb.append(", view=true");
        sb.append(")");
        return sb.toString();
    }

    /** Print Spark/Polars-style schema tree (one-line per field). */
    public void printSchema() {
        NumpySchema.forArray(this).printSchema();
    }

    /** Return the schema descriptor for this array. */
    public NumpySchema schema() {
        return NumpySchema.forArray(this);
    }

    /** Print first {@code n} elements (default 20) using NumPy-style formatting.
     *  2-D arrays display as a proper row-per-line matrix with head/tail row truncation.
     *  1-D and higher-rank arrays display as a nested block. */
    public void show() { show(20); }

    /** Show at most {@code maxRows} rows of a 2-D array (or {@code maxRows} elements
     *  for 1-D, or {@code maxRows} sub-arrays for N-D). NumPy-style. */
    public void show(int maxRows) {
        int ndim = shape.length;
        if (ndim == 1) {
            System.out.println(new NumpyShow().format(this));
        } else if (ndim == 2) {
            // Show first maxRows rows with NumPy-style head/tail.
            long rows = shape[0];
            long cols = shape[1];
            if (rows <= maxRows) {
                // Show all rows
                System.out.println(new NumpyShow().format(this));
            } else {
                // Build a slice with first maxRows/2 and last maxRows/2 rows
                int half = maxRows / 2;
                NDArray top = slice(0, half, 0, cols);
                NDArray bot = slice(rows - half, rows, 0, cols);
                NumpyShow ns = new NumpyShow(NumpyShow.DEFAULT_THRESHOLD, NumpyShow.DEFAULT_EDGEITEMS, NumpyShow.DEFAULT_LINEWIDTH);
                System.out.println(ns.format(top));
                System.out.println("...");
                System.out.println(ns.format(bot));
            }
        } else {
            // N-D: show full nested form (truncated at outer dimension)
            long firstDim = shape[0];
            if (firstDim <= maxRows) {
                System.out.println(new NumpyShow().format(this));
            } else {
                // Show only the first chunk and last chunk
                int half = maxRows / 2;
                long[] topShape = shape.clone();
                topShape[0] = half;
                long[] botShape = shape.clone();
                botShape[0] = half;
                long[] botStart = shape.clone();
                botStart[0] = firstDim - half;
                NDArray top = slice(topShape);
                NDArray bot = slice(botStart, botShape);
                NumpyShow ns = new NumpyShow();
                System.out.println(ns.format(top));
                System.out.println("...");
                System.out.println(ns.format(bot));
            }
        }
        System.out.printf("[%d elements, dtype=%s, shape=%s]%n", size, dtype, java.util.Arrays.toString(shape));
    }

    /**
     * Format the first {@code n} scalar elements as a NumPy-style block.
     * Note: for 2-D arrays, prefer {@link #show(int)} which preserves matrix layout.
     */
    public String format(int n) {
        int ndim = shape.length;
        if (ndim == 2 && shape[1] > 0) {
            // Preserve 2-D matrix layout: show at most n rows.
            long rows = shape[0];
            long cols = shape[1];
            int showRows = (int) Math.min(n, rows);
            NDArray visible = slice(0, showRows, 0, cols);
            return new NumpyShow().format(visible);
        }
        return new NumpyShow().format(head(n));
    }

    /** Format all elements (with NumPy's head/tail truncation rules). */
    public String formatAll() {
        return new NumpyShow().format(this);
    }

    /** Slice the first {@code n} elements as a new flat array (1-D). */
    public NDArray head(int n) {
        long take = Math.min(n, (int) size);
        NDArray out = new NDArray(dtype, new long[]{take});
        if (isFloatOrComplex(dtype)) {
            for (int i = 0; i < take; i++) out.setDouble(i, getDouble(i));
        } else {
            for (int i = 0; i < take; i++) out.setLong(i, getLong(i));
        }
        return out;
    }

    /** 2-D slice: rows [r0, r1) and columns [c0, c1). Returns a new NDArray. */
    public NDArray slice(long r0, long r1, long c0, long c1) {
        long[] newShape = new long[]{Math.max(0, r1 - r0), Math.max(0, c1 - c0)};
        NDArray out = new NDArray(dtype, newShape);
        for (long r = r0; r < r1; r++) {
            for (long c = c0; c < c1; c++) {
                long dstIdx = (r - r0) * newShape[1] + (c - c0);
                long srcIdx = r * shape[1] + c;
                if (isFloatOrComplex(dtype)) {
                    out.setDouble((int) dstIdx, getDouble((int) srcIdx));
                } else {
                    out.setLong((int) dstIdx, getLong((int) srcIdx));
                }
            }
        }
        return out;
    }

    /** N-D slice starting at {@code start} with output {@code outShape}. */
    public NDArray slice(long[] start, long[] outShape) {
        if (start.length != shape.length || outShape.length != shape.length) {
            throw new IllegalArgumentException("start and outShape must match array rank " + shape.length);
        }
        long total = 1;
        for (long d : outShape) total *= d;
        long[] idx = new long[shape.length];
        long[] flatIdx = new long[shape.length];
        NDArray out = new NDArray(dtype, outShape);
        for (long flat = 0; flat < total; flat++) {
            // Unravel flat into output coordinates
            long tmp = flat;
            for (int d = outShape.length - 1; d >= 0; d--) {
                flatIdx[d] = tmp % outShape[d];
                tmp /= outShape[d];
            }
            // Map to source coordinates
            long srcFlat = 0, stride = 1;
            for (int d = shape.length - 1; d >= 0; d--) {
                srcFlat += (start[d] + flatIdx[d]) * stride;
                stride *= shape[d];
            }
            if (isFloatOrComplex(dtype)) {
                out.setDouble((int) flat, getDouble((int) srcFlat));
            } else {
                out.setLong((int) flat, getLong((int) srcFlat));
            }
        }
        return out;
    }

    /** Slice from origin with given output shape. */
    public NDArray slice(long[] outShape) { return slice(new long[shape.length], outShape); }

    /** Slice the first {@code n} rows of a 2-D array (all columns). */
    public NDArray headRows(int n) {
        long rows = shape[0];
        return slice(0, Math.min(n, rows), 0, shape[1]);
    }
}
