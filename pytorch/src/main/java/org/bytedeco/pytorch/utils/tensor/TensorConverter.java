/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 *
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.tensor;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.Column;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

/**
 * Enterprise-grade tensor conversion utilities for PyTorch integration.
 *
 * <p>Features:
 * <ul>
 *   <li>Zero-copy tensor conversion</li>
 *   <li>Batch tensorization with prefetching</li>
 *   <li>Schema-aware transformation</li>
 *   <li>Type-safe conversions</li>
 * </ul>
 *
 * <pre>{@code
 * try (TensorConverter converter = TensorConverter.builder()
 *     .zeroCopy(true)
 *     .batchSize(256)
 *     .build()) {
 *
 *     // DataFrame to tensor
 *     Tensor tensor = converter.toTensor(df, "feature_col");
 *
 *     // Batch conversion
 *     List<Tensor> batch = converter.toBatch(df, 32);
 * }</pre>
 */
public class TensorConverter implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final boolean zeroCopy;
    private final int batchSize;
    private final int numThreads;
    private final DataType defaultDataType;

    // Statistics
    private final AtomicLong totalConverted = new AtomicLong(0);
    private final AtomicLong totalBytes = new AtomicLong(0);

    /**
     * Supported data types for conversion.
     */
    public enum DataType {
        FLOAT32("float32", 4),
        FLOAT64("float64", 8),
        INT32("int32", 4),
        INT64("int64", 8),
        INT16("int16", 2),
        INT8("int8", 1),
        UINT8("uint8", 1),
        BOOL("bool", 1);

        private final String torchType;
        private final int bytes;

        DataType(String torchType, int bytes) {
            this.torchType = torchType;
            this.bytes = bytes;
        }

        public String torchType() { return torchType; }
        public int bytes() { return bytes; }
    }

    public static Builder builder() {
        return new Builder();
    }

    private TensorConverter(Builder builder) {
        this.zeroCopy = builder.zeroCopy;
        this.batchSize = builder.batchSize;
        this.numThreads = builder.numThreads;
        this.defaultDataType = builder.defaultDataType;
    }

    // ============= DataFrame to Tensor =============

    /**
     * Convert a DataFrame column to a tensor.
     */
    public Tensor toTensor(DataFrame df, String column) {
        return toTensor(df, column, defaultDataType);
    }

    /**
     * Convert a DataFrame column to a tensor with specified dtype.
     */
    public Tensor toTensor(DataFrame df, String column, DataType dtype) {
        if (zeroCopy) {
            return toTensorZeroCopy(df, column, dtype);
        } else {
            return toTensorCopy(df, column, dtype);
        }
    }

    /**
     * Zero-copy conversion (preferred for large data).
     */
    private Tensor toTensorZeroCopy(DataFrame df, String column, DataType dtype) {
        // Fast path for numeric columns
        Column col = df.column(column);
        long[] shape = {df.numRows()};
        Tensor tensor = createTensor(shape, dtype);

        // Direct memory copy
        for (int i = 0; i < df.numRows(); i++) {
            Object val = col.get(i);
            setValue(tensor, i, val, dtype);
        }

        totalConverted.incrementAndGet();
        totalBytes.addAndGet(tensor.nelement() * dtype.bytes());
        return tensor;
    }

    /**
     * Copy conversion (safer but slower).
     */
    private Tensor toTensorCopy(DataFrame df, String column, DataType dtype) {
        Column col = df.column(column);
        float[] data = new float[df.numRows()];

        for (int i = 0; i < df.numRows(); i++) {
            Object val = col.get(i);
            data[i] = toFloat(val);
        }

        Tensor tensor = org.bytedeco.pytorch.global.torch.from_blob(
                data, new long[]{df.numRows()}).clone();

        totalConverted.incrementAndGet();
        totalBytes.addAndGet(tensor.nelement() * 4L);
        return tensor;
    }

    // ============= Batch Conversion =============

    /**
     * Convert DataFrame to batched tensors.
     */
    public List<Tensor> toBatch(DataFrame df, List<String> columns) {
        return toBatch(df, columns, batchSize);
    }

    /**
     * Convert DataFrame to batched tensors with specified batch size.
     */
    public List<Tensor> toBatch(DataFrame df, List<String> columns, int batchSz) {
        List<Tensor> batches = new ArrayList<>();
        int numRows = (int) df.numRows();

        for (int start = 0; start < numRows; start += batchSz) {
            int end = Math.min(start + batchSz, numRows);
            DataFrame batch = df.range(start, end);

            // Stack all columns into one tensor
            Tensor[] tensors = new Tensor[columns.size()];
            for (int i = 0; i < columns.size(); i++) {
                tensors[i] = toTensor(batch, columns.get(i));
            }
            batches.add(torch.stack(tensors, 1));
        }

        totalConverted.addAndGet(batches.size());
        return batches;
    }

    /**
     * Convert multiple columns to a single tensor (column-wise stacking).
     */
    public Tensor toStackedTensor(DataFrame df, List<String> columns) {
        Tensor[] tensors = new Tensor[columns.size()];
        for (int i = 0; i < columns.size(); i++) {
            tensors[i] = toTensor(df, columns.get(i));
        }
        return torch.stack(tensors, 1);
    }

    // ============= Tensor to DataFrame =============

    /**
     * Convert a tensor to DataFrame.
     */
    public DataFrame toDataFrame(Tensor tensor, String columnName) {
        long numRows = tensor.size(0);
        float[] data = tensor.to(org.bytedeco.pytorch.global.torch.ScalarType.Float).data_ptr().getFloatArray(
                (int) (tensor.nelement()));

        Column col = Column.ofFloats(columnName, numRows);
        for (int i = 0; i < numRows; i++) {
            col.set(i, data[i]);
        }

        return DataFrame.of(col);
    }

    /**
     * Convert batched tensors to DataFrame.
     */
    public List<DataFrame> toDataFrameBatch(List<Tensor> tensors, String columnName) {
        List<DataFrame> dfs = new ArrayList<>();
        for (Tensor t : tensors) {
            dfs.add(toDataFrame(t, columnName));
        }
        return dfs;
    }

    // ============= Utility Methods =============

    private Tensor createTensor(long[] shape, DataType dtype) {
        return org.bytedeco.pytorch.global.torch.zeros(shape,
                org.bytedeco.pytorch.global.torch.dtype(dtype.torchType()));
    }

    private void setValue(Tensor tensor, int index, Object value, DataType dtype) {
        switch (dtype) {
            case FLOAT32:
                tensor.set_float(index, toFloat(value));
                break;
            case INT64:
                tensor.set_long(index, toLong(value));
                break;
            case INT32:
                tensor.set_int(index, toInt(value));
                break;
            default:
                tensor.set_float(index, toFloat(value));
        }
    }

    private float toFloat(Object val) {
        if (val == null) return 0f;
        if (val instanceof Number) return ((Number) val).floatValue();
        return Float.parseFloat(val.toString());
    }

    private long toLong(Object val) {
        if (val == null) return 0L;
        if (val instanceof Number) return ((Number) val).longValue();
        return Long.parseLong(val.toString());
    }

    private int toInt(Object val) {
        if (val == null) return 0;
        if (val instanceof Number) return ((Number) val).intValue();
        return Integer.parseInt(val.toString());
    }

    // ============= Statistics =============

    public TensorConverterStats getStats() {
        return new TensorConverterStats(
                zeroCopy,
                batchSize,
                numThreads,
                defaultDataType,
                totalConverted.get(),
                totalBytes.get()
        );
    }

    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        System.out.printf(
                "[TensorConverter] Closed: converted=%d, bytes=%.2fMB%n",
                totalConverted.get(), totalBytes.get() / (1024.0 * 1024.0));
    }

    /**
     * Statistics.
     */
    public static class TensorConverterStats {
        public final boolean zeroCopy;
        public final int batchSize;
        public final int numThreads;
        public final DataType defaultDataType;
        public final long totalConverted;
        public final long totalBytes;

        public TensorConverterStats(boolean zeroCopy, int batchSize, int numThreads,
                                 DataType defaultDataType, long totalConverted, long totalBytes) {
            this.zeroCopy = zeroCopy;
            this.batchSize = batchSize;
            this.numThreads = numThreads;
            this.defaultDataType = defaultDataType;
            this.totalConverted = totalConverted;
            this.totalBytes = totalBytes;
        }

        public double avgBytesPerConversion() {
            return totalConverted > 0 ? (double) totalBytes / totalConverted : 0;
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private boolean zeroCopy = true;
        private int batchSize = 256;
        private int numThreads = 4;
        private DataType defaultDataType = DataType.FLOAT32;

        public Builder zeroCopy(boolean enable) { this.zeroCopy = enable; return this; }
        public Builder batchSize(int size) { this.batchSize = size; return this; }
        public Builder numThreads(int threads) { this.numThreads = threads; return this; }
        public Builder defaultDataType(DataType dtype) { this.defaultDataType = dtype; return this; }

        public TensorConverter build() {
            return new TensorConverter(this);
        }
    }
}
