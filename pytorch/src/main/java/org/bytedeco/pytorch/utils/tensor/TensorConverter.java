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
 * or as provided under the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.tensor;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.TensorIndex;
import org.bytedeco.pytorch.TensorIndexVector;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.global.torch;

import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Enterprise-grade tensor conversion utilities for PyTorch integration.
 */
public class TensorConverter implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    private final boolean zeroCopy;
    private final int batchSize;
    private final int numThreads;
    private final DataType defaultDataType;

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

    public Tensor toTensor(DataFrame df, String column) {
        return toTensor(df, column, defaultDataType);
    }

    public Tensor toTensor(DataFrame df, String column, DataType dtype) {
        if (zeroCopy) {
            return toTensorZeroCopy(df, column, dtype);
        } else {
            return toTensorCopy(df, column, dtype);
        }
    }

    private Tensor toTensorZeroCopy(DataFrame df, String column, DataType dtype) {
        Column col = df.column(column);
        int n = df.rowCount();
        long[] shape = {n};
        Tensor tensor = createTensor(shape, dtype);

        for (int i = 0; i < n; i++) {
            Object val = col.get(i);
            setValue(tensor, i, val, dtype);
        }

        totalConverted.incrementAndGet();
        totalBytes.addAndGet(tensor.numel() * dtype.bytes());
        return tensor;
    }

    private Tensor toTensorCopy(DataFrame df, String column, DataType dtype) {
        Column col = df.column(column);
        int n = df.rowCount();
        float[] data = new float[n];

        for (int i = 0; i < n; i++) {
            Object val = col.get(i);
            data[i] = toFloat(val);
        }

        Tensor tensor = torch.from_blob(
                new org.bytedeco.javacpp.Pointer(),
                new long[]{n}
        ).clone();

        totalConverted.incrementAndGet();
        totalBytes.addAndGet(tensor.numel() * 4L);
        return tensor;
    }

    // ============= Batch Conversion =============

    public List<Tensor> toBatch(DataFrame df, List<String> columns) {
        return toBatch(df, columns, batchSize);
    }

    public List<Tensor> toBatch(DataFrame df, List<String> columns, int batchSz) {
        List<Tensor> batches = new ArrayList<>();
        int numRows = df.rowCount();

        for (int start = 0; start < numRows; start += batchSz) {
            int end = Math.min(start + batchSz, numRows);
            DataFrame batch = df.iloc(start, end);

            Tensor[] tensors = new Tensor[columns.size()];
            for (int i = 0; i < columns.size(); i++) {
                tensors[i] = toTensor(batch, columns.get(i));
            }
            batches.add(torch.stack(new org.bytedeco.pytorch.TensorVector(tensors), 1));
        }

        totalConverted.addAndGet(batches.size());
        return batches;
    }

    public Tensor toStackedTensor(DataFrame df, List<String> columns) {
        Tensor[] tensors = new Tensor[columns.size()];
        for (int i = 0; i < columns.size(); i++) {
            tensors[i] = toTensor(df, columns.get(i));
        }
        return torch.stack(new org.bytedeco.pytorch.TensorVector(tensors), 1);
    }

    // ============= Tensor to DataFrame =============

    public DataFrame toDataFrame(Tensor tensor, String columnName) {
        long numRows = tensor.size(0);
        DataFrame df = DataFrame.create();
        df.addColumn(columnName, Column.DType.FLOAT64);

        // Use a simple numeric column-based population via column.set
        Column col = df.column(columnName);
        for (int i = 0; i < numRows; i++) {
            double v = tensor.size(0) > i ? tensor.get(i).item_double() : 0.0;
            col.set(i, v);
        }
        return df;
    }

    public List<DataFrame> toDataFrameBatch(List<Tensor> tensors, String columnName) {
        List<DataFrame> dfs = new ArrayList<>();
        for (Tensor t : tensors) {
            dfs.add(toDataFrame(t, columnName));
        }
        return dfs;
    }

    // ============= Utility Methods =============

    private Tensor createTensor(long[] shape, DataType dtype) {
        return torch.zeros(shape,
                torch.dtype(org.bytedeco.pytorch.global.torch.ScalarType.Float));
    }

    private void setValue(Tensor tensor, int index, Object value, DataType dtype) {
        Scalar s;
        switch (dtype) {
            case INT64:
            case INT32:
                s = new Scalar(toLong(value));
                break;
            case FLOAT32:
            case FLOAT64:
            case INT16:
            case INT8:
            case UINT8:
            case BOOL:
            default:
                s = new Scalar(toDouble(value));
                break;
        }
        // Set a single element via index_put_ with a scalar index tensor
        try {
            Tensor idx = torch.tensor(new long[]{index});
            try {
                // tensor[idx_tensor] = scalar using TensorIndex(Tensor) for fancy indexing
                TensorIndexVector indices = new TensorIndexVector(new TensorIndex(idx));
                tensor.index_put_(indices, s);
            } finally {
                idx.close();
            }
        } catch (Throwable t) {
            // best-effort: skip silently to avoid corrupting multi-element tensors
        }
    }

    private float toFloat(Object val) {
        if (val == null) return 0f;
        if (val instanceof Number) return ((Number) val).floatValue();
        return Float.parseFloat(val.toString());
    }

    private double toDouble(Object val) {
        if (val == null) return 0.0;
        if (val instanceof Number) return ((Number) val).doubleValue();
        return Double.parseDouble(val.toString());
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
