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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.daft.window;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.daft.expr.Expression;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Window functions for DaftDataFrame.
 *
 * <p>Supports:
 * <ul>
 *   <li>ROW_NUMBER, RANK, DENSE_RANK, PERCENT_RANK</li>
 *   <li>LAG, LEAD (access neighboring rows)</li>
 *   <li>FIRST_VALUE, LAST_VALUE, NTH_VALUE</li>
 *   <li>NTILE (bucketing)</li>
 *   <li>Cumulative: SUM, AVG, MAX, MIN over window</li>
 * </ul>
 *
 * <pre>{@code
 * DaftDataFrame df = DaftDataFrame.fromParquet("data.parquet");
 * df.select(
 *     col("user_id"),
 *     col("ts"),
 *     Functions.row_number().over(Window.partitionBy("user_id").orderBy("ts")),
 *     Functions.lag("ts", 1).over(Window.partitionBy("user_id")),
 *     Functions.lead("value", 1).over(Window.partitionBy("user_id"))
 * ).show();
 * }</pre>
 */
public final class DaftWindow {

    private final List<String> partitionBy;
    private final List<Expression> orderBy;
    private final WindowFrame frame;

    private DaftWindow(List<String> partitionBy, List<Expression> orderBy, WindowFrame frame) {
        this.partitionBy = partitionBy != null ? partitionBy : Collections.<String>emptyList();
        this.orderBy = orderBy != null ? orderBy : Collections.<Expression>emptyList();
        this.frame = frame != null ? frame : WindowFrame.unbounded();
    }

    public static DaftWindow of() {
        return new DaftWindow(null, null, WindowFrame.unbounded());
    }

    public DaftWindow partitionBy(String... columns) {
        List<String> parts = new ArrayList<>(this.partitionBy);
        for (String c : columns) parts.add(c);
        return new DaftWindow(parts, this.orderBy, this.frame);
    }

    public DaftWindow orderBy(Expression... exprs) {
        List<Expression> order = new ArrayList<>(this.orderBy);
        for (Expression e : exprs) order.add(e);
        return new DaftWindow(this.partitionBy, order, this.frame);
    }

    public DaftWindow orderBy(String column) {
        return orderBy(new org.bytedeco.pytorch.utils.daft.expr.ColumnRef(column));
    }

    public DaftWindow frame(WindowFrame frame) {
        return new DaftWindow(this.partitionBy, this.orderBy, frame);
    }

    public List<String> partitionBy() { return partitionBy; }
    public List<Expression> orderBy() { return orderBy; }
    public WindowFrame frame() { return frame; }

    /**
     * Apply window function to a DataFrame.
     * Returns new DataFrame with window column added.
     */
    public DataFrame apply(DataFrame df, String outputColumn, WindowFunction fn) {
        DataFrame result = df.copy();
        List<Integer> partitionIndices = resolvePartitionIndices(df);
        List<Integer> orderIndices = resolveOrderIndices(df);
        boolean hasPartition = !partitionBy.isEmpty();
        boolean hasOrder = !orderBy.isEmpty();

        Column out = new Column(outputColumn, Column.DType.INT64);
        int rows = result.rowCount();
        List<Object> values = new ArrayList<>(rows);

        for (int i = 0; i < rows; i++) {
            int start = frame.start(hasPartition, partitionIndices, i);
            int end = frame.end(hasPartition, partitionIndices, i, rows);
            values.add(fn.apply(df, i, start, end, hasPartition, hasOrder,
                    partitionIndices, orderIndices));
        }

        out.addAll(values);
        return result.withColumn(outputColumn, values);
    }

    private List<Integer> resolvePartitionIndices(DataFrame df) {
        if (partitionBy.isEmpty()) return Collections.emptyList();
        List<Integer> indices = new ArrayList<>();
        for (String col : partitionBy) {
            for (int i = 0; i < df.columnCount(); i++) {
                if (df.column(i).name().equals(col)) {
                    indices.add(i);
                    break;
                }
            }
        }
        return indices;
    }

    private List<Integer> resolveOrderIndices(DataFrame df) {
        if (orderBy.isEmpty()) return Collections.emptyList();
        List<Integer> indices = new ArrayList<>();
        for (Expression e : orderBy) {
            if (e instanceof org.bytedeco.pytorch.utils.daft.expr.ColumnRef) {
                String name = ((org.bytedeco.pytorch.utils.daft.expr.ColumnRef) e).name();
                for (int i = 0; i < df.columnCount(); i++) {
                    if (df.column(i).name().equals(name)) {
                        indices.add(i);
                        break;
                    }
                }
            }
        }
        return indices;
    }

    /**
     * Window frame specification (ROWS / RANGE between start and end).
     */
    public static final class WindowFrame {
        public enum Mode { ROWS, RANGE }

        private final Mode mode;
        private final int startOffset;
        private final int endOffset;
        private final boolean startUnbounded;
        private final boolean endUnbounded;

        private WindowFrame(Mode mode, int startOffset, int endOffset,
                           boolean startUnbounded, boolean endUnbounded) {
            this.mode = mode;
            this.startOffset = startOffset;
            this.endOffset = endOffset;
            this.startUnbounded = startUnbounded;
            this.endUnbounded = endUnbounded;
        }

        public static WindowFrame unbounded() {
            return new WindowFrame(Mode.ROWS, 0, 0, true, true);
        }

        public static WindowFrame rowsBetween(int start, int end) {
            return new WindowFrame(Mode.ROWS, start, end, false, false);
        }

        public static WindowFrame rangeBetween(int start, int end) {
            return new WindowFrame(Mode.RANGE, start, end, false, false);
        }

        public WindowFrame rowsUnboundedPreceding() {
            return new WindowFrame(mode, 0, endOffset, true, endUnbounded);
        }

        public WindowFrame rowsUnboundedFollowing() {
            return new WindowFrame(mode, startOffset, 0, startUnbounded, true);
        }

        public WindowFrame rowsNPreceding(int n) {
            return new WindowFrame(mode, -n, endOffset, false, endUnbounded);
        }

        public WindowFrame rowsNFollowing(int n) {
            return new WindowFrame(mode, startOffset, n, startUnbounded, false);
        }

        int start(boolean hasPartition, List<Integer> partitionIndices, int currentRow) {
            if (startUnbounded || !hasPartition) return 0;
            return Math.max(0, currentRow + startOffset);
        }

        int end(boolean hasPartition, List<Integer> partitionIndices, int currentRow, int totalRows) {
            if (endUnbounded || !hasPartition) return totalRows;
            return Math.min(totalRows, currentRow + endOffset + 1);
        }
    }

    /**
     * Window function interface.
     */
    public interface WindowFunction {
        Object apply(DataFrame df, int row, int start, int end,
                    boolean hasPartition, boolean hasOrder,
                    List<Integer> partitionIndices, List<Integer> orderIndices);
    }

    /**
     * Built-in window functions.
     */
    public static final class Functions {
        private Functions() {}

        public static WindowFunction rowNumber() {
            return (df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx) -> {
                if (hasPartition) {
                    int groupStart = findGroupStart(partitionIndices(df, partIdx, row), row);
                    return row - groupStart + 1;
                }
                return row + 1;
            };
        }

        public static WindowFunction rank() {
            return (df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx) -> {
                if (hasPartition && hasOrder) {
                    int groupStart = findGroupStart(df, partIdx, orderIdx, row);
                    return groupStart + 1;
                }
                return row + 1;
            };
        }

        public static WindowFunction denseRank() {
            return (df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx) -> {
                if (hasPartition && hasOrder) {
                    Object currentVal = getOrderValue(df, orderIdx, row);
                    int rank = 1;
                    int groupStart = findGroupStart(df, partIdx, orderIdx, row);
                    for (int i = groupStart; i < row; i++) {
                        Object val = getOrderValue(df, orderIdx, i);
                        if (compare(val, currentVal) < 0) rank++;
                    }
                    return rank;
                }
                return row + 1;
            };
        }

        public static WindowFunction percentRank() {
            return (df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx) -> {
                long r = ((Number) rank().apply(df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx)).longValue();
                long total = (Long) rowNumber().apply(df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx);
                if (total <= 1) return 0.0;
                return (double) (r - 1) / (total - 1);
            };
        }

        public static WindowFunction ntile(int buckets) {
            return (df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx) -> {
                if (!hasPartition) {
                    long total = df.rowCount();
                    return (row * buckets / total) + 1;
                }
                int groupSize = findGroupSize(df, partIdx, row);
                int offsetInGroup = row - findGroupStart(partitionIndices(df, partIdx, row), row);
                return (offsetInGroup * buckets / groupSize) + 1;
            };
        }

        public static WindowFunction lag(String column, int offset) {
            return (df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx) -> {
                int colIdx = findColumnIndex(df, column);
                if (colIdx < 0) return null;
                int targetRow;
                if (hasPartition) {
                    int groupStart = findGroupStart(partitionIndices(df, partIdx, row), row);
                    targetRow = row - offset;
                    if (targetRow < groupStart) return null;
                } else {
                    targetRow = row - offset;
                    if (targetRow < 0) return null;
                }
                return df.column(colIdx).get(targetRow);
            };
        }

        public static WindowFunction lead(String column, int offset) {
            return (df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx) -> {
                int colIdx = findColumnIndex(df, column);
                if (colIdx < 0) return null;
                int targetRow;
                if (hasPartition) {
                    int groupEnd = findGroupEnd(df, partIdx, row);
                    targetRow = row + offset;
                    if (targetRow >= groupEnd) return null;
                } else {
                    targetRow = row + offset;
                    if (targetRow >= df.rowCount()) return null;
                }
                return df.column(colIdx).get(targetRow);
            };
        }

        public static WindowFunction firstValue(String column) {
            return lag(column, Integer.MAX_VALUE);
        }

        public static WindowFunction lastValue(String column) {
            return lead(column, Integer.MAX_VALUE);
        }

        public static WindowFunction nthValue(String column, int n) {
            return lag(column, n - 1);
        }

        public static WindowFunction sum(String column) {
            return (df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx) -> {
                int colIdx = findColumnIndex(df, column);
                if (colIdx < 0) return null;
                double sum = 0;
                for (int i = start; i < end; i++) {
                    Object v = df.column(colIdx).get(i);
                    if (v instanceof Number) sum += ((Number) v).doubleValue();
                }
                return sum;
            };
        }

        public static WindowFunction avg(String column) {
            return (df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx) -> {
                int colIdx = findColumnIndex(df, column);
                if (colIdx < 0) return null;
                double sum = 0;
                int count = 0;
                for (int i = start; i < end; i++) {
                    Object v = df.column(colIdx).get(i);
                    if (v instanceof Number) {
                        sum += ((Number) v).doubleValue();
                        count++;
                    }
                }
                return count == 0 ? null : sum / count;
            };
        }

        public static WindowFunction count(String column) {
            return (df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx) -> {
                int colIdx = findColumnIndex(df, column);
                if (colIdx < 0) return end - start;
                int count = 0;
                for (int i = start; i < end; i++) {
                    if (df.column(colIdx).get(i) != null) count++;
                }
                return count;
            };
        }

        public static WindowFunction min(String column) {
            return (df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx) -> {
                int colIdx = findColumnIndex(df, column);
                if (colIdx < 0) return null;
                Double min = null;
                for (int i = start; i < end; i++) {
                    Object v = df.column(colIdx).get(i);
                    if (v instanceof Number) {
                        double d = ((Number) v).doubleValue();
                        if (min == null || d < min) min = d;
                    }
                }
                return min;
            };
        }

        public static WindowFunction max(String column) {
            return (df, row, start, end, hasPartition, hasOrder, partIdx, orderIdx) -> {
                int colIdx = findColumnIndex(df, column);
                if (colIdx < 0) return null;
                Double max = null;
                for (int i = start; i < end; i++) {
                    Object v = df.column(colIdx).get(i);
                    if (v instanceof Number) {
                        double d = ((Number) v).doubleValue();
                        if (max == null || d > max) max = d;
                    }
                }
                return max;
            };
        }

        // ---- helpers ----

        private static List<List<Object>> partitionIndices(DataFrame df, List<Integer> indices, int row) {
            if (indices.isEmpty()) return Collections.emptyList();
            List<List<Object>> result = new ArrayList<>();
            for (int idx : indices) {
                List<Object> vals = new ArrayList<>();
                vals.add(df.column(idx).get(row));
                result.add(vals);
            }
            return result;
        }

        private static int findGroupStart(DataFrame df, List<Integer> partIdx, List<Integer> orderIdx, int row) {
            if (partIdx.isEmpty()) return 0;
            List<Object> currentKey = new ArrayList<>();
            for (int idx : partIdx) currentKey.add(df.column(idx).get(row));

            for (int i = row - 1; i >= 0; i--) {
                boolean same = true;
                for (int j = 0; j < partIdx.size(); j++) {
                    Object v = df.column(partIdx.get(j)).get(i);
                    if (!java.util.Objects.equals(v, currentKey.get(j))) {
                        same = false;
                        break;
                    }
                }
                if (!same) return i + 1;
            }
            return 0;
        }

        private static int findGroupStart(List<List<Object>> partitionKeys, int row) {
            for (int i = row - 1; i >= 0; i--) {
                boolean same = true;
                for (List<Object> keys : partitionKeys) {
                    if (!java.util.Objects.equals(keys.get(0), keys.get(i))) {
                        same = false;
                        break;
                    }
                }
                if (!same) return i + 1;
            }
            return 0;
        }

        private static int findGroupEnd(DataFrame df, List<Integer> partIdx, int row) {
            if (partIdx.isEmpty()) return df.rowCount();
            List<Object> currentKey = new ArrayList<>();
            for (int idx : partIdx) currentKey.add(df.column(idx).get(row));

            for (int i = row + 1; i < df.rowCount(); i++) {
                boolean same = true;
                for (int j = 0; j < partIdx.size(); j++) {
                    Object v = df.column(partIdx.get(j)).get(i);
                    if (!java.util.Objects.equals(v, currentKey.get(j))) {
                        same = false;
                        break;
                    }
                }
                if (!same) return i;
            }
            return df.rowCount();
        }

        private static int findGroupSize(DataFrame df, List<Integer> partIdx, int row) {
            int start = findGroupStart(df, partIdx, null, row);
            int end = findGroupEnd(df, partIdx, row);
            return end - start;
        }

        private static Object getOrderValue(DataFrame df, List<Integer> orderIdx, int row) {
            if (orderIdx.isEmpty()) return row;
            return df.column(orderIdx.get(0)).get(row);
        }

        @SuppressWarnings("unchecked")
        private static int compare(Object a, Object b) {
            if (a == null && b == null) return 0;
            if (a == null) return -1;
            if (b == null) return 1;
            if (a instanceof Number && b instanceof Number) {
                return Double.compare(((Number) a).doubleValue(), ((Number) b).doubleValue());
            }
            return ((Comparable) a).compareTo(b);
        }

        private static int findColumnIndex(DataFrame df, String column) {
            for (int i = 0; i < df.columnCount(); i++) {
                if (df.column(i).name().equals(column)) return i;
            }
            return -1;
        }
    }
}
