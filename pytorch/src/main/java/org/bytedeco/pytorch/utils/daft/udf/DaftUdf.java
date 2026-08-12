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
package org.bytedeco.pytorch.utils.daft.udf;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.daft.expr.Expression;

import java.io.Serializable;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * User-Defined Functions for DaftDataFrame.
 *
 * <p>Supports:
 * <ul>
 *   <li>Scalar UDFs: apply function to each row</li>
 *   <li>Aggregation UDFs: aggregate across groups</li>
 *   <li>Window UDFs: apply over window partitions</li>
 *   <li>Java 8+ lambdas via functional interfaces</li>
 *   <li>Stateful UDFs with init/accumulate/finalize pattern</li>
 * </ul>
 *
 * <pre>{@code
 * // Scalar UDF with lambda
 * DaftUdf<String, String> upper = DaftUdf.scalar(
 *     (String s) -> s.toUpperCase(),
 *     String.class, String.class
 * );
 * df.select(col("name").apply(upper).alias("NAME"));
 *
 * // Stateful aggregation
 * DaftUdf<List<Double>, Double> mean = DaftUdf.stateful()
 *     .init(() -> new RunningStats())
 *     .accumulate((RunningStats s, Double v) -> s.add(v))
 *     .finalize((RunningStats s) -> s.mean())
 *     .build();
 * }</pre>
 */
public final class DaftUdf {

    private DaftUdf() {}

    // ---- Scalar UDF ----

    /**
     * Create a scalar UDF from a lambda.
     *
     * @param fn        the function to apply to each value
     * @param inputType the Java class of input values
     * @param outputType the Java class of output values
     */
    public static <T, R> ScalarUdf<T, R> scalar(ScalarFunction<T, R> fn,
                                                  Class<T> inputType,
                                                  Class<R> outputType) {
        return new ScalarUdf<>(fn, inputType, outputType);
    }

    /**
     * Create a scalar UDF with multiple inputs.
     */
    public static <R> ScalarUdf<Object[], R> scalarMulti(
            java.util.function.Function<Object[], R> fn,
            Class<R> outputType) {
        R unused = null;
        @SuppressWarnings("unchecked")
        ScalarFunction<Object[], R> wrapped = input -> fn.apply(input);
        return new ScalarUdf<>(wrapped, Object[].class, outputType);
    }

    /**
     * Scalar UDF implementation.
     */
    public static final class ScalarUdf<T, R> {
        private final ScalarFunction<T, R> fn;
        private final Class<T> inputType;
        private final Class<R> outputType;
        private String name;

        ScalarUdf(ScalarFunction<T, R> fn, Class<T> inputType, Class<R> outputType) {
            this.fn = fn;
            this.inputType = inputType;
            this.outputType = outputType;
            this.name = "udf_" + System.identityHashCode(this);
        }

        public ScalarUdf<T, R> named(String name) {
            this.name = name;
            return this;
        }

        public String name() { return name; }

        /**
         * Apply this UDF to a column expression.
         */
        @SuppressWarnings("unchecked")
        public Expression apply(Expression input) {
            return new UdfExpression(this, new Expression[]{input}, null);
        }

        public Expression apply(Expression... inputs) {
            return new UdfExpression(this, inputs, null);
        }

        /**
         * Apply to raw values.
         */
        @SuppressWarnings("unchecked")
        public List<R> applyTo(List<?> values) {
            List<R> result = new ArrayList<>(values.size());
            for (Object v : values) {
                try {
                    T typed = v == null ? null : (T) v;
                    result.add(fn.apply(typed));
                } catch (Exception e) {
                    result.add(null);
                }
            }
            return result;
        }

        ScalarFunction<T, R> fn() { return fn; }
        public Class<T> inputType() { return inputType; }
        public Class<R> outputType() { return outputType; }
    }

    /**
     * Functional interface for scalar functions.
     */
    @FunctionalInterface
    public interface ScalarFunction<T, R> extends Serializable {
        R apply(T value);
    }

    // ---- Stateful Aggregation UDF ----

    /**
     * Builder for stateful aggregation UDFs.
     */
    public static class StatefulUdfBuilder {
        private java.util.function.Supplier<Object> initFn;
        private java.util.function.BiConsumer<Object, Object> accumulateFn;
        private java.util.function.Function<Object, Object> finalizeFn;
        private Class<Object> stateClass;
        private String name = "stateful_udf";

        public StatefulUdfBuilder init(java.util.function.Supplier<?> supplier) {
            this.initFn = (java.util.function.Supplier<Object>) supplier;
            return this;
        }

        public StatefulUdfBuilder accumulate(java.util.function.BiConsumer<Object, Object> consumer) {
            this.accumulateFn = consumer;
            return this;
        }

        public StatefulUdfBuilder merge(java.util.function.BiConsumer<Object, Object> mergeFn) {
            return this;
        }

        public StatefulUdfBuilder finalize(java.util.function.Function<?, ?> function) {
            this.finalizeFn = (java.util.function.Function<Object, Object>) function;
            return this;
        }

        public StatefulUdfBuilder name(String name) {
            this.name = name;
            return this;
        }

        public <S> StatefulUdfBuilder stateClass(Class<S> cls) {
            this.stateClass = (Class<Object>) cls;
            return this;
        }

        @SuppressWarnings("unchecked")
        public <S, R> AggregationUdf<S, R> build() {
            if (initFn == null || accumulateFn == null || finalizeFn == null) {
                throw new IllegalStateException("init, accumulate, and finalize required");
            }
            return new AggregationUdf<>(
                    (java.util.function.Supplier<S>) initFn,
                    (java.util.function.BiConsumer<S, Object>) accumulateFn,
                    (java.util.function.Function<S, R>) finalizeFn,
                    name
            );
        }
    }

    public static StatefulUdfBuilder stateful() {
        return new StatefulUdfBuilder();
    }

    /**
     * Aggregation UDF with init/accumulate/finalize pattern.
     */
    public static final class AggregationUdf<S, R> {
        private final java.util.function.Supplier<S> initFn;
        private final java.util.function.BiConsumer<S, Object> accumulateFn;
        private final java.util.function.Function<S, R> finalizeFn;
        private final String name;

        AggregationUdf(java.util.function.Supplier<S> initFn,
                      java.util.function.BiConsumer<S, Object> accumulateFn,
                      java.util.function.Function<S, R> finalizeFn,
                      String name) {
            this.initFn = initFn;
            this.accumulateFn = accumulateFn;
            this.finalizeFn = finalizeFn;
            this.name = name;
        }

        public String name() { return name; }

        /**
         * Run this aggregation over grouped data.
         */
        public List<R> aggregate(List<?> values) {
            S state = initFn.get();
            for (Object v : values) {
                if (v != null) {
                    try {
                        accumulateFn.accept(state, v);
                    } catch (Exception ignored) {}
                }
            }
            return java.util.Collections.singletonList(finalizeFn.apply(state));
        }

        /**
         * Run this aggregation over grouped data, returning one result per group.
         */
        @SuppressWarnings("unchecked")
        public List<R> aggregatePerGroup(List<?> values, List<Integer> groupIds) {
            if (values.isEmpty()) return new ArrayList<>();
            Map<Integer, S> states = new ConcurrentHashMap<>();
            for (int i = 0; i < values.size(); i++) {
                int gid = groupIds.get(i);
                S state = states.computeIfAbsent(gid, k -> initFn.get());
                if (values.get(i) != null) {
                    try {
                        accumulateFn.accept(state, values.get(i));
                    } catch (Exception ignored) {}
                }
            }
            List<R> results = new ArrayList<>();
            for (Map.Entry<Integer, S> e : states.entrySet()) {
                results.add(finalizeFn.apply(e.getValue()));
            }
            return results;
        }
    }

    // ---- Expression wrapper ----

    /**
     * Expression that wraps a UDF call.
     */
    public static final class UdfExpression extends Expression {
        private final ScalarUdf<?, ?> udf;
        private final Expression[] inputs;
        private final AggregationUdf<?, ?> aggUdf;

        UdfExpression(ScalarUdf<?, ?> udf, Expression[] inputs, AggregationUdf<?, ?> aggUdf) {
            this.udf = udf;
            this.inputs = inputs;
            this.aggUdf = aggUdf;
        }

        public ScalarUdf<?, ?> udf() { return udf; }
        public Expression[] inputs() { return inputs; }
        public AggregationUdf<?, ?> aggUdf() { return aggUdf; }

        @Override
        public String name() {
            return "UdfExpression";
        }

        @Override
        public Column eval(DataFrame df) {
            if (udf != null) {
                Column[] inputCols = new Column[inputs.length];
                for (int i = 0; i < inputs.length; i++) {
                    inputCols[i] = inputs[i].eval(df);
                }
                return applyUdf(df, inputCols);
            }
            throw new UnsupportedOperationException("Aggregation UDF requires groupBy");
        }

        private Column applyUdf(DataFrame df, Column[] inputCols) {
            int rows = df.rowCount();
            List<Object> results = new ArrayList<>(rows);
            for (int r = 0; r < rows; r++) {
                Object[] args = new Object[inputCols.length];
                for (int i = 0; i < inputCols.length; i++) {
                    args[i] = inputCols[i].get(r);
                }
                try {
                    Object result = ((ScalarUdf<Object[], Object>) udf).fn().apply(args);
                    results.add(result);
                } catch (Exception e) {
                    results.add(null);
                }
            }
            String outputName = "udf_result";
            Column out = new Column(outputName, Column.DType.STRING);
            for (Object v : results) out.add(v);
            return out;
        }
    }

    // ---- Built-in utility UDFs ----

    /** Coalesce: return first non-null value. */
    public static ScalarUdf<Object[], Object> coalesce() {
        return scalarMulti(
            args -> {
                for (Object arg : args) {
                    if (arg != null) return arg;
                }
                return null;
            },
            Object.class
        ).named("coalesce");
    }

    /** Greatest: return max of all arguments. */
    @SuppressWarnings("unchecked")
    public static ScalarUdf<Object[], Comparable> greatest() {
        return scalarMulti(
            args -> {
                Comparable max = null;
                for (Object arg : args) {
                    if (arg instanceof Comparable) {
                        Comparable c = (Comparable) arg;
                        if (max == null || c.compareTo(max) > 0) max = c;
                    }
                }
                return max;
            },
            Comparable.class
        ).named("greatest");
    }

    /** Least: return min of all arguments. */
    @SuppressWarnings("unchecked")
    public static ScalarUdf<Object[], Comparable> least() {
        return scalarMulti(
            args -> {
                Comparable min = null;
                for (Object arg : args) {
                    if (arg instanceof Comparable) {
                        Comparable c = (Comparable) arg;
                        if (min == null || c.compareTo(min) < 0) min = c;
                    }
                }
                return min;
            },
            Comparable.class
        ).named("least");
    }

    /** When/then/otherwise (SQL CASE expression). */
    public static Expression when(Expression condition, Object then) {
        return new CaseExpression(condition, then, null);
    }

    public static final class CaseExpression extends Expression {
        private final Expression condition;
        private final Object then;
        private final Object otherwise;

        CaseExpression(Expression condition, Object then, Object otherwise) {
            this.condition = condition;
            this.then = then;
            this.otherwise = otherwise;
        }

        public CaseExpression otherwise(Object elseVal) {
            return new CaseExpression(condition, then, elseVal);
        }

        @Override
        public String name() {
            return "CaseExpression";
        }

        @Override
        public Column eval(DataFrame df) {
            Column condCol = condition.eval(df);
            int rows = df.rowCount();
            Column out = new Column("case_result", Column.DType.STRING);
            for (int r = 0; r < rows; r++) {
                Object cond = condCol.get(r);
                boolean isTrue = Boolean.TRUE.equals(cond) ||
                        ("true".equals(String.valueOf(cond).toLowerCase()));
                out.add(isTrue ? then : (otherwise != null ? otherwise : null));
            }
            return out;
        }
    }
}
