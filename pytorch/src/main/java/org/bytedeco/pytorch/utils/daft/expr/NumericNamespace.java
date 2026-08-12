/*
 * Numeric / url / list namespaces.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * Numeric namespace — element-wise math operations on numeric columns.
 */
public final class NumericNamespace {

    private final Expression inner;
    public NumericNamespace(Expression inner) { this.inner = Objects.requireNonNull(inner); }

    public Expression abs()   { return new NumFn(inner, "abs",   v -> Math.abs(num(v))); }
    public Expression sqrt()  { return new NumFn(inner, "sqrt",  v -> Math.sqrt(num(v))); }
    public Expression log()   { return new NumFn(inner, "log",   v -> Math.log(num(v))); }
    public Expression exp()   { return new NumFn(inner, "exp",   v -> Math.exp(num(v))); }
    public Expression ceil()  { return new NumFn(inner, "ceil",  v -> Math.ceil(num(v))); }
    public Expression floor() { return new NumFn(inner, "floor", v -> Math.floor(num(v))); }
    public Expression sin()   { return new NumFn(inner, "sin",   v -> Math.sin(num(v))); }
    public Expression cos()   { return new NumFn(inner, "cos",   v -> Math.cos(num(v))); }
    public Expression round() { return new NumFn(inner, "round", v -> (double) Math.round(num(v))); }

    public Expression negate() {
        return new NumFn(inner, "neg", v -> -num(v));
    }

    private static double num(Object v) {
        if (v == null) return 0.0;
        if (v instanceof Number) return ((Number) v).doubleValue();
        try { return Double.parseDouble(v.toString()); }
        catch (NumberFormatException e) { return 0.0; }
    }
}

final class NumFn extends Expression {
    private final Expression inner;
    private final String tag;
    private final UnaryOperator<Double> fn;

    NumFn(Expression inner, String tag, UnaryOperator<Double> fn) {
        this.inner = inner; this.tag = tag; this.fn = fn;
    }
    @Override public String name() { return inner.name() + "." + tag; }
    @Override public Column eval(DataFrame df) {
        Column c = inner.eval(df);
        if (c == null) return null;
        Column out = new Column(name(), Column.DType.FLOAT64);
        for (int i = 0; i < c.size(); i++) {
            Object v = c.get(i);
            if (v == null) { out.add(null); continue; }
            try { out.add(fn.apply(((Number) v).doubleValue())); }
            catch (RuntimeException e) { out.add(null); }
        }
        return out;
    }
}

/**
 * URL namespace — parse + download.
 *
 * <p>{@code .url.download()} returns a binary column with the URL content
 * bytes (Daft 0.2+). Multiplexed through {@code java.net.HttpURLConnection}.
 */