/*
 * String / datetime / url / numeric expression extensions.
 *
 * Mirrors Daft's fluent API:
 *   col("name").str.lengths()
 *   col("name").str.upper()
 *   col("ts").dt.day()
 *   col("url").url.download()
 *   col("score").mean()    (scalar over groupby)
 *   col("x").abs() / .sqrt() / .log()
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Extension namespace: string operations on a column expression.
 *
 * <pre>{@code
 *   Expression e = col("name").str.lengths();
 *   Expression e2 = col("url").url.download();
 * }</pre>
 */
public final class StringNamespace {

    private final Expression inner;

    public StringNamespace(Expression inner) {
        this.inner = Objects.requireNonNull(inner, "inner");
    }

    public Expression lengths() {
        return new StrFn(inner, "lengths", s -> String.valueOf((long) s.length()));
    }

    public Expression lower() {
        return new StrFn(inner, "lower", s -> s.toLowerCase(Locale.ROOT));
    }

    public Expression upper() {
        return new StrFn(inner, "upper", s -> s.toUpperCase(Locale.ROOT));
    }

    public Expression trim() {
        return new StrFn(inner, "trim", String::trim);
    }

    public Expression reverse() {
        return new StrFn(inner, "reverse", s -> new StringBuilder(s).reverse().toString());
    }

    public Expression contains(String sub) {
        Objects.requireNonNull(sub, "sub");
        return new BoolStrFn(inner, "contains", s -> s.contains(sub));
    }

    public Expression startsWith(String prefix) {
        Objects.requireNonNull(prefix, "prefix");
        return new BoolStrFn(inner, "startsWith", s -> s.startsWith(prefix));
    }

    public Expression endsWith(String suffix) {
        Objects.requireNonNull(suffix, "suffix");
        return new BoolStrFn(inner, "endsWith", s -> s.endsWith(suffix));
    }

    public Expression regexMatch(String regex) {
        Objects.requireNonNull(regex, "regex");
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(regex);
        return new BoolStrFn(inner, "regex", s -> p.matcher(s).find());
    }

    public Expression replace(String oldStr, String newStr) {
        Objects.requireNonNull(oldStr, "oldStr");
        Objects.requireNonNull(newStr, "newStr");
        return new StrFn(inner, "replace", s -> s.replace(oldStr, newStr));
    }
}

/** String-fn expression. */
class StrFn extends Expression {
    private final Expression inner;
    private final String tag;
    private final UnaryOperator<String> fn;

    StrFn(Expression inner, String tag, UnaryOperator<String> fn) {
        this.inner = inner; this.tag = tag; this.fn = fn;
    }
    @Override public String name() { return inner.name() + "." + tag; }
    @Override public Column eval(DataFrame df) {
        Column c = inner.eval(df);
        if (c == null) return null;
        Column out = new Column(name(), Column.DType.STRING);
        for (int i = 0; i < c.size(); i++) {
            Object v = c.get(i);
            if (v == null) { out.add(null); continue; }
            try {
                out.add(fn.apply(v.toString()));
            } catch (RuntimeException e) {
                out.add(null);
            }
        }
        return out;
    }
}

/** Boolean string-fn expression. */
class BoolStrFn extends Expression {
    private final Expression inner;
    private final String tag;
    private final java.util.function.Function<String, Boolean> fn;

    BoolStrFn(Expression inner, String tag, java.util.function.Function<String, Boolean> fn) {
        this.inner = inner; this.tag = tag; this.fn = fn;
    }
    @Override public String name() { return inner.name() + "." + tag; }
    @Override public Column eval(DataFrame df) {
        Column c = inner.eval(df);
        if (c == null) return null;
        Column out = new Column(name(), Column.DType.BOOLEAN);
        for (int i = 0; i < c.size(); i++) {
            Object v = c.get(i);
            if (v == null) { out.add(null); continue; }
            try {
                out.add(fn.apply(v.toString()));
            } catch (RuntimeException e) {
                out.add(null);
            }
        }
        return out;
    }
}
