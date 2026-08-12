/*
 * DateTime / numeric / url / list namespaces on column expressions.
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

/**
 * Datetime namespace — extract components from a temporal column.
 *
 * <pre>{@code
 *   col("ts").dt.day();
 *   col("ts").dt.month();
 * }</pre>
 */
public final class DateTimeNamespace {

    private final Expression inner;
    public DateTimeNamespace(Expression inner) { this.inner = Objects.requireNonNull(inner, "inner"); }

    public Expression day()           { return new DtFn(inner, "day", this::dayOf); }
    public Expression month()         { return new DtFn(inner, "month", this::monthOf); }
    public Expression year()          { return new DtFn(inner, "year", this::yearOf); }
    public Expression hour()          { return new DtFn(inner, "hour", this::hourOf); }
    public Expression minute()        { return new DtFn(inner, "minute", this::minuteOf); }
    public Expression second()        { return new DtFn(inner, "second", this::secondOf); }
    public Expression dayOfWeek()     { return new DtFn(inner, "dayOfWeek", this::dayOfWeekOf); }
    public Expression epochSeconds()  { return new DtFn(inner, "epochSeconds", this::epochSecondsOf); }

    private LocalDateTime toLdt(Object v) {
        if (v instanceof LocalDateTime) return (LocalDateTime) v;
        if (v instanceof Instant) return LocalDateTime.ofInstant((Instant) v, ZoneId.systemDefault());
        if (v instanceof LocalDate) return ((LocalDate) v).atStartOfDay();
        if (v instanceof Number) return LocalDateTime.ofInstant(
                Instant.ofEpochMilli(((Number) v).longValue()), ZoneId.systemDefault());
        if (v instanceof String) {
            try { return LocalDateTime.parse((String) v); } catch (Exception ignored) {}
            try { return LocalDateTime.ofInstant(Instant.parse((String) v), ZoneId.systemDefault()); } catch (Exception ignored) {}
        }
        return null;
    }

    private int dayOf(Object v) { LocalDateTime t = toLdt(v); return t == null ? 0 : t.getDayOfMonth(); }
    private int monthOf(Object v) { LocalDateTime t = toLdt(v); return t == null ? 0 : t.getMonthValue(); }
    private int yearOf(Object v) { LocalDateTime t = toLdt(v); return t == null ? 0 : t.getYear(); }
    private int hourOf(Object v) { LocalDateTime t = toLdt(v); return t == null ? 0 : t.getHour(); }
    private int minuteOf(Object v) { LocalDateTime t = toLdt(v); return t == null ? 0 : t.getMinute(); }
    private int secondOf(Object v) { LocalDateTime t = toLdt(v); return t == null ? 0 : t.getSecond(); }
    private int dayOfWeekOf(Object v) {
        LocalDateTime t = toLdt(v);
        return t == null ? 0 : (t.getDayOfWeek().getValue());
    }
    private long epochSecondsOf(Object v) {
        LocalDateTime t = toLdt(v);
        return t == null ? 0L : t.atZone(ZoneId.systemDefault()).toEpochSecond();
    }
}

final class DtFn extends Expression {
    private final Expression inner;
    private final String tag;
    private final java.util.function.Function<Object, Object> fn;

    DtFn(Expression inner, String tag, java.util.function.Function<Object, Object> fn) {
        this.inner = inner; this.tag = tag; this.fn = fn;
    }
    @Override public String name() { return inner.name() + "." + tag; }
    @Override public Column eval(DataFrame df) {
        Column c = inner.eval(df);
        if (c == null) return null;
        Column out = new Column(name(), Column.DType.INT64);
        for (int i = 0; i < c.size(); i++) {
            Object v = c.get(i);
            if (v == null) { out.add(null); continue; }
            try { out.add(fn.apply(v)); } catch (RuntimeException e) { out.add(null); }
        }
        return out;
    }
}

/**
 * Numeric namespace — element-wise math.
 */
