/*
 * Daft 列表达式 — Python {@code df["col"]}, {@code col("x").alias("y")} 等价物.
 *
 * 设计: 每个表达式都是不可变的, 可链式. 求值由 engine 在 DataFrame 上做向量化.
 *
 * Daft 的表达式 API 风格: df.select(col("a"), col("b").alias("c"), lit(1))
 *   所以表达式主要返回 Column (作为字段引用 / 计算结果), engine 负责把它们
 *   装回 DataFrame.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/**
 * 列表达式 (algebra tree).
 *
 * <p>Each subclass implements {@link #eval(DataFrame)} returning a
 * {@link Column} (or null for star / literal — engine handles those).
 * Expressions are immutable and thread-safe.
 */
public abstract class Expression {

    public abstract String name();

    public abstract Column eval(DataFrame df);

    public Expression alias(String newName) {
        Objects.requireNonNull(newName, "alias");
        return new Aliased(this, newName);
    }

    public Expression cast(DataType type) {
        return new Cast(this, type);
    }

    /**
     * Cast to a {@link Column.DType} (engine-level type) by wrapping in a Cast expression.
     */
    public Expression cast(Column.DType type) {
        Objects.requireNonNull(type, "type");
        return new Cast(this, DataType.fromColumn(type));
    }

    public Expression fillNull(Object value) {
        return new FillNull(this, value);
    }

    public Expression isNull() {
        return new IsNull(this);
    }

    /** Alias of {@link #isNull()}. */
    public Expression isNotNull() {
        return new IsNotNull(this);
    }

    // ---- Comparison operators ----

    public Expression eq(Object other) {
        return new BinaryExpression(this, BinaryExpression.Op.EQ, lit(other), null);
    }

    public Expression ne(Object other) {
        return new BinaryExpression(this, BinaryExpression.Op.NEQ, lit(other), null);
    }

    public Expression lt(Object other) {
        return new BinaryExpression(this, BinaryExpression.Op.LT, lit(other), null);
    }

    public Expression le(Object other) {
        return new BinaryExpression(this, BinaryExpression.Op.LTE, lit(other), null);
    }

    public Expression gt(Object other) {
        return new BinaryExpression(this, BinaryExpression.Op.GT, lit(other), null);
    }

    public Expression ge(Object other) {
        return new BinaryExpression(this, BinaryExpression.Op.GTE, lit(other), null);
    }

    // ---- Arithmetic operators ----

    public Expression add(Object other) {
        return new BinaryExpression(this, BinaryExpression.Op.ADD, lit(other), null);
    }

    public Expression sub(Object other) {
        return new BinaryExpression(this, BinaryExpression.Op.SUB, lit(other), null);
    }

    public Expression mul(Object other) {
        return new BinaryExpression(this, BinaryExpression.Op.MUL, lit(other), null);
    }

    public Expression div(Object other) {
        return new BinaryExpression(this, BinaryExpression.Op.DIV, lit(other), null);
    }

    public Expression mod(Object other) {
        return new BinaryExpression(this, BinaryExpression.Op.MOD, lit(other), null);
    }

    /** String namespace: {@code col("x").str.upper()}. */
    public StringNamespace str() { return new StringNamespace(this); }

    /** Datetime namespace: {@code col("ts").dt.day()}. */
    public DateTimeNamespace dt() { return new DateTimeNamespace(this); }

    /** Numeric namespace: {@code col("x").abs()}. */
    public NumericNamespace numeric() { return new NumericNamespace(this); }

    /** URL namespace: {@code col("url").url.download()}. */
    public UrlNamespace url() { return new UrlNamespace(this); }

    /** Image namespace: {@code col("img").image.decode()}. */
    public ImageNamespace image() { return new ImageNamespace(this); }

    /** Audio namespace: {@code col("aud").audio.decode()}. */
    public AudioNamespace audio() { return new AudioNamespace(this); }

    /** Video namespace: {@code col("vid").video.decode()}. */
    public VideoNamespace video() { return new VideoNamespace(this); }

    /** Embedding namespace: {@code col("txt").embedding.encode_text()}. */
    public EmbeddingNamespace embedding() { return new EmbeddingNamespace(this); }

    /** Reference to a column by name. */
    public static Expression col(String name) {
        return new ColumnRef(name);
    }

    /** Literal value (boxed). */
    public static Expression lit(Object value) {
        return new Literal(value);
    }

    /** Star wildcard for select(). */
    public static List<Expression> star() {
        List<Expression> l = new ArrayList<>();
        l.add(new Star());
        return l;
    }
}

/** String operations (Daft: .str.lengths / .str.upper / .str.contains / ...). */