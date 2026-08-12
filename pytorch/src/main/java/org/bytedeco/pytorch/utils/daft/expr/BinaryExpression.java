/*
 * BinaryExpression - left OP right arithmetic/comparison.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.function.BinaryOperator;

public final class BinaryExpression extends Expression {
    public enum Op { ADD, SUB, MUL, DIV, MOD, EQ, NEQ, LT, LTE, GT, GTE, AND, OR }
    private final Expression left;
    private final Expression right;
    private final Op op;
    private final String alias;
    public BinaryExpression(Expression left, Op op, Expression right, String alias) {
        this.left = left; this.op = op; this.right = right; this.alias = alias;
    }
    @Override public String name() { return alias != null ? alias : (left.name() + "_" + op + "_" + right.name()); }
    @Override public Column eval(DataFrame df) {
        Column lc = left.eval(df);
        Column rc = right.eval(df);
        if (lc == null || rc == null) return null;
        int n = Math.min(lc.size(), rc.size());
        Column out = new Column(name(), resultType(op, lc.dtype(), rc.dtype()));
        for (int i = 0; i < n; i++) {
            out.add(applyOp(op, lc.get(i), rc.get(i)));
        }
        return out;
    }

    private static Column.DType resultType(Op op, Column.DType a, Column.DType b) {
        switch (op) {
            case EQ: case NEQ: case LT: case LTE: case GT: case GTE:
            case AND: case OR:
                return Column.DType.BOOLEAN;
            default:
                if (a == Column.DType.FLOAT64 || b == Column.DType.FLOAT64) return Column.DType.FLOAT64;
                if (a == Column.DType.FLOAT32 || b == Column.DType.FLOAT32) return Column.DType.FLOAT32;
                if (a == Column.DType.INT64 || b == Column.DType.INT64) return Column.DType.INT64;
                return Column.DType.INT32;
        }
    }

    private static Object applyOp(Op op, Object a, Object b) {
        if (a == null || b == null) return null;
        try {
            switch (op) {
                case ADD: return numericOp((x, y) -> x + y, a, b);
                case SUB: return numericOp((x, y) -> x - y, a, b);
                case MUL: return numericOp((x, y) -> x * y, a, b);
                case DIV: return numericOp((x, y) -> x / y, a, b);
                case MOD: return numericOp((x, y) -> x % y, a, b);
                case EQ:  return a.equals(b);
                case NEQ: return !a.equals(b);
                case LT:  return compareNumeric(a, b) < 0;
                case LTE: return compareNumeric(a, b) <= 0;
                case GT:  return compareNumeric(a, b) > 0;
                case GTE: return compareNumeric(a, b) >= 0;
                case AND: return Boolean.TRUE.equals(a) && Boolean.TRUE.equals(b);
                case OR:  return Boolean.TRUE.equals(a) || Boolean.TRUE.equals(b);
                default: throw new IllegalStateException("unknown op: " + op);
            }
        } catch (ClassCastException e) {
            return null;
        }
    }

    private static Object numericOp(BinaryOperator<Double> f, Object a, Object b) {
        if (!(a instanceof Number) || !(b instanceof Number)) return null;
        return f.apply(((Number) a).doubleValue(), ((Number) b).doubleValue());
    }

    private static int compareNumeric(Object a, Object b) {
        if (!(a instanceof Number) || !(b instanceof Number)) return 0;
        return Double.compare(((Number) a).doubleValue(), ((Number) b).doubleValue());
    }
}
