/*
 * Cast - cast wrapper. Casts an expression's column to a target DataType.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

public final class Cast extends Expression {
    private final Expression inner;
    private final DataType target;
    public Cast(Expression inner, DataType target) {
        this.inner = inner; this.target = target;
    }
    @Override public String name() { return inner.name() + "::" + target; }
    @Override public Column eval(DataFrame df) {
        Column c = inner.eval(df);
        if (c == null) return null;
        Column out = new Column(c.name(), Column.DType.STRING); // placeholder; real type by target
        for (int i = 0; i < c.size(); i++) {
            out.add(castValue(c.get(i), target));
        }
        return out.rename(name());
    }

    private static Object castValue(Object v, DataType t) {
        if (v == null) return null;
        switch (t.kind) {
            case INT8:  return ((Number) v).byteValue();
            case INT16: return ((Number) v).shortValue();
            case INT32: return ((Number) v).intValue();
            case INT64: return ((Number) v).longValue();
            case FLOAT32: return ((Number) v).floatValue();
            case FLOAT64: return ((Number) v).doubleValue();
            case BOOL:  return v instanceof Boolean ? v : Boolean.parseBoolean(v.toString());
            case STRING: return v.toString();
            default: return v;
        }
    }
}
