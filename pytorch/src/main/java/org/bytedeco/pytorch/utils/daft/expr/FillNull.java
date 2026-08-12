/*
 * FillNull - fill nulls with a constant value.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

public final class FillNull extends Expression {
    private final Expression inner;
    private final Object value;
    public FillNull(Expression inner, Object value) { this.inner = inner; this.value = value; }
    @Override public String name() { return inner.name(); }
    @Override public Column eval(DataFrame df) {
        Column c = inner.eval(df);
        if (c == null) return null;
        Column out = new Column(c.name(), c.dtype());
        for (int i = 0; i < c.size(); i++) {
            Object v = c.get(i);
            out.add(v == null ? value : v);
        }
        return out;
    }
}
