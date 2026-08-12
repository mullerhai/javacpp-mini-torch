/*
 * IsNull - boolean column indicating null positions.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

public final class IsNull extends Expression {
    private final Expression inner;
    public IsNull(Expression inner) { this.inner = inner; }
    @Override public String name() { return inner.name() + "_is_null"; }
    @Override public Column eval(DataFrame df) {
        Column c = inner.eval(df);
        if (c == null) return null;
        Column out = new Column(name(), Column.DType.BOOLEAN);
        for (int i = 0; i < c.size(); i++) {
            out.add(c.get(i) == null);
        }
        return out;
    }
}
