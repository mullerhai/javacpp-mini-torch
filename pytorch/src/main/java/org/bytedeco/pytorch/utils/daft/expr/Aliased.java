/*
 * Aliased - rename wrapper around another expression.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

public final class Aliased extends Expression {
    private final Expression inner;
    private final String alias;
    public Aliased(Expression inner, String alias) {
        this.inner = inner; this.alias = alias;
    }
    @Override public String name() { return alias; }
    @Override public Column eval(DataFrame df) {
        Column c = inner.eval(df);
        return c == null ? null : c.rename(alias);
    }
}
