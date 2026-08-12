/*
 * Literal - constant value (boxed).
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

public final class Literal extends Expression {
    private final Object value;
    public Literal(Object value) { this.value = value; }
    @Override public String name() { return "lit(" + value + ")"; }
    @Override public Column eval(DataFrame df) { return null; }
    public Object value() { return value; }
}
