/*
 * ColumnRef - references a column by name.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.Objects;

public final class ColumnRef extends Expression {
    private final String name;
    public ColumnRef(String name) { this.name = name; }
    @Override public String name() { return name; }
    @Override public Column eval(DataFrame df) {
        Objects.requireNonNull(df, "df");
        return df.column(name);
    }
}
