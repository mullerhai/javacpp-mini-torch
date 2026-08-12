/*
 * Star - wildcard expression that selects all columns.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

public final class Star extends Expression {
    @Override public String name() { return "*"; }
    @Override public Column eval(DataFrame df) { return null; }
}
