/*
 * MediaFn - generic media-function expression that wraps a unary transformation
 * from any data cell (bytes / DataValue / String) to the result cell.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.function.Function;

public final class MediaFn extends Expression {
    private final Object owner;
    private final String tag;
    private final Function<Object, Object> fn;

    MediaFn(Object owner, String tag, Function<Object, Object> fn) {
        this.owner = owner;
        this.tag = tag;
        this.fn = fn;
    }
    @Override public String name() { return tag; }
    @Override public Column eval(DataFrame df) {
        // MediaFn is invoked by the engine, which knows the source column.
        // We resolve the source column lazily via the owner expression field
        // if available. We expose a hook for the engine to inject.
        throw new UnsupportedOperationException("MediaFn.eval requires engine dispatch");
    }

    public Object apply(Object value) {
        return fn.apply(value);
    }
}
