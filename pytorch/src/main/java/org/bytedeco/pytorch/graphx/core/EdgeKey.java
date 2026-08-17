/*
 * GraphX: Enterprise-Grade NetworkX Implementation for Java
 *
 * Inspired by Python NetworkX (https://github.com/networkx/networkx)
 * Licensed under BSD 3-Clause (original NetworkX license).
 */
package org.bytedeco.pytorch.graphx.core;

import java.util.Objects;

/**
 * Immutable edge identifier for {@link org.bytedeco.pytorch.graphx.core.MultiGraph}.
 * For {@link org.bytedeco.pytorch.graphx.core.Graph}/{@link org.bytedeco.pytorch.graphx.core.DiGraph},
 * the {@code key} field is always 0 (single edge per pair).
 *
 * @param <N> node type
 */
public final class EdgeKey<N> {
    private final N u;
    private final N v;
    private final int key;

    public EdgeKey(N u, N v, int key) {
        this.u = Objects.requireNonNull(u, "u");
        this.v = Objects.requireNonNull(v, "v");
        this.key = key;
    }

    public static <N> EdgeKey<N> of(N u, N v) {
        return new EdgeKey<>(u, v, 0);
    }

    public static <N> EdgeKey<N> of(N u, N v, int key) {
        return new EdgeKey<>(u, v, key);
    }

    public N u() { return u; }
    public N v() { return v; }
    public int key() { return key; }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof EdgeKey<?>)) return false;
        EdgeKey<?> e = (EdgeKey<?>) o;
        return key == e.key && u.equals(e.u) && v.equals(e.v);
    }

    @Override
    public int hashCode() {
        return Objects.hash(u, v, key);
    }

    @Override
    public String toString() {
        return "(" + u + ", " + v + ", " + key + ")";
    }
}