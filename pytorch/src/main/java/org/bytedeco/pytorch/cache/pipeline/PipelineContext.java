/*
 * PipelineContext -- per-run state that flows through a pipeline DAG.
 *
 * <p>Stages read and write typed attributes; attributes are recorded in a
 * {@link PipelineReport} so the run is audit-ready.
 *
 * <p>Threading: the context is intended to be used by a single stage at a
 * time; mutability is the responsibility of the scheduler.
 */
package org.bytedeco.pytorch.cache.pipeline;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

public final class PipelineContext {

    private final String pipelineName;
    private final long startedAtMs;
    private final Map<String, Object> attrs = new LinkedHashMap<>();
    private final Map<String, Long> counters = new LinkedHashMap<>();
    private final Map<String, String> errors = new LinkedHashMap<>();

    public PipelineContext(String pipelineName) {
        this.pipelineName = pipelineName;
        this.startedAtMs = System.currentTimeMillis();
    }

    public String name() { return pipelineName; }
    public long startedAtMs() { return startedAtMs; }

    public PipelineContext set(String key, Object value) { attrs.put(key, value); return this; }
    public <T> Optional<T> get(String key, Class<T> type) {
        Object v = attrs.get(key);
        if (v == null || !type.isInstance(v)) return Optional.empty();
        return Optional.of(type.cast(v));
    }
    public Object get(String key) { return attrs.get(key); }
    public Map<String, Object> attributes() { return attrs; }

    public PipelineContext increment(String counter) { increment(counter, 1); return this; }
    public PipelineContext increment(String counter, long delta) {
        counters.merge(counter, delta, Long::sum);
        return this;
    }

    public PipelineContext recordError(String stage, String message) {
        errors.put(stage, message);
        return this;
    }

    public Map<String, Long> counters() { return counters; }
    public Map<String, String> errors() { return errors; }
    public boolean hasErrors() { return !errors.isEmpty(); }

    public PipelineContext copy() {
        PipelineContext c = new PipelineContext(pipelineName);
        c.attrs.putAll(this.attrs);
        c.counters.putAll(this.counters);
        c.errors.putAll(this.errors);
        return c;
    }
}
