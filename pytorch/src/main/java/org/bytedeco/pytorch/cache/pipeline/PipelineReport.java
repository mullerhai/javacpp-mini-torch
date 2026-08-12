/*
 * PipelineReport -- final output of a pipeline run.
 *
 * <p>Captures: total wall-clock time, per-stage duration, counter snapshot,
 * error list, and skipped/failed flags. Designed to be serialisable to JSON
 * for export to observability platforms.
 */
package org.bytedeco.pytorch.cache.pipeline;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class PipelineReport {

    public static final class StageStat {
        public final String name;
        public final long durationNanos;
        public final PipelineStage.StageResult result;
        public final String error;

        public StageStat(String name, long durationNanos, PipelineStage.StageResult result, String error) {
            this.name = name;
            this.durationNanos = durationNanos;
            this.result = result;
            this.error = error;
        }
    }

    private final String pipelineName;
    private final long startedAtMs;
    private final long endedAtMs;
    private final boolean success;
    private final boolean skipped;
    private final List<StageStat> stages = new ArrayList<>();
    private final Map<String, Long> counters = new LinkedHashMap<>();
    private final Map<String, String> errors = new LinkedHashMap<>();

    public PipelineReport(String pipelineName, long startedAtMs, long endedAtMs,
                          boolean success, boolean skipped) {
        this.pipelineName = pipelineName;
        this.startedAtMs = startedAtMs;
        this.endedAtMs = endedAtMs;
        this.success = success;
        this.skipped = skipped;
    }

    public void recordStage(String name, long durationNanos, PipelineStage.StageResult result, String error) {
        stages.add(new StageStat(name, durationNanos, result, error));
    }

    public void recordError(String stage, String error) {
        errors.put(stage, error);
    }

    public void setCounters(Map<String, Long> c) { counters.putAll(c); }
    public void setErrors(Map<String, String> e) { errors.putAll(e); }

    public String name() { return pipelineName; }
    public long startedAtMs() { return startedAtMs; }
    public long endedAtMs() { return endedAtMs; }
    public long durationMs() { return endedAtMs - startedAtMs; }
    public boolean success() { return success; }
    public boolean skipped() { return skipped; }
    public List<StageStat> stages() { return stages; }
    public Map<String, Long> counters() { return counters; }
    public Map<String, String> errors() { return errors; }

    @Override
    public String toString() {
        return "PipelineReport{name=" + pipelineName + ", success=" + success
                + ", skipped=" + skipped + ", durationMs=" + durationMs()
                + ", stages=" + stages.size() + ", errors=" + errors.size() + "}";
    }
}
