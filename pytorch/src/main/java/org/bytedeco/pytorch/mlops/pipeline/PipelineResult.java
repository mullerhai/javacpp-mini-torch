/*
 * Pipeline result types used by {@link PipelineExecutor}.
 *
 * <p>Forwarded from {org.bytedeco.pytorch.dataframe.feature.pipeline.Pipeline}
 * steps when they expose {@code StepResult}-bearing result objects.
 */
package org.bytedeco.pytorch.mlops.pipeline;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** Top-level result of executing a pipeline. */
public final class PipelineResult {
    private final Map<String, StepResult> stepResults;
    private final boolean success;
    private final Throwable error;

    public PipelineResult(Map<String, StepResult> stepResults, boolean success, Throwable error) {
        this.stepResults = stepResults == null
                ? Collections.emptyMap()
                : Collections.unmodifiableMap(new LinkedHashMap<>(stepResults));
        this.success = success;
        this.error = error;
    }

    public Map<String, StepResult> stepResults() { return stepResults; }
    public boolean success() { return success; }
    public Throwable error() { return error; }

    /** Per-step result. */
    public static final class StepResult {
        private final String name;
        private final boolean success;
        private final long durationMs;
        private final Map<String, Object> output;
        private final Throwable error;

        public StepResult(String name, boolean success, long durationMs,
                          Map<String, Object> output, Throwable error) {
            this.name = name;
            this.success = success;
            this.durationMs = durationMs;
            this.output = output == null
                    ? Collections.emptyMap()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(output));
            this.error = error;
        }

        public String name() { return name; }
        public boolean success() { return success; }
        public long durationMs() { return durationMs; }
        public Map<String, Object> output() { return output; }
        public Throwable error() { return error; }
    }
}
