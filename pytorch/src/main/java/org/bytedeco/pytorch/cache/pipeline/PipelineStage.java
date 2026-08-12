/*
 * PipelineStage -- one node in a pipeline DAG.
 *
 * <p>A stage is a pure function over a {@link PipelineContext}. It reads
 * inputs, mutates state, and reports its outcomes. Failures are flagged via
 * {@link PipelineContext#recordError(String, String)}; the scheduler decides
 * whether to retry, skip, or abort based on {@link StageResult}.
 */
package org.bytedeco.pytorch.cache.pipeline;

public interface PipelineStage {

    String name();

    StageResult apply(PipelineContext ctx) throws Exception;

    enum StageResult {
        CONTINUE,        // proceed to next stage
        SKIP,            // skip remaining stages (e.g. feature flag is off)
        RETRY,           // ask scheduler to retry this stage
        ABORT_FATAL      // abort the entire pipeline run
    }
}
