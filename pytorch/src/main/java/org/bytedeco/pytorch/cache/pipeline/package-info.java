/*
 * Pipeline -- declarative, multi-stage cache data pipelines.
 *
 * <p>This package implements the four pipeline archetypes used in modern
 * production cache fleets:
 * <ol>
 *   <li>{@link org.bytedeco.pytorch.cache.pipeline.RefreshPipeline} --
 *       orchestrates stale-while-revalidate refresh at scale (cron + per-key)</li>
 *   <li>{@link org.bytedeco.pytorch.cache.pipeline.ReplicationPipeline} --
 *       master → replica replication across regions (region pinning, fallback)</li>
 *   <li>{@link org.bytedeco.pytorch.cache.pipeline.BackfillPipeline} --
 *       rebuild from a primary source (cold-start, migration)</li>
 *   <li>{@link org.bytedeco.pytorch.cache.pipeline.MetricsPipeline} --
 *       export per-tier metrics to an external sink</li>
 * </ol>
 *
 * <p>A pipeline is a DAG of {@link PipelineStage} nodes executed on a
 * {@link PipelineScheduler}. Stages are pure functions over a
 * {@link PipelineContext}; the scheduler owns threading, retries, and
 * error routing.
 *
 * <p>The model is deliberately inspired by Bitnami Venafi / Netflix Hollow /
 * Google TensorFlow Extended pipelines -- stages are stateless, the context
 * carries mutable state, and every pipeline emits a {@link PipelineReport}.
 */
package org.bytedeco.pytorch.cache.pipeline;
