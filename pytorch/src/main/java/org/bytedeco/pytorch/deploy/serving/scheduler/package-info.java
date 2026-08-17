/*
 * Scheduler module for model serving.
 *
 * This module provides enterprise-grade scheduling capabilities for model inference:
 *   - ModelScheduler: Priority-based request queuing with adaptive batching
 *   - Warmup strategies for GPU JIT compilation
 *   - Backpressure handling for high-load scenarios
 *   - Multi-model serving coordination
 *
 * Usage:
 * <pre>{@code
 * ModelScheduler scheduler = ModelScheduler.builder().build();
 *
 * // Register a model
 * scheduler.registerModel(ModelDefinition.builder("recommender")
 *     .version("v1", ModelVersion.builder("v1")
 *         .artifactPath("/models/rec_v1.pt")
 *         .build())
 *     .build());
 *
 * // Submit requests
 * InferenceRequest req = InferenceRequest.builder("req-1", "recommender")
 *     .priority(Priority.P1_INTERACTIVE)
 *     .input(inputTensor)
 *     .build();
 *
 * scheduler.submit(req).thenAccept(result -> {
 *     // Handle result
 * });
 * }</pre>
 */
package org.bytedeco.pytorch.deploy.serving.scheduler;
