package org.bytedeco.pytorch.nn.options;

import ai.onnxruntime.OrtSession;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.nn.*;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import org.bytedeco.pytorch.*;
import ai.onnxruntime.OrtLoggingLevel;
import java.util.ArrayList;
import java.util.List;

/**
 * Options for ONNX Runtime session configuration.
 */
public class ONNXOptions {

    /**
     * Graph optimization level.
     */
    public enum GraphOptimizationLevel {
        NO_OPT(0),
        BASIC_OPT(1),
        EXTENDED_OPT(2),
        FULL_OPT(3),
        ALL_OPT(99);

        private final int level;

        GraphOptimizationLevel(int level) {
            this.level = level;
        }

        public OrtSession.SessionOptions.OptLevel toORTLevel() {
            switch (this) {
                case NO_OPT:      return OptLevel.NO_OPT;
                case BASIC_OPT:   return OptLevel.BASIC_OPT;
                case EXTENDED_OPT:return OptLevel.EXTENDED_OPT;
                case FULL_OPT:    return OptLevel.LAYOUT_OPT;
                case ALL_OPT:     return OptLevel.ALL_OPT;
                default:          return OptLevel.ALL_OPT;
            }
        }

        public static GraphOptimizationLevel fromORTLevel(int level) {
            switch (level) {
                case 0:  return NO_OPT;
                case 1:  return BASIC_OPT;
                case 2:  return EXTENDED_OPT;
                case 3:  return FULL_OPT;
                case 99: return ALL_OPT;
                default: return ALL_OPT;
            }
        }
    }

    private List<String> providers = new ArrayList<>();
    private int deviceId = 0;
    private long gpuMemLimit = -1;
    private int interOpNumThreads = 0;
    private int intraOpNumThreads = 0;
    private GraphOptimizationLevel graphOptimizationLevel = GraphOptimizationLevel.ALL_OPT;
    private boolean parallelExecution = true;
    private boolean enableTelemetry = false;

    public ONNXOptions() {}

    /**
     * Get execution providers.
     */
    public List<String> getProviders() {
        return providers;
    }

    /**
     * Set execution providers.
     */
    public ONNXOptions providers(List<String> providers) {
        this.providers = providers != null ? providers : new ArrayList<>();
        return this;
    }

    /**
     * Add execution provider.
     */
    public ONNXOptions addProvider(String provider) {
        if (provider != null) {
            this.providers.add(provider);
        }
        return this;
    }

    /**
     * Use CUDA provider.
     */
    public ONNXOptions useCUDA() {
        return addProvider("CUDAExecutionProvider");
    }

    /**
     * Use CPU provider only.
     */
    public ONNXOptions useCPU() {
        this.providers.clear();
        return addProvider("CPUExecutionProvider");
    }

    /**
     * Get device ID.
     */
    public int getDeviceId() {
        return deviceId;
    }

    /**
     * Set device ID for GPU providers.
     */
    public ONNXOptions deviceId(int deviceId) {
        this.deviceId = deviceId;
        return this;
    }

    /**
     * Get GPU memory limit.
     */
    public long getGpuMemLimit() {
        return gpuMemLimit;
    }

    /**
     * Set GPU memory limit in bytes.
     */
    public ONNXOptions gpuMemLimit(long bytes) {
        this.gpuMemLimit = bytes;
        return this;
    }

    /**
     * Get inter-op threads.
     */
    public int getInterOpNumThreads() {
        return interOpNumThreads;
    }

    /**
     * Set inter-op threads.
     */
    public ONNXOptions interOpNumThreads(int threads) {
        this.interOpNumThreads = threads;
        return this;
    }

    /**
     * Get intra-op threads.
     */
    public int getIntraOpNumThreads() {
        return intraOpNumThreads;
    }

    /**
     * Set intra-op threads.
     */
    public ONNXOptions intraOpNumThreads(int threads) {
        this.intraOpNumThreads = threads;
        return this;
    }

    /**
     * Get graph optimization level.
     */
    public GraphOptimizationLevel getGraphOptimizationLevel() {
        return graphOptimizationLevel;
    }

    /**
     * Set graph optimization level.
     */
    public ONNXOptions graphOptimizationLevel(GraphOptimizationLevel level) {
        this.graphOptimizationLevel = level;
        return this;
    }

    /**
     * Enable parallel execution.
     */
    public boolean isParallelExecution() {
        return parallelExecution;
    }

    /**
     * Set parallel execution mode.
     */
    public ONNXOptions parallelExecution(boolean parallel) {
        this.parallelExecution = parallel;
        return this;
    }

    /**
     * Enable telemetry.
     */
    public boolean isEnableTelemetry() {
        return enableTelemetry;
    }

    /**
     * Set telemetry.
     */
    public ONNXOptions enableTelemetry(boolean enable) {
        this.enableTelemetry = enable;
        return this;
    }

    /**
     * Create options optimized for inference.
     */
    public static ONNXOptions inference() {
        return new ONNXOptions()
            .useCUDA()
            .graphOptimizationLevel(GraphOptimizationLevel.ALL_OPT)
            .parallelExecution(true);
    }

    /**
     * Create options for CPU inference.
     */
    public static ONNXOptions cpu() {
        return new ONNXOptions()
            .useCPU()
            .graphOptimizationLevel(GraphOptimizationLevel.EXTENDED_OPT)
            .parallelExecution(false);
    }

    /**
     * Create options for quick testing.
     */
    public static ONNXOptions testing() {
        return new ONNXOptions()
            .useCPU()
            .graphOptimizationLevel(GraphOptimizationLevel.NO_OPT)
            .parallelExecution(false)
            .enableTelemetry(false);
    }
}
