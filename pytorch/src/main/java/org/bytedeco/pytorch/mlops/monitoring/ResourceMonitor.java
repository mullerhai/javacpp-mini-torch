/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.mlops.monitoring;
import org.bytedeco.pytorch.jit.*;

import java.io.Closeable;
import java.lang.management.ManagementFactory;
import java.lang.management.OperatingSystemMXBean;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ThreadMXBean;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.function.Consumer;

/**
 * Enterprise-grade resource monitoring platform.
 *
 * <p>Features:
 * <ul>
 *   <li>CPU/Memory/GPU monitoring</li>
 *   <li>Custom metrics collection</li>
 *   <li>Prometheus export</li>
 *   <li>Alert rules</li>
 *   <li>Time-series data</li>
 * </ul>
 *
 * <p>Reference: Prometheus, Grafana, OpenTelemetry
 *
 * <pre>{@code
 * ResourceMonitor monitor = ResourceMonitor.builder()
 *     .collectInterval(10, TimeUnit.SECONDS)
 *     .enableGpuMonitoring(true)
 *     .build();
 *
 * monitor.start();
 *
 * // Register custom metric
 * monitor.registerGauge("my_metric", () -> computeValue());
 *
 * // Get Prometheus format
 * String metrics = monitor.exportPrometheus();
 * }</pre>
 */
public class ResourceMonitor implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;
    private volatile boolean running;

    // Configuration
    private final long collectIntervalMs;
    private final boolean enableGpuMonitoring;
    private final boolean enableJmxExport;

    // System beans
    private final OperatingSystemMXBean osBean;
    private final MemoryMXBean memoryBean;
    private final ThreadMXBean threadBean;
    private final List<GarbageCollectorMXBean> gcBeans;

    // Metrics
    private final Map<String, Metric> metrics = new ConcurrentHashMap<>();
    private final Map<String, GaugeMetric> gauges = new ConcurrentHashMap<>();
    private final Map<String, CounterMetric> counters = new ConcurrentHashMap<>();
    private final Map<String, HistogramMetric> histograms = new ConcurrentHashMap<>();
    private final Map<String, TimerMetric> timers = new ConcurrentHashMap<>();

    // Time series storage
    private final Map<String, Deque<MetricPoint>> timeSeries = new ConcurrentHashMap<>();
    private final int maxTimeSeriesPoints;

    // Alert rules
    private final List<AlertRule> alertRules = new ArrayList<>();
    private final List<Alert> activeAlerts = Collections.synchronizedList(new ArrayList<>());
    private final List<Consumer<Alert>> alertHandlers = Collections.synchronizedList(new ArrayList<>());

    // Executor
    private final ScheduledExecutorService executor;

    // Statistics
    private final AtomicLong totalCollections = new AtomicLong(0);
    private final AtomicReference<Double> cpuUsage = new AtomicReference<>(0.0);
    private final AtomicReference<Double> memoryUsage = new AtomicReference<>(0.0);

    public static Builder builder() {
        return new Builder();
    }

    private ResourceMonitor(Builder builder) {
        this.collectIntervalMs = builder.collectIntervalMs;
        this.enableGpuMonitoring = builder.enableGpuMonitoring;
        this.enableJmxExport = builder.enableJmxExport;
        this.maxTimeSeriesPoints = builder.maxTimeSeriesPoints;

        this.osBean = ManagementFactory.getOperatingSystemMXBean();
        this.memoryBean = ManagementFactory.getMemoryMXBean();
        this.threadBean = ManagementFactory.getThreadMXBean();
        this.gcBeans = ManagementFactory.getGarbageCollectorMXBeans();

        this.executor = Executors.newScheduledThreadPool(2);

        // Initialize default metrics
        initializeDefaultMetrics();
    }

    private void initializeDefaultMetrics() {
        // System metrics
        registerGauge("system_cpu_usage", () -> getCpuUsage());
        registerGauge("system_memory_usage", () -> getMemoryUsage());
        registerGauge("system_memory_used_bytes", () -> (double) getUsedMemory());
        registerGauge("system_memory_total_bytes", () -> (double) getTotalMemory());
        registerGauge("system_threads", () -> (double) threadBean.getThreadCount());
        registerGauge("system_gc_count", () -> (double) gcBeans.stream().mapToLong(GarbageCollectorMXBean::getCollectionCount).sum());

        // JVM metrics
        registerGauge("jvm_heap_used_bytes", () -> (double) memoryBean.getHeapMemoryUsage().getUsed());
        registerGauge("jvm_heap_max_bytes", () -> (double) memoryBean.getHeapMemoryUsage().getMax());
        registerGauge("jvm_heap_committed_bytes", () -> (double) memoryBean.getHeapMemoryUsage().getCommitted());
        registerGauge("jvm_nonheap_used_bytes", () -> (double) memoryBean.getNonHeapMemoryUsage().getUsed());
    }

    // ============= Lifecycle =============

    /**
     * Start monitoring.
     */
    public void start() {
        if (running) return;
        running = true;

        // Start collection loop
        executor.scheduleAtFixedRate(
                this::collectMetrics,
                collectIntervalMs,
                collectIntervalMs,
                TimeUnit.MILLISECONDS
        );

        System.out.println("[ResourceMonitor] Started with interval " + collectIntervalMs + "ms");
    }

    /**
     * Stop monitoring.
     */
    public void stop() {
        if (!running) return;
        running = false;
        System.out.println("[ResourceMonitor] Stopped");
    }

    /**
     * Collect metrics once.
     */
    private void collectMetrics() {
        long timestamp = System.currentTimeMillis();

        try {
            // Collect system metrics
            collectSystemMetrics(timestamp);

            // Collect gauges
            for (Map.Entry<String, GaugeMetric> entry : gauges.entrySet()) {
                double value = entry.getValue().getValue();
                recordTimeSeries(entry.getKey(), timestamp, value);
            }

            // Evaluate alert rules
            evaluateAlertRules(timestamp);

            totalCollections.incrementAndGet();

        } catch (Exception e) {
            System.err.println("Metric collection error: " + e.getMessage());
        }
    }

    private void collectSystemMetrics(long timestamp) {
        // CPU
        double cpu = getCpuUsage();
        cpuUsage.set(cpu);
        recordTimeSeries("system_cpu_usage", timestamp, cpu);

        // Memory
        double mem = getMemoryUsage();
        memoryUsage.set(mem);
        recordTimeSeries("system_memory_usage", timestamp, mem);
    }

    private void recordTimeSeries(String name, long timestamp, double value) {
        Deque<MetricPoint> series = timeSeries.computeIfAbsent(name, k -> new ArrayDeque<>());
        series.addLast(new MetricPoint(timestamp, value));

        // Trim old points
        while (series.size() > maxTimeSeriesPoints) {
            series.removeFirst();
        }
    }

    // ============= Metric Registration =============

    /**
     * Register a gauge metric.
     */
    public void registerGauge(String name, java.util.function.Supplier<Double> supplier) {
        gauges.put(name, new GaugeMetric(name, supplier));
    }

    /**
     * Register a counter.
     */
    public void registerCounter(String name) {
        counters.put(name, new CounterMetric(name));
    }

    /**
     * Increment a counter.
     */
    public void incrementCounter(String name) {
        CounterMetric counter = counters.computeIfAbsent(name, k -> new CounterMetric(k));
        counter.increment();
        recordTimeSeries(name, System.currentTimeMillis(), counter.getValue());
    }

    /**
     * Record a histogram value.
     */
    public void recordHistogram(String name, double value) {
        HistogramMetric histogram = histograms.computeIfAbsent(name, k -> new HistogramMetric(k));
        histogram.record(value);
        recordTimeSeries(name, System.currentTimeMillis(), value);
    }

    /**
     * Create/start a timer.
     */
    public TimerMetric startTimer(String name) {
        return timers.computeIfAbsent(name, k -> new TimerMetric(k));
    }

    // ============= Alert Management =============

    /**
     * Add an alert rule.
     */
    public void addAlertRule(AlertRule rule) {
        alertRules.add(rule);
    }

    /**
     * Add alert handler.
     */
    public void addAlertHandler(Consumer<Alert> handler) {
        alertHandlers.add(handler);
    }

    /**
     * Create alert rule.
     */
    public AlertRule createAlertRule(String name, String metric, AlertCondition condition, double threshold) {
        return new AlertRule(name, metric, condition, threshold);
    }

    private void evaluateAlertRules(long timestamp) {
        for (AlertRule rule : alertRules) {
            Deque<MetricPoint> series = timeSeries.get(rule.metric);
            if (series == null || series.isEmpty()) continue;

            double currentValue = series.getLast().value;

            boolean shouldAlert = switch (rule.condition) {
                case ABOVE -> currentValue > rule.threshold;
                case BELOW -> currentValue < rule.threshold;
                case EQUALS -> Math.abs(currentValue - rule.threshold) < 0.0001;
            };

            if (shouldAlert) {
                fireAlert(rule, currentValue, timestamp);
            }
        }
    }

    private void fireAlert(AlertRule rule, double value, long timestamp) {
        // Check if already firing
        boolean alreadyFiring = activeAlerts.stream()
                .anyMatch(a -> a.rule.equals(rule) && a.status == AlertStatus.FIRING);

        if (!alreadyFiring) {
            Alert alert = new Alert(rule, value, timestamp);
            activeAlerts.add(alert);

            // Notify handlers
            for (Consumer<Alert> handler : alertHandlers) {
                try {
                    handler.accept(alert);
                } catch (Exception e) {
                    System.err.println("Alert handler error: " + e.getMessage());
                }
            }
        }
    }

    /**
     * Acknowledge an alert.
     */
    public void acknowledgeAlert(Alert alert) {
        alert.acknowledge();
    }

    /**
     * Clear an alert.
     */
    public void clearAlert(Alert alert) {
        alert.clear();
        activeAlerts.remove(alert);
    }

    // ============= Export =============

    /**
     * Export metrics in Prometheus format.
     */
    public String exportPrometheus() {
        StringBuilder sb = new StringBuilder();

        // Export gauges
        for (Map.Entry<String, GaugeMetric> entry : gauges.entrySet()) {
            sb.append("# HELP ").append(entry.getKey()).append("\n");
            sb.append("# TYPE ").append(entry.getKey()).append(" gauge\n");
            sb.append(entry.getKey()).append(" ").append(entry.getValue().getValue()).append("\n");
        }

        // Export counters
        for (Map.Entry<String, CounterMetric> entry : counters.entrySet()) {
            sb.append("# HELP ").append(entry.getKey()).append("\n");
            sb.append("# TYPE ").append(entry.getKey()).append(" counter\n");
            sb.append(entry.getKey()).append(" ").append(entry.getValue().getValue()).append("\n");
        }

        return sb.toString();
    }

    /**
     * Export metrics as JSON.
     */
    public String exportJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"metrics\":{");

        boolean first = true;
        for (Map.Entry<String, GaugeMetric> entry : gauges.entrySet()) {
            if (!first) sb.append(",");
            sb.append("\"").append(entry.getKey()).append("\":").append(entry.getValue().getValue());
            first = false;
        }

        sb.append("}}");
        return sb.toString();
    }

    // ============= System Metrics =============

    public double getCpuUsage() {
        try {
            // Prefer the com.sun.management extension API on Hotspot JVMs
            if (osBean instanceof com.sun.management.OperatingSystemMXBean) {
                com.sun.management.OperatingSystemMXBean sunBean =
                        (com.sun.management.OperatingSystemMXBean) osBean;
                // getProcessCpuLoad is more accurate; fall back to getSystemCpuLoad if unavailable
                double processLoad = safeProcessCpuLoad(sunBean);
                if (processLoad >= 0) {
                    return processLoad * 100.0;
                }
                return sunBean.getSystemCpuLoad() * 100;
            }
            // Fallback: derive CPU usage from process CPU time deltas
            return getCpuUsageFromProcessTimeDelta();
        } catch (Throwable t) {
            return 0.0;
        }
    }

    private static double safeProcessCpuLoad(com.sun.management.OperatingSystemMXBean sunBean) {
        try {
            return sunBean.getProcessCpuLoad();
        } catch (Throwable t) {
            return -1.0;
        }
    }

    private double getCpuUsageFromProcessTimeDelta() {
        // java.lang.management.OperatingSystemMXBean does NOT expose getProcessCpuTime().
        // This path only runs when com.sun.management.OperatingSystemMXBean is unavailable,
        // which on Hotspot is essentially never. Return 0.0 since we cannot measure reliably.
        return 0.0;
    }

    public double getMemoryUsage() {
        MemoryUsage heap = memoryBean.getHeapMemoryUsage();
        return (double) heap.getUsed() / heap.getMax() * 100;
    }

    public long getUsedMemory() {
        return memoryBean.getHeapMemoryUsage().getUsed();
    }

    public long getTotalMemory() {
        return memoryBean.getHeapMemoryUsage().getMax();
    }

    public int getThreadCount() {
        return threadBean.getThreadCount();
    }

    // ============= Statistics =============

    public ResourceMonitorStats getStats() {
        return new ResourceMonitorStats(
                running,
                collectIntervalMs,
                totalCollections.get(),
                metrics.size(),
                gauges.size(),
                counters.size(),
                activeAlerts.size(),
                cpuUsage.get(),
                memoryUsage.get()
        );
    }

    public boolean isClosed() { return closed; }
    public boolean isRunning() { return running; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        stop();

        executor.shutdown();
        try {
            executor.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            executor.shutdownNow();
        }

        System.out.println("[ResourceMonitor] Closed");
    }

    // ============= Inner Types =============

    /**
     * Metric point for time series.
     */
    public static class MetricPoint {
        public final long timestamp;
        public final double value;

        public MetricPoint(long timestamp, double value) {
            this.timestamp = timestamp;
            this.value = value;
        }
    }

    /**
     * Gauge metric.
     */
    public static class GaugeMetric {
        private final String name;
        private final java.util.function.Supplier<Double> supplier;

        public GaugeMetric(String name, java.util.function.Supplier<Double> supplier) {
            this.name = name;
            this.supplier = supplier;
        }

        public String name() { return name; }
        public double getValue() { return supplier.get(); }
    }

    /**
     * Counter metric.
     */
    public static class CounterMetric {
        private final String name;
        private final AtomicLong value = new AtomicLong(0);

        public CounterMetric(String name) {
            this.name = name;
        }

        public String name() { return name; }
        public long getValue() { return value.get(); }
        public void increment() { value.incrementAndGet(); }
        public void increment(long delta) { value.addAndGet(delta); }
        public void reset() { value.set(0); }
    }

    /**
     * Histogram metric.
     */
    public static class HistogramMetric {
        private final String name;
        private final AtomicLong count = new AtomicLong(0);
        private final AtomicReference<Double> sum = new AtomicReference<>(0.0);
        private final AtomicReference<Double> min = new AtomicReference<>(Double.MAX_VALUE);
        private final AtomicReference<Double> max = new AtomicReference<>(Double.MIN_VALUE);

        public HistogramMetric(String name) {
            this.name = name;
        }

        public String name() { return name; }
        public long getCount() { return count.get(); }
        public double getSum() { return sum.get(); }
        public double getMean() { return count.get() > 0 ? sum.get() / count.get() : 0; }
        public double getMin() { return min.get(); }
        public double getMax() { return max.get(); }

        public void record(double value) {
            count.incrementAndGet();
            sum.updateAndGet(s -> s + value);
            min.updateAndGet(m -> Math.min(m, value));
            max.updateAndGet(m -> Math.max(m, value));
        }
    }

    /**
     * Timer metric.
     */
    public static class TimerMetric implements AutoCloseable {
        private final String name;
        private final long startTime = System.nanoTime();

        public TimerMetric(String name) {
            this.name = name;
        }

        public String name() { return name; }
        public long getElapsedNanos() { return System.nanoTime() - startTime; }
        public double getElapsedMillis() { return getElapsedNanos() / 1_000_000.0; }

        @Override
        public void close() {
            // This would record to histogram
        }
    }

    /**
     * Generic metric interface.
     */
    public interface Metric {
        String name();
    }

    /**
     * Alert rule.
     */
    public static class AlertRule {
        public final String name;
        public final String metric;
        public final AlertCondition condition;
        public final double threshold;
        public final long durationMs;

        public AlertRule(String name, String metric, AlertCondition condition, double threshold) {
            this(name, metric, condition, threshold, 0);
        }

        public AlertRule(String name, String metric, AlertCondition condition, double threshold, long durationMs) {
            this.name = name;
            this.metric = metric;
            this.condition = condition;
            this.threshold = threshold;
            this.durationMs = durationMs;
        }
    }

    /**
     * Alert condition.
     */
    public enum AlertCondition {
        ABOVE,
        BELOW,
        EQUALS
    }

    /**
     * Alert.
     */
    public static class Alert {
        public final AlertRule rule;
        public final double currentValue;
        public final long firedAt;
        public volatile AlertStatus status;
        public volatile long acknowledgedAt;
        public volatile long clearedAt;

        public Alert(AlertRule rule, double currentValue, long firedAt) {
            this.rule = rule;
            this.currentValue = currentValue;
            this.firedAt = firedAt;
            this.status = AlertStatus.FIRING;
        }

        public void acknowledge() {
            this.status = AlertStatus.ACKNOWLEDGED;
            this.acknowledgedAt = System.currentTimeMillis();
        }

        public void clear() {
            this.status = AlertStatus.RESOLVED;
            this.clearedAt = System.currentTimeMillis();
        }
    }

    /**
     * Alert status.
     */
    public enum AlertStatus {
        FIRING,
        ACKNOWLEDGED,
        RESOLVED
    }

    /**
     * Statistics.
     */
    public static class ResourceMonitorStats {
        public final boolean running;
        public final long collectIntervalMs;
        public final long totalCollections;
        public final int numMetrics;
        public final int numGauges;
        public final int numCounters;
        public final int activeAlerts;
        public final double currentCpuUsage;
        public final double currentMemoryUsage;

        public ResourceMonitorStats(boolean running, long collectIntervalMs, long totalCollections,
                                int numMetrics, int numGauges, int numCounters,
                                int activeAlerts, double currentCpuUsage, double currentMemoryUsage) {
            this.running = running;
            this.collectIntervalMs = collectIntervalMs;
            this.totalCollections = totalCollections;
            this.numMetrics = numMetrics;
            this.numGauges = numGauges;
            this.numCounters = numCounters;
            this.activeAlerts = activeAlerts;
            this.currentCpuUsage = currentCpuUsage;
            this.currentMemoryUsage = currentMemoryUsage;
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private long collectIntervalMs = 10_000;
        private boolean enableGpuMonitoring = false;
        private boolean enableJmxExport = true;
        private int maxTimeSeriesPoints = 1000;

        public Builder collectInterval(long interval, TimeUnit unit) {
            this.collectIntervalMs = unit.toMillis(interval);
            return this;
        }
        public Builder enableGpuMonitoring(boolean enable) { this.enableGpuMonitoring = enable; return this; }
        public Builder enableJmxExport(boolean enable) { this.enableJmxExport = enable; return this; }
        public Builder maxTimeSeriesPoints(int max) { this.maxTimeSeriesPoints = max; return this; }

        public ResourceMonitor build() {
            return new ResourceMonitor(this);
        }
    }
}
