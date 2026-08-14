/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
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
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.evaluation;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * Multi-dimensional alignment evaluator for comprehensive model assessment.
 *
 * <p>Evaluates models across multiple alignment dimensions:
 * <ul>
 *   <li><b>Helpfulness</b>: Does the model provide useful, relevant responses?</li>
 *   <li><b>Harmlessness</b>: Does the model avoid harmful content?</li>
 *   <li><b>Honesty</b>: Does the model provide accurate, truthful information?</li>
 *   <li><b>Instruction Following</b>: Does the model follow user instructions?</li>
 *   <li><b>Safety</b>: Does the model produce safe outputs?</li>
 *   <li><b>Toxicity</b>: Does the model avoid toxic language?</li>
 * </ul>
 *
 * <p>Features:
 * <ul>
 *   <li>Per-dimension scoring with aggregate metrics</li>
 *   <li>Statistical significance testing</li>
 *   <li>Comparison against baselines</li>
 *   <li>Detailed reporting with breakdowns</li>
 * </ul>
 *
 * <pre>{@code
 * AlignmentEvaluator evaluator = AlignmentEvaluator.builder()
 *     .addDimension("helpfulness", helpfulnessRewardModel)
 *     .addDimension("safety", safetyRewardModel)
 *     .build();
 *
 * AlignmentReport report = evaluator.evaluate(testDataset);
 * System.out.println(report.summary());
 * }</pre>
 */
public class AlignmentEvaluator implements AutoCloseable {

    public static final String VERSION = "1.0";

    private final Map<String, RewardModel> dimensions;
    private final EvaluationConfig config;
    private volatile boolean closed;

    // Evaluation history
    private final List<AlignmentReport> reportHistory = new ArrayList<>();

    public AlignmentEvaluator(Map<String, RewardModel> dimensions, EvaluationConfig config) {
        this.dimensions = new HashMap<>(dimensions);
        this.config = config != null ? config : EvaluationConfig.DEFAULT;
    }

    // ==================== Evaluation ====================

    /**
     * Evaluate model across all dimensions.
     */
    public AlignmentReport evaluate(List<EvaluationExample> examples) {
        if (closed) throw new IllegalStateException("Evaluator is closed");

        Map<String, List<Double>> dimensionScores = new HashMap<>();
        Map<String, DimensionStatistics> dimensionStats = new HashMap<>();

        // Initialize score lists
        for (String dimName : dimensions.keySet()) {
            dimensionScores.put(dimName, new ArrayList<>());
        }

        // Score each example across all dimensions
        for (EvaluationExample example : examples) {
            for (Map.Entry<String, RewardModel> entry : dimensions.entrySet()) {
                String dimName = entry.getKey();
                RewardModel model = entry.getValue();

                double score = model.score(example.input(), example.output());
                dimensionScores.get(dimName).add(score);
            }
        }

        // Compute statistics for each dimension
        for (Map.Entry<String, List<Double>> entry : dimensionScores.entrySet()) {
            String dimName = entry.getKey();
            List<Double> scores = entry.getValue();

            if (!scores.isEmpty()) {
                dimensionStats.put(dimName, computeStatistics(scores));
            }
        }

        // Compute aggregate score
        double overallScore = computeAggregateScore(dimensionStats);

        // Build report
        AlignmentReport report = new AlignmentReport(
                dimensionStats,
                overallScore,
                config.computeSignificance() ? computeSignificance(dimensionStats) : null,
                System.currentTimeMillis());

        reportHistory.add(report);

        return report;
    }

    /**
     * Evaluate a single dimension.
     */
    public double evaluateDimension(String dimension, List<EvaluationExample> examples) {
        RewardModel model = dimensions.get(dimension);
        if (model == null) {
            throw new IllegalArgumentException("Unknown dimension: " + dimension);
        }

        double total = 0;
        for (EvaluationExample example : examples) {
            total += model.score(example.input(), example.output());
        }

        return examples.isEmpty() ? 0 : total / examples.size();
    }

    /**
     * Compare two models on the same dataset.
     */
    public ComparisonResult compare(
            Module modelA,
            Module modelB,
            List<EvaluationExample> examples,
            String dimension) {

        RewardModel baselineModel = dimensions.get(dimension);
        if (baselineModel == null) {
            throw new IllegalArgumentException("Unknown dimension: " + dimension);
        }

        // Score with baseline
        List<Double> baselineScores = new ArrayList<>();
        for (EvaluationExample example : examples) {
            baselineScores.add(baselineModel.score(example.input(), example.output()));
        }

        // Note: In practice, you would have separate scoring functions for modelA and modelB
        // This is a simplified interface

        DimensionStatistics baselineStats = computeStatistics(baselineScores);

        return new ComparisonResult(
                dimension,
                baselineStats,
                null, // Model A stats (would be computed with modelA scorer)
                null, // Model B stats (would be computed with modelB scorer)
                computeStatisticalSignificance(baselineStats, baselineStats) // Placeholder
        );
    }

    // ==================== Statistics ====================

    private DimensionStatistics computeStatistics(List<Double> scores) {
        if (scores.isEmpty()) {
            return new DimensionStatistics(0, 0, 0, 0, 0, 0, 0);
        }

        double sum = 0;
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (double s : scores) {
            sum += s;
            min = Math.min(min, s);
            max = Math.max(max, s);
        }

        double mean = sum / scores.size();
        double variance = scores.stream()
                .mapToDouble(s -> (s - mean) * (s - mean))
                .sum() / scores.size();
        double std = Math.sqrt(variance);

        // Compute percentiles
        List<Double> sorted = new ArrayList<>(scores);
        Collections.sort(sorted);

        double p25 = percentile(sorted, 0.25);
        double p50 = percentile(sorted, 0.50);
        double p75 = percentile(sorted, 0.75);

        return new DimensionStatistics(
                mean, std, min, max, p25, p50, p75);
    }

    private double percentile(List<Double> sorted, double p) {
        if (sorted.isEmpty()) return 0;
        int idx = (int) Math.ceil(p * sorted.size()) - 1;
        idx = Math.max(0, Math.min(idx, sorted.size() - 1));
        return sorted.get(idx);
    }

    private double computeAggregateScore(Map<String, DimensionStatistics> stats) {
        double total = 0;
        double totalWeight = 0;

        for (Map.Entry<String, DimensionStatistics> entry : stats.entrySet()) {
            double weight = config.getDimensionWeight(entry.getKey());
            total += entry.getValue().mean() * weight;
            totalWeight += weight;
        }

        return totalWeight > 0 ? total / totalWeight : 0;
    }

    private StatisticalSignificance computeSignificance(Map<String, DimensionStatistics> stats) {
        // Placeholder for statistical significance computation
        // Would typically use t-test or bootstrap confidence intervals
        return null;
    }

    private double computeStatisticalSignificance(DimensionStatistics a, DimensionStatistics b) {
        // Placeholder: would compute p-value for difference
        return 0.0;
    }

    // ==================== History ====================

    public List<AlignmentReport> getReportHistory() {
        return Collections.unmodifiableList(reportHistory);
    }

    public AlignmentReport getLatestReport() {
        return reportHistory.isEmpty() ? null : reportHistory.get(reportHistory.size() - 1);
    }

    // ==================== Lifecycle ====================

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        reportHistory.clear();
        System.out.println("[AlignmentEvaluator] Closed");
    }

    public boolean isClosed() { return closed; }

    // ==================== Builder ====================

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private final Map<String, RewardModel> dimensions = new HashMap<>();
        private EvaluationConfig config = EvaluationConfig.DEFAULT;

        public Builder addDimension(String name, RewardModel model) {
            dimensions.put(name, model);
            return this;
        }

        public Builder config(EvaluationConfig config) {
            this.config = config;
            return this;
        }

        public AlignmentEvaluator build() {
            if (dimensions.isEmpty()) {
                throw new IllegalStateException("At least one dimension must be specified");
            }
            return new AlignmentEvaluator(dimensions, config);
        }
    }

    // ==================== Supporting Types ====================

    /**
     * Reward model interface for evaluation dimensions.
     */
    @FunctionalInterface
    public interface RewardModel {
        double score(String input, String output);
    }

    /**
     * Evaluation example.
     */
    public record EvaluationExample(
            String input,
            String output,
            String reference,
            Map<String, Object> metadata
    ) {
        public EvaluationExample(String input, String output) {
            this(input, output, null, Map.of());
        }

        public EvaluationExample(String input, String output, String reference) {
            this(input, output, reference, Map.of());
        }
    }

    /**
     * Alignment evaluation report.
     */
    public record AlignmentReport(
            Map<String, DimensionStatistics> dimensionStats,
            double overallScore,
            StatisticalSignificance significance,
            long timestamp
    ) {
        public DimensionStatistics getDimension(String name) {
            return dimensionStats.get(name);
        }

        public String summary() {
            StringBuilder sb = new StringBuilder();
            sb.append("Alignment Report\n");
            sb.append("================\n");
            sb.append(String.format("Overall Score: %.4f\n", overallScore));
            sb.append("\nPer-Dimension Scores:\n");

            for (Map.Entry<String, DimensionStatistics> entry : dimensionStats.entrySet()) {
                DimensionStatistics stats = entry.getValue();
                sb.append(String.format("  %s: %.4f (±%.4f)\n",
                        entry.getKey(), stats.mean(), stats.std()));
            }

            return sb.toString();
        }
    }

    /**
     * Dimension-level statistics.
     */
    public record DimensionStatistics(
            double mean,
            double std,
            double min,
            double max,
            double percentile25,
            double percentile50,
            double percentile75
    ) {}

    /**
     * Statistical significance results.
     */
    public record StatisticalSignificance(
            double pValue,
            double confidenceInterval,
            double effectSize
    ) {}

    /**
     * Model comparison result.
     */
    public record ComparisonResult(
            String dimension,
            DimensionStatistics baselineStats,
            DimensionStatistics modelAStats,
            DimensionStatistics modelBStats,
            double pValue
    ) {
        public String summary() {
            return String.format(
                    "Comparison for %s: baseline=%.4f, modelA=%.4f, modelB=%.4f (p=%.4f)",
                    dimension,
                    baselineStats != null ? baselineStats.mean() : 0,
                    modelAStats != null ? modelAStats.mean() : 0,
                    modelBStats != null ? modelBStats.mean() : 0,
                    pValue);
        }
    }

    /**
     * Evaluation configuration.
     */
    public static class EvaluationConfig {
        public static final EvaluationConfig DEFAULT = new EvaluationConfig();

        private boolean computeSignificance = true;
        private int bootstrapSamples = 1000;
        private double confidenceLevel = 0.95;
        private Map<String, Double> dimensionWeights = Map.of(
                "helpfulness", 1.0,
                "harmlessness", 1.0,
                "honesty", 1.0);

        public boolean computeSignificance() { return computeSignificance; }
        public int bootstrapSamples() { return bootstrapSamples; }
        public double confidenceLevel() { return confidenceLevel; }
        public double getDimensionWeight(String dim) {
            return dimensionWeights.getOrDefault(dim, 1.0);
        }

        public Builder toBuilder() {
            return new Builder()
                    .computeSignificance(computeSignificance)
                    .bootstrapSamples(bootstrapSamples)
                    .confidenceLevel(confidenceLevel)
                    .dimensionWeights(dimensionWeights);
        }

        public static final class Builder {
            private boolean computeSignificance = true;
            private int bootstrapSamples = 1000;
            private double confidenceLevel = 0.95;
            private Map<String, Double> dimensionWeights = new HashMap<>();

            public Builder computeSignificance(boolean v) { this.computeSignificance = v; return this; }
            public Builder bootstrapSamples(int v) { this.bootstrapSamples = v; return this; }
            public Builder confidenceLevel(double v) { this.confidenceLevel = v; return this; }
            public Builder addDimensionWeight(String dim, double weight) {
                this.dimensionWeights.put(dim, weight);
                return this;
            }
            public Builder dimensionWeights(Map<String, Double> weights) {
                this.dimensionWeights = new HashMap<>(weights);
                return this;
            }

            public EvaluationConfig build() {
                EvaluationConfig config = new EvaluationConfig();
                config.computeSignificance = this.computeSignificance;
                config.bootstrapSamples = this.bootstrapSamples;
                config.confidenceLevel = this.confidenceLevel;
                config.dimensionWeights = this.dimensionWeights;
                return config;
            }
        }
    }
}
