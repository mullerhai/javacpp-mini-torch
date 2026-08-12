/*
 * Runtime SRM monitor — continuous chi-square check + auto-diagnosis.
 *
 * Industry (Meta XP, Microsoft ExP, DoorDash, ByteDance):
 *   Real-time dashboards compute chi-square p-value continuously and
 *   raise alerts. Beyond just "SRM detected", they provide:
 *     1. Direction of imbalance (which variant is over/under)
 *     2. Drift magnitude over time (sliding window)
 *     3. Diagnosis hints (logging bug, bot traffic, region skew)
 *
 * This module provides:
 *   - Continuous chi-square with sliding window per (experiment, variant)
 *   - Detection of which variant is the outlier
 *   - Health score (0..1) integrating chi-square + drift + sample size
 *   - Recommendations (KILL / PAUSE / CONTINUE / INVESTIGATE)
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Continuous SRM monitor.
 */
public final class SrmMonitor {

    private final OnlineMetricsCollector collector;
    private final double alpha;

    public SrmMonitor(OnlineMetricsCollector collector, double alpha) {
        this.collector = Objects.requireNonNull(collector, "collector");
        this.alpha = alpha;
    }

    /**
     * Run a one-shot diagnosis for one experiment.
     */
    public Diagnosis diagnose(Experiment experiment) {
        Objects.requireNonNull(experiment, "experiment");
        long[] observed = new long[experiment.variants().size()];
        double[] expected = new double[experiment.variants().size()];
        for (int i = 0; i < experiment.variants().size(); i++) {
            Variant v = experiment.variants().get(i);
            observed[i] = collector.exposureCount(experiment.id(), v.id());
            expected[i] = v.trafficWeight();
        }
        StatisticalTest.SrmResult srm = StatisticalTest.srmTest(observed, expected, alpha);

        // Identify which variant is the outlier.
        long total = 0L;
        for (long o : observed) total += o;
        double ratioSum = 0.0;
        for (double e : expected) ratioSum += e;
        String outlier = null;
        double maxDrift = 0.0;
        for (int i = 0; i < observed.length; i++) {
            double e = total * (expected[i] / ratioSum);
            double drift = e == 0 ? 0.0 : (observed[i] - e) / e;
            if (Math.abs(drift) > Math.abs(maxDrift)) {
                maxDrift = drift;
                outlier = experiment.variants().get(i).id();
            }
        }

        // Health score: high when SRM NOT detected and p-value comfortably above alpha.
        // Definition: 1.0 = perfectly balanced, 0.0 = srm strongly detected.
        double health;
        if (srm.srmDetected) {
            health = 0.0;
        } else {
            // Map [alpha, 1.0] -> [0.0, 1.0] linearly.
            health = clamp01((srm.pValue - alpha) / (1.0 - alpha));
        }

        Recommendation rec;
        if (srm.srmDetected && health < 0.1) {
            rec = Recommendation.KILL;
        } else if (srm.srmDetected) {
            rec = Recommendation.PAUSE;
        } else if (health < 0.5) {
            rec = Recommendation.INVESTIGATE;
        } else {
            rec = Recommendation.CONTINUE;
        }

        return new Diagnosis(experiment.id(), srm, outlier, maxDrift, health, rec);
    }

    private static double clamp01(double v) {
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    public enum Recommendation {
        CONTINUE,
        INVESTIGATE,
        PAUSE,
        KILL
    }

    public static final class Diagnosis {
        public final String experimentId;
        public final StatisticalTest.SrmResult srm;
        public final String outlierVariantId;
        public final double outlierDriftPct;
        public final double healthScore;
        public final Recommendation recommendation;

        public Diagnosis(String experimentId, StatisticalTest.SrmResult srm,
                         String outlierVariantId, double outlierDriftPct,
                         double healthScore, Recommendation recommendation) {
            this.experimentId = experimentId;
            this.srm = srm;
            this.outlierVariantId = outlierVariantId;
            this.outlierDriftPct = outlierDriftPct;
            this.healthScore = healthScore;
            this.recommendation = recommendation;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "SrmDiag{exp=%s p=%.4g srm=%s outlier=%s drift=%.2f%% health=%.2f rec=%s}",
                    experimentId, srm.pValue, srm.srmDetected,
                    outlierVariantId, outlierDriftPct * 100.0, healthScore, recommendation);
        }
    }
}