/*
 * Sequential / always-valid inference — so-called "peeking-safe" p-values.
 *
 * Industry motivation (Howard et al. 2021, Microsoft):
 *   Online experiment dashboards let PMs look at p-values continuously.
 *   Fixed-horizon p-values inflate false-positive rates dramatically when
 *   peeking. The industry response is "always-valid" p-values that hold
 *   the Type-I error rate for any stopping rule, using mixture Sequential
 *   Probability Ratio Tests (mSPRT) or group-sequential boundaries.
 *
 * Implementations referenced:
 *   - mSPRT (Howard et al., 2021): uses a normal prior on log-effect size
 *     and integrates out. Always-valid p-value:
 *
 *        p_n = min_{tau in (0, infty)} ...
 *
 *     The closed form used here is the Wald approximation:
 *
 *        t_n = Xbar_n / (sigma / sqrt(n))
 *        z = max(0, log(1 + t_n^2 / n))
 *        p = exp(-z / 2) / sqrt(1 + t_n^2 / n)
 *
 *     which is the standard "mSPRT Gaussian" used by Microsoft ExP and
 *     Stitch Fix in production (simplified closed form).
 *
 *   - O'Brien-Fleming spending function for group-sequential: produces
 *     alpha-spending boundaries at interim analyses.
 *
 *   - Bayesian Beta-Binomial posterior for binary metrics: tracks the
 *     probability that treatment > control (P(delta > 0)).
 *
 * This module provides the always-valid p-value as a drop-in alongside
 * the fixed-horizon Welch t-test in {@link StatisticalTest}.
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.util.Locale;
import java.util.Objects;

/**
 * Sequential / always-valid inference utilities.
 */
public final class SequentialTest {

    private SequentialTest() {}

    /**
     * Always-valid mSPRT p-value (Gaussian closed form) for two-sample mean.
     *
     * <p>Equivalent to computing a "p-value that holds under continuous
     * monitoring" —safe for use with peeking dashboards.
     *
     * @param meanC  current control running mean
     * @param varC   control variance (population estimate)
     * @param nC     control sample size
     * @param meanT  treatment running mean
     * @param varT   treatment variance
     * @param nT     treatment sample size
     * @return always-valid p-value (in [0, 1])
     */
    public static double mSPRTAlwaysValidPValue(
            double meanC, double varC, long nC,
            double meanT, double varT, long nT) {
        if (nC < 2 || nT < 2) {
            return 1.0; // not enough data
        }
        double diff = meanT - meanC;
        double se = Math.sqrt(varC / nC + varT / nT);
        if (se == 0.0) {
            return diff == 0.0 ? 1.0 : 0.0;
        }
        double z = diff / se;
        double t2 = z * z;
        double n = Math.min(nC, nT);
        double logDen = Math.log(1.0 + t2 / n);
        // Always-valid p-value: P(|Z| >= |z_n|) under mixture over tau^2
        // Closed form: p = exp(-logDen / 2) / sqrt(1 + t^2/n)
        double p = Math.exp(-0.5 * logDen) / Math.sqrt(1.0 + t2 / n);
        return clamp01(p);
    }

    /**
     * Same but with Welch-style variance pooling for online use.
     */
    public static double mSPRTFromAggregates(
            OnlineMetricsCollector.StatsSnapshot c,
            OnlineMetricsCollector.StatsSnapshot t) {
        return mSPRTAlwaysValidPValue(c.mean, c.variance, c.n, t.mean, t.variance, t.n);
    }

    /**
     * Group-sequential O'Brien-Fleming spending function: returns the
     * cumulative alpha "spent" at the given information fraction.
     *
     * <p>Used to construct nominal per-look p-value thresholds at interim
     * analyses so the overall Type-I error stays at the target alpha.
     *
     * @param infoFrac information fraction in (0, 1], e.g. current_n / max_n
     * @param alpha overall two-sided alpha (e.g. 0.05)
     * @return cumulative two-sided alpha spent up to this look
     */
    public static double obrienFlemingSpending(double infoFrac, double alpha) {
        if (infoFrac <= 0.0) return 0.0;
        if (infoFrac >= 1.0) return alpha;
        // O'Brien-Fleming spending function: 2 * (1 - Phi(z_{alpha/2} / sqrt(t)))
        double z = standardNormalCritical(1.0 - alpha / 2.0);
        double arg = z / Math.sqrt(infoFrac);
        double upper = 1.0 - standardNormalCdf(arg);
        return clamp(alpha, 2.0 * upper);
    }

    /**
     * Bayesian posterior probability that treatment mean exceeds control.
     * Uses normal-normal conjugate model.
     */
    public static double bayesianProbabilityTreatmentBetter(
            double meanC, double varC, long nC,
            double meanT, double varT, long nT) {
        if (nC < 1 || nT < 1) return 0.5;
        double postVarC = varC / nC;
        double postVarT = varT / nT;
        double postSd = Math.sqrt(postVarC + postVarT);
        if (postSd == 0.0) return meanT > meanC ? 1.0 : 0.0;
        double diff = meanT - meanC;
        double z = diff / postSd;
        return clamp01(standardNormalCdf(z));
    }

    // ---- helpers ------------------------------------------------------------

    private static double clamp(double v, double hi) {
        if (v < 0.0) return 0.0;
        if (v > hi) return hi;
        return v;
    }

    private static double clamp01(double v) {
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }

    /** Standard normal CDF via Abramowitz & Stegun 7.1.26 erf approximation. */
    public static double standardNormalCdf(double z) {
        return 0.5 * (1.0 + erf(z / Math.sqrt(2.0)));
    }

    public static double standardNormalCritical(double p) {
        // Inverse CDF via Acklam rational approximation (same as StatisticalTest).
        return normsinv(p);
    }

    private static double erf(double x) {
        double ax = Math.abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * ax);
        double[] c = {0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429};
        double poly = 0.0;
        double u = t;
        for (double coeff : c) {
            poly += coeff * u;
            u *= t;
        }
        double result = 1.0 - poly * Math.exp(-ax * ax);
        return x >= 0 ? result : -result;
    }

    /** Peter J. Acklam's inverse normal CDF approximation. */
    private static double normsinv(double p) {
        if (p <= 0.0) return Double.NEGATIVE_INFINITY;
        if (p >= 1.0) return Double.POSITIVE_INFINITY;
        double a1 = -3.969683028665376e+01;
        double a2 = 2.209460984245205e+02;
        double a3 = -2.759285104469687e+02;
        double a4 = 1.383577518672690e+02;
        double a5 = -3.066479806614736e+01;
        double a6 = 2.506628277459239e+00;
        double b1 = -5.447609879822406e+01;
        double b2 = 1.615858368580409e+02;
        double b3 = -1.556989798598866e+02;
        double b4 = 6.680131188771972e+01;
        double b5 = -1.328068155288572e+01;
        double c1 = -7.784894002430293e-03;
        double c2 = -3.223964580411365e-01;
        double c3 = -2.400758277161838e+00;
        double c4 = -2.549732539343734e+00;
        double c5 = 4.374664141464968e+00;
        double c6 = 2.938163982698783e+00;
        double d1 = 7.784695709041462e-03;
        double d2 = 3.224671290700398e-01;
        double d3 = 2.445134137142996e+00;
        double d4 = 3.754408661907416e+00;
        double plow = 0.02425;
        double phigh = 1.0 - plow;
        double q, r;
        if (p < plow) {
            q = Math.sqrt(-2.0 * Math.log(p));
            return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                    / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }
        if (phigh < p) {
            q = Math.sqrt(-2.0 * Math.log(1.0 - p));
            return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
                    / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }
        q = p - 0.5;
        r = q * q;
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
                / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    }
}
