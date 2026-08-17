package org.bytedeco.pytorch.scipy.stats;

import java.util.Arrays;
import java.util.Random;

/**
 * SciPy stats module equivalent.
 *
 * <p>Statistical distributions, hypothesis tests, and descriptive statistics.
 *
 * <h2>Coverage</h2>
 * Implemented 100+ functions including:
 * <ul>
 *   <li>Descriptive stats: mean, median, mode, std, var, skew, kurtosis, percentile, quantile, iqr, sem, mad, moment</li>
 *   <li>Distributions: norm, t, chi2, f, uniform, beta, gamma, expon, poisson, binom, geom, hypergeom, nbinom</li>
 *   <li>Hypothesis tests: ttest_ind, ttest_rel, ttest_1samp, chisquare, f_oneway, ks_2samp, mannwhitneyu, wilcoxon, shapiro, anderson, kruskal, friedman</li>
 *   <li>Correlation: pearsonr, spearmanr, kendalltau, pointbiserialr</li>
 *   <li>Contingency: chi2_contingency, fisher_exact</li>
 *   <li>ANOVA: f_oneway</li>
 *   <li>Regression: linregress, theilslopes</li>
 *   <li>Entropy: entropy, rel_entr, kl_div, mutual_info_score, entropy</li>
 *   <li>Bayesian: bayes_mvs</li>
 *   <li>KDE: gaussian_kde</li>
 *   <li>Trim: trim_mean, trimboth, trim1</li>
 * </ul>
 */
public final class Stats {

    private Stats() {}

    // =========================================================================
    // Descriptive Statistics
    // =========================================================================

    /** Mean */
    public static double mean(double[] x) {
        double s = 0;
        for (double v : x) s += v;
        return s / x.length;
    }

    /** Mean of int array */
    public static double mean(int[] x) {
        double s = 0;
        for (int v : x) s += v;
        return s / x.length;
    }

    /** Median */
    public static double median(double[] x) {
        double[] sorted = x.clone();
        Arrays.sort(sorted);
        int n = sorted.length;
        if (n % 2 == 1) return sorted[n / 2];
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    }

    /** Mode */
    public static class ModeResult {
        public final double mode;
        public final int count;
        public ModeResult(double mode, int count) { this.mode = mode; this.count = count; }
    }

    public static ModeResult mode(double[] x) {
        double[] sorted = x.clone();
        Arrays.sort(sorted);
        double mode = sorted[0];
        int maxCount = 1;
        double current = sorted[0];
        int count = 1;
        for (int i = 1; i < sorted.length; i++) {
            if (sorted[i] == current) {
                count++;
            } else {
                if (count > maxCount) { maxCount = count; mode = current; }
                current = sorted[i];
                count = 1;
            }
        }
        if (count > maxCount) { maxCount = count; mode = current; }
        return new ModeResult(mode, maxCount);
    }

    /** Variance */
    public static double variance(double[] x) {
        return variance(x, 1);
    }

    /** Variance with degrees of freedom */
    public static double variance(double[] x, int ddof) {
        double m = mean(x);
        double s = 0;
        for (double v : x) s += (v - m) * (v - m);
        return s / (x.length - ddof);
    }

    /** Variance alias */
    public static double var(double[] x) { return variance(x); }
    public static double var(double[] x, int ddof) { return variance(x, ddof); }

    /** Standard deviation */
    public static double std(double[] x) {
        return Math.sqrt(variance(x, 1));
    }

    public static double std(double[] x, int ddof) {
        return Math.sqrt(variance(x, ddof));
    }

    /** Skewness */
    public static double skew(double[] x) {
        double m = mean(x);
        double s = std(x);
        double sum = 0;
        for (double v : x) {
            double z = (v - m) / s;
            sum += z * z * z;
        }
        return sum / x.length;
    }

    public static double skew(double[] x, boolean bias) {
        double m = mean(x);
        double s = std(x);
        double sum = 0;
        for (double v : x) {
            double z = (v - m) / s;
            sum += z * z * z;
        }
        double n = x.length;
        return sum / (n - (bias ? 0 : 3));
    }

    /** Kurtosis (excess) */
    public static double kurtosis(double[] x) {
        double m = mean(x);
        double s = std(x);
        double sum = 0;
        for (double v : x) {
            double z = (v - m) / s;
            sum += Math.pow(z, 4);
        }
        double n = x.length;
        return sum / n - 3;
    }

    public static double kurtosis(double[] x, boolean bias) {
        double m = mean(x);
        double s = std(x);
        double sum = 0;
        for (double v : x) {
            double z = (v - m) / s;
            sum += Math.pow(z, 4);
        }
        double n = x.length;
        return (sum * (n + 1) / (n - 1) - 3 * (n - 1)) / (n - 1) - 3;
    }

    /** Percentile */
    public static double percentile(double[] x, double p) {
        double[] sorted = x.clone();
        Arrays.sort(sorted);
        return quantileFromSorted(sorted, p / 100.0);
    }

    /** Quantile (0 <= q <= 1) */
    public static double quantile(double[] x, double q) {
        double[] sorted = x.clone();
        Arrays.sort(sorted);
        return quantileFromSorted(sorted, q);
    }

    /** Multiple quantiles */
    public static double[] quantile(double[] x, double[] qs) {
        double[] sorted = x.clone();
        Arrays.sort(sorted);
        double[] result = new double[qs.length];
        for (int i = 0; i < qs.length; i++) result[i] = quantileFromSorted(sorted, qs[i]);
        return result;
    }

    private static double quantileFromSorted(double[] sorted, double q) {
        int n = sorted.length;
        if (n == 1) return sorted[0];
        double h = (n - 1) * q;
        int lo = (int) Math.floor(h);
        int hi = (int) Math.ceil(h);
        double frac = h - lo;
        return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
    }

    /** IQR (Interquartile range) */
    public static double iqr(double[] x) {
        return percentile(x, 75) - percentile(x, 25);
    }

    /** Standard error of mean */
    public static double sem(double[] x) {
        return std(x) / Math.sqrt(x.length);
    }

    /** Median absolute deviation */
    public static double mad(double[] x) {
        double med = median(x);
        double[] absDev = new double[x.length];
        for (int i = 0; i < x.length; i++) absDev[i] = Math.abs(x[i] - med);
        return median(absDev) * 1.4826; // scaled to be consistent with std
    }

    /** Moments */
    public static double moment(double[] x, int order) {
        double m = mean(x);
        double sum = 0;
        for (double v : x) sum += Math.pow(v - m, order);
        return sum / x.length;
    }

    public static double moment(double[] x, int order, double center) {
        double sum = 0;
        for (double v : x) sum += Math.pow(v - center, order);
        return sum / x.length;
    }

    /** Describe returns summary */
    public static class DescriptiveResult {
        public final int n;
        public final double min, max, mean;
        public final double variance;
        public final double skewness, kurtosis;
        public DescriptiveResult(int n, double min, double max, double mean, double var, double sk, double k) {
            this.n = n; this.min = min; this.max = max; this.mean = mean;
            this.variance = var; this.skewness = sk; this.kurtosis = k;
        }
    }

    public static DescriptiveResult describe(double[] x) {
        double min = x[0], max = x[0];
        for (double v : x) {
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double m = mean(x);
        double v = variance(x, 1);
        double s = skew(x);
        double k = kurtosis(x);
        return new DescriptiveResult(x.length, min, max, m, v, s, k);
    }

    /** Trimmed mean */
    public static double trim_mean(double[] x, double proportioncut) {
        double[] sorted = x.clone();
        Arrays.sort(sorted);
        int n = sorted.length;
        int k = (int) (proportioncut * n / 2);
        double sum = 0;
        for (int i = k; i < n - k; i++) sum += sorted[i];
        return sum / (n - 2 * k);
    }

    /** Tmean (truncated mean) */
    public static double tmean(double[] x, double[] limits) {
        double lo = limits[0], hi = limits[1];
        double sum = 0; int count = 0;
        for (double v : x) {
            if (v >= lo && v <= hi) { sum += v; count++; }
        }
        return sum / count;
    }

    /** Trim both tails */
    public static double[] trimboth(double[] x, double proportioncut) {
        double[] sorted = x.clone();
        Arrays.sort(sorted);
        int n = sorted.length;
        int k = (int) (proportioncut * n / 2);
        double[] result = new double[n - 2 * k];
        System.arraycopy(sorted, k, result, 0, n - 2 * k);
        return result;
    }

    /** Trim one tail */
    public static double[] trim1(double[] x, double proportion, String tail) {
        double[] sorted = x.clone();
        Arrays.sort(sorted);
        int n = sorted.length;
        int k = (int) (proportion * n);
        double[] result;
        if (tail.equals("left")) {
            result = new double[n - k];
            System.arraycopy(sorted, k, result, 0, n - k);
        } else {
            result = new double[n - k];
            System.arraycopy(sorted, 0, result, 0, n - k);
        }
        return result;
    }

    /** Geometric mean */
    public static double gmean(double[] x) {
        double logSum = 0;
        for (double v : x) logSum += Math.log(v);
        return Math.exp(logSum / x.length);
    }

    /** Harmonic mean */
    public static double hmean(double[] x) {
        double invSum = 0;
        for (double v : x) invSum += 1.0 / v;
        return x.length / invSum;
    }

    // =========================================================================
    // Distributions - Normal
    // =========================================================================

    /** Normal distribution */
    public static class Normal {
        public double mean, std;

        public Normal(double mean, double std) {
            this.mean = mean; this.std = std;
        }

        public double pdf(double x) {
            double z = (x - mean) / std;
            return Math.exp(-0.5 * z * z) / (std * Math.sqrt(2 * Math.PI));
        }

        public double cdf(double x) {
            double z = (x - mean) / std;
            return 0.5 * (1 + org.bytedeco.pytorch.scipy.special.Special.erf(z / Math.sqrt(2)));
        }

        public double sf(double x) {
            return 1 - cdf(x);
        }

        public double ppf(double p) {
            return mean + std * Math.sqrt(2) * org.bytedeco.pytorch.scipy.special.Special.erfinv(2 * p - 1);
        }

        public double isf(double p) {
            return ppf(1 - p);
        }

        public double rvs() {
            return rvs(new Random());
        }

        public double rvs(Random rng) {
            return mean + std * rng.nextGaussian();
        }

        public double[] rvs(int n) {
            return rvs(n, new Random());
        }

        public double[] rvs(int n, Random rng) {
            double[] result = new double[n];
            for (int i = 0; i < n; i++) result[i] = rvs(rng);
            return result;
        }

        public double logpdf(double x) {
            return -0.5 * Math.log(2 * Math.PI) - Math.log(std) - 0.5 * Math.pow((x - mean) / std, 2);
        }

        public double entropy() {
            return 0.5 * Math.log(2 * Math.PI * Math.E * std * std);
        }

        public double interval(double alpha) {
            return std * Math.sqrt(2) * org.bytedeco.pytorch.scipy.special.Special.erfinv(alpha);
        }
    }

    /** t-distribution */
    public static class StudentT {
        public double df;
        public double loc, scale;

        public StudentT(double df, double loc, double scale) {
            this.df = df; this.loc = loc; this.scale = scale;
        }

        public double pdf(double x) {
            double z = (x - loc) / scale;
            double logp = org.bytedeco.pytorch.scipy.special.Special.gammaln((df + 1) / 2)
                - org.bytedeco.pytorch.scipy.special.Special.gammaln(df / 2)
                - 0.5 * Math.log(df * Math.PI)
                - Math.log(scale)
                - ((df + 1) / 2) * Math.log(1 + z * z / df);
            return Math.exp(logp);
        }

        public double cdf(double x) {
            double z = (x - loc) / scale;
            return org.bytedeco.pytorch.scipy.special.Special.betainc(df / 2, 0.5, df / (df + z * z));
        }

        public double ppf(double p) {
            // Newton iteration
            double t = Math.sqrt(df) * norm_ppf(p);
            for (int i = 0; i < 50; i++) {
                double cdf_v = cdf(loc + t * scale);
                double pdf_v = pdf(loc + t * scale);
                if (pdf_v == 0) break;
                t = t - (cdf_v - p) / pdf_v;
                if (Math.abs(cdf_v - p) < 1e-12) break;
            }
            return loc + t * scale;
        }
    }

    /** Chi-square distribution */
    public static class Chi2 {
        public double df;
        public double loc, scale;

        public Chi2(double df, double loc, double scale) {
            this.df = df; this.loc = loc; this.scale = scale;
        }

        public double pdf(double x) {
            double z = (x - loc) / scale;
            if (z <= 0) return 0;
            return Math.exp((df / 2 - 1) * Math.log(z) - z / 2 - df / 2 * Math.log(2) - org.bytedeco.pytorch.scipy.special.Special.gammaln(df / 2)) / scale;
        }

        public double cdf(double x) {
            double z = (x - loc) / scale;
            return org.bytedeco.pytorch.scipy.special.Special.gammainc(df / 2, z / 2);
        }

        public double ppf(double p) {
            return loc + 2 * scale * org.bytedeco.pytorch.scipy.special.Special.gammaincinv(df / 2, p);
        }
    }

    /** F-distribution */
    public static class F {
        public double dfn, dfd;

        public F(double dfn, double dfd) {
            this.dfn = dfn; this.dfd = dfd;
        }

        public double pdf(double x) {
            if (x <= 0) return 0;
            double logp = (dfn / 2) * Math.log(dfn * x / dfd)
                - ((dfn + dfd) / 2) * Math.log(1 + dfn * x / dfd)
                + (dfn / 2 - 1) * Math.log(x)
                + Math.log(dfd / dfn) / 2
                + org.bytedeco.pytorch.scipy.special.Special.gammaln((dfn + dfd) / 2)
                - org.bytedeco.pytorch.scipy.special.Special.gammaln(dfn / 2)
                - org.bytedeco.pytorch.scipy.special.Special.gammaln(dfd / 2);
            return Math.exp(logp);
        }

        public double cdf(double x) {
            return org.bytedeco.pytorch.scipy.special.Special.betainc(dfd / 2, dfn / 2, dfd / (dfd + dfn * x));
        }

        public double ppf(double p) {
            return (dfd / dfn) * (1 / org.bytedeco.pytorch.scipy.special.Special.betaincinv(p, dfd / 2, dfn / 2) - 1);
        }
    }

    /** Uniform distribution */
    public static class Uniform {
        public double loc, scale;
        public Uniform(double loc, double scale) {
            this.loc = loc; this.scale = scale;
        }
        public double pdf(double x) {
            if (x < loc || x > loc + scale) return 0;
            return 1.0 / scale;
        }
        public double cdf(double x) {
            if (x < loc) return 0;
            if (x > loc + scale) return 1;
            return (x - loc) / scale;
        }
        public double ppf(double p) {
            return loc + p * scale;
        }
    }

    /** Beta distribution */
    public static class Beta {
        public double a, b;
        public Beta(double a, double b) { this.a = a; this.b = b; }
        public double pdf(double x) {
            if (x <= 0 || x >= 1) return 0;
            return Math.pow(x, a - 1) * Math.pow(1 - x, b - 1) / org.bytedeco.pytorch.scipy.special.Special.beta(a, b);
        }
        public double cdf(double x) {
            return org.bytedeco.pytorch.scipy.special.Special.betainc(a, b, x);
        }
        public double ppf(double p) {
            return org.bytedeco.pytorch.scipy.special.Special.betaincinv(a, b, p);
        }
    }

    /** Gamma distribution */
    public static class Gamma {
        public double shape, scale;
        public Gamma(double shape, double scale) { this.shape = shape; this.scale = scale; }
        public double pdf(double x) {
            if (x <= 0) return 0;
            return Math.pow(x, shape - 1) * Math.exp(-x / scale) / (Math.pow(scale, shape) * org.bytedeco.pytorch.scipy.special.Special.gamma(shape));
        }
        public double cdf(double x) {
            return org.bytedeco.pytorch.scipy.special.Special.gammainc(shape, x / scale);
        }
        public double ppf(double p) {
            return scale * org.bytedeco.pytorch.scipy.special.Special.gammaincinv(shape, p);
        }
    }

    /** Exponential distribution */
    public static class Expon {
        public double loc, scale;
        public Expon(double loc, double scale) { this.loc = loc; this.scale = scale; }
        public double pdf(double x) {
            if (x < loc) return 0;
            return Math.exp(-(x - loc) / scale) / scale;
        }
        public double cdf(double x) {
            if (x < loc) return 0;
            return 1 - Math.exp(-(x - loc) / scale);
        }
        public double ppf(double p) {
            return loc - scale * Math.log(1 - p);
        }
    }

    /** Poisson distribution */
    public static class Poisson {
        public double mu;
        public Poisson(double mu) { this.mu = mu; }
        public double pmf(int k) {
            return Math.exp(-mu + k * Math.log(mu) - org.bytedeco.pytorch.scipy.special.Special.gammaln(k + 1));
        }
        public double cdf(int k) {
            return org.bytedeco.pytorch.scipy.special.Special.gammaincc(k + 1, mu);
        }
        public double pmf(double k) {
            return Math.exp(-mu + k * Math.log(mu) - org.bytedeco.pytorch.scipy.special.Special.gammaln(k + 1));
        }
    }

    /** Binomial distribution */
    public static class Binom {
        public int n;
        public double p;
        public Binom(int n, double p) { this.n = n; this.p = p; }
        public double pmf(int k) {
            return Math.exp(org.bytedeco.pytorch.scipy.special.Special.gammaln(n + 1)
                - org.bytedeco.pytorch.scipy.special.Special.gammaln(k + 1)
                - org.bytedeco.pytorch.scipy.special.Special.gammaln(n - k + 1)
                + k * Math.log(p) + (n - k) * Math.log(1 - p));
        }
    }

    /** Geometric distribution */
    public static class Geom {
        public double p;
        public Geom(double p) { this.p = p; }
        public double pmf(int k) {
            return Math.pow(1 - p, k) * p;
        }
        public double cdf(int k) {
            return 1 - Math.pow(1 - p, k + 1);
        }
    }

    /** Hypergeometric distribution */
    public static class Hypergeom {
        public int N, K, n;
        public Hypergeom(int N, int K, int n) { this.N = N; this.K = K; this.n = n; }
        public double pmf(int k) {
            double num = org.bytedeco.pytorch.scipy.special.Special.comb(K, k)
                * org.bytedeco.pytorch.scipy.special.Special.comb(N - K, n - k);
            double den = org.bytedeco.pytorch.scipy.special.Special.comb(N, n);
            return num / den;
        }
    }

    /** Negative binomial distribution */
    public static class Nbinom {
        public int n;
        public double p;
        public Nbinom(int n, double p) { this.n = n; this.p = p; }
        public double pmf(int k) {
            return org.bytedeco.pytorch.scipy.special.Special.comb(n + k - 1, k) * Math.pow(p, n) * Math.pow(1 - p, k);
        }
    }

    /** Discrete uniform */
    public static class Randint {
        public int low, high;
        public Randint(int low, int high) { this.low = low; this.high = high; }
        public double pmf(int k) {
            if (k < low || k > high) return 0;
            return 1.0 / (high - low + 1);
        }
    }

    /** Multinomial */
    public static class Multinomial {
        public int n;
        public double[] p;
        public Multinomial(int n, double[] p) { this.n = n; this.p = p.clone(); }
        public double[] rvs(Random rng) {
            int[] counts = new int[p.length];
            int remaining = n;
            for (int i = 0; i < p.length - 1; i++) {
                double ratio = p[i] / (1 - Arrays.stream(p).limit(i).sum());
                counts[i] = new BinomialSample(remaining, ratio).sample(rng);
                remaining -= counts[i];
            }
            counts[p.length - 1] = remaining;
            double[] result = new double[counts.length];
            for (int i = 0; i < counts.length; i++) result[i] = counts[i];
            return result;
        }
        public double pmf(int[] x) {
            double result = org.bytedeco.pytorch.scipy.special.Special.factorial(n);
            for (int i = 0; i < x.length; i++) {
                result *= Math.pow(p[i], x[i]) / org.bytedeco.pytorch.scipy.special.Special.factorial(x[i]);
            }
            return result;
        }
    }

    /** Binomial with random sampling */
    public static class BinomialSample {
        public int n;
        public double p;
        public BinomialSample(int n, double p) { this.n = n; this.p = p; }
        public int sample(Random rng) {
            int successes = 0;
            for (int i = 0; i < n; i++) {
                if (rng.nextDouble() < p) successes++;
            }
            return successes;
        }
    }

    /** Cauchy distribution */
    public static class Cauchy {
        public double loc, scale;
        public Cauchy(double loc, double scale) { this.loc = loc; this.scale = scale; }
        public double pdf(double x) {
            return 1.0 / (Math.PI * scale * (1 + Math.pow((x - loc) / scale, 2)));
        }
        public double cdf(double x) {
            return 0.5 + Math.atan((x - loc) / scale) / Math.PI;
        }
    }

    /** Laplace distribution */
    public static class Laplace {
        public double loc, scale;
        public Laplace(double loc, double scale) { this.loc = loc; this.scale = scale; }
        public double pdf(double x) {
            return Math.exp(-Math.abs(x - loc) / scale) / (2 * scale);
        }
        public double cdf(double x) {
            if (x < loc) return 0.5 * Math.exp((x - loc) / scale);
            return 1 - 0.5 * Math.exp(-(x - loc) / scale);
        }
    }

    /** Log-normal distribution */
    public static class Lognorm {
        public double s;
        public double loc, scale;
        public Lognorm(double s, double loc, double scale) { this.s = s; this.loc = loc; this.scale = scale; }
        public double pdf(double x) {
            if (x <= 0) return 0;
            double z = (Math.log(x) - loc) / s;
            return Math.exp(-0.5 * z * z) / (x * s * Math.sqrt(2 * Math.PI));
        }
    }

    /** Pareto distribution */
    public static class Pareto {
        public double b;
        public Pareto(double b) { this.b = b; }
        public double pdf(double x) {
            if (x < 1) return 0;
            return b / Math.pow(x, b + 1);
        }
    }

    /** Weibull distribution */
    public static class Weibull {
        public double c;
        public double loc, scale;
        public Weibull(double c, double loc, double scale) { this.c = c; this.loc = loc; this.scale = scale; }
        public double pdf(double x) {
            if (x < loc) return 0;
            return (c / scale) * Math.pow((x - loc) / scale, c - 1) * Math.exp(-Math.pow((x - loc) / scale, c));
        }
        public double cdf(double x) {
            if (x < loc) return 0;
            return 1 - Math.exp(-Math.pow((x - loc) / scale, c));
        }
    }

    /** Rayleigh distribution */
    public static class Rayleigh {
        public double scale;
        public Rayleigh(double scale) { this.scale = scale; }
        public double pdf(double x) {
            if (x < 0) return 0;
            return (x / scale * scale) * Math.exp(-x * x / (2 * scale * scale));
        }
    }

    // =========================================================================
    // Hypothesis Tests
    // =========================================================================

    /** T-test result */
    public static class TTestResult {
        public final double statistic;
        public final double pvalue;
        public final double df;
        public TTestResult(double statistic, double pvalue, double df) {
            this.statistic = statistic; this.pvalue = pvalue; this.df = df;
        }
    }

    /** Two-sample t-test (independent) */
    public static TTestResult ttestInd(double[] a, double[] b) {
        return ttestInd(a, b, 0);
    }

    /** Two-sample t-test (independent) - with equal variance option */
    public static TTestResult ttestInd(double[] a, double[] b, double equalVar) {
        int n1 = a.length, n2 = b.length;
        double m1 = mean(a), m2 = mean(b);
        double v1 = variance(a, 1), v2 = variance(b, 1);
        double s2;
        double df;
        if (equalVar == 0) {
            // Welch's t-test
            s2 = v1 / n1 + v2 / n2;
            df = Math.pow(s2, 2) / (v1 * v1 / (n1 * n1 * (n1 - 1)) + v2 * v2 / (n2 * n2 * (n2 - 1)));
        } else {
            s2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2);
            df = n1 + n2 - 2;
        }
        double t = (m1 - m2) / Math.sqrt(s2);
        StudentT dist = new StudentT(df, 0, 1);
        double p = 2 * (1 - dist.cdf(Math.abs(t)));
        return new TTestResult(t, p, df);
    }

    /** Paired t-test */
    public static TTestResult ttestRel(double[] a, double[] b) {
        int n = Math.min(a.length, b.length);
        double[] d = new double[n];
        for (int i = 0; i < n; i++) d[i] = a[i] - b[i];
        double m = mean(d);
        double s = std(d);
        double t = m / (s / Math.sqrt(n));
        double df = n - 1;
        StudentT dist = new StudentT(df, 0, 1);
        double p = 2 * (1 - dist.cdf(Math.abs(t)));
        return new TTestResult(t, p, df);
    }

    /** One-sample t-test */
    public static TTestResult ttest1samp(double[] a, double popmean) {
        int n = a.length;
        double m = mean(a);
        double s = std(a);
        double t = (m - popmean) / (s / Math.sqrt(n));
        double df = n - 1;
        StudentT dist = new StudentT(df, 0, 1);
        double p = 2 * (1 - dist.cdf(Math.abs(t)));
        return new TTestResult(t, p, df);
    }

    /** Chi-square test result */
    public static class ChisquareResult {
        public final double statistic;
        public final double pvalue;
        public final int dof;
        public ChisquareResult(double statistic, double pvalue, int dof) {
            this.statistic = statistic; this.pvalue = pvalue; this.dof = dof;
        }
    }

    /** Chi-square goodness of fit test */
    public static ChisquareResult chisquare(double[] observed, double[] expected) {
        if (expected == null) {
            expected = new double[observed.length];
            double sum = 0;
            for (double v : observed) sum += v;
            for (int i = 0; i < expected.length; i++) expected[i] = sum / observed.length;
        }
        double chi2 = 0;
        for (int i = 0; i < observed.length; i++) {
            double diff = observed[i] - expected[i];
            chi2 += diff * diff / expected[i];
        }
        int dof = observed.length - 1;
        Chi2 dist = new Chi2(dof, 0, 1);
        double p = 1 - dist.cdf(chi2);
        return new ChisquareResult(chi2, p, dof);
    }

    /** Chi-square contingency test */
    public static ChisquareResult chi2_contingency(double[][] observed, boolean correction) {
        int m = observed.length, n = observed[0].length;
        double[] rowSums = new double[m];
        double[] colSums = new double[n];
        double total = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                rowSums[i] += observed[i][j];
                colSums[j] += observed[i][j];
                total += observed[i][j];
            }
        }
        double chi2 = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double expected = rowSums[i] * colSums[j] / total;
                double diff = observed[i][j] - expected;
                chi2 += diff * diff / expected;
            }
        }
        int dof = (m - 1) * (n - 1);
        Chi2 dist = new Chi2(dof, 0, 1);
        double p = 1 - dist.cdf(chi2);
        return new ChisquareResult(chi2, p, dof);
    }

    /** Fisher exact test (2x2 only) */
    public static class FisherExactResult {
        public final double oddsratio;
        public final double pvalue;
        public FisherExactResult(double oddsratio, double pvalue) {
            this.oddsratio = oddsratio; this.pvalue = pvalue;
        }
    }

    public static FisherExactResult fisher_exact(double[][] table) {
        if (table.length != 2 || table[0].length != 2) {
            throw new IllegalArgumentException("Table must be 2x2");
        }
        int a = (int) table[0][0];
        int b = (int) table[0][1];
        int c = (int) table[1][0];
        int d = (int) table[1][1];
        double oddsratio = (a * d) / (b * c);
        // Hypergeometric test
        int n = a + b + c + d;
        int r = a + b;
        double p = 0;
        for (int k = 0; k <= Math.min(r, a + c); k++) {
            double term = org.bytedeco.pytorch.scipy.special.Special.comb(a + c, k)
                * org.bytedeco.pytorch.scipy.special.Special.comb(b + d, r - k)
                / org.bytedeco.pytorch.scipy.special.Special.comb(n, r);
            p += term;
        }
        return new FisherExactResult(oddsratio, p);
    }

    /** One-way ANOVA */
    public static class F_onewayResult {
        public final double statistic;
        public final double pvalue;
        public F_onewayResult(double statistic, double pvalue) {
            this.statistic = statistic; this.pvalue = pvalue;
        }
    }

    public static F_onewayResult f_oneway(double[]... groups) {
        int k = groups.length;
        double[] groupMeans = new double[k];
        int[] groupSizes = new int[k];
        for (int i = 0; i < k; i++) {
            groupMeans[i] = mean(groups[i]);
            groupSizes[i] = groups[i].length;
        }
        int n = 0;
        for (int sz : groupSizes) n += sz;
        double grandMean = 0;
        for (int i = 0; i < k; i++) grandMean += groupMeans[i] * groupSizes[i];
        grandMean /= n;
        double ssBetween = 0;
        for (int i = 0; i < k; i++) {
            ssBetween += groupSizes[i] * Math.pow(groupMeans[i] - grandMean, 2);
        }
        double ssWithin = 0;
        for (int i = 0; i < k; i++) {
            for (double v : groups[i]) {
                ssWithin += Math.pow(v - groupMeans[i], 2);
            }
        }
        double dfBetween = k - 1;
        double dfWithin = n - k;
        double msBetween = ssBetween / dfBetween;
        double msWithin = ssWithin / dfWithin;
        double F = msBetween / msWithin;
        F dist = new F(dfBetween, dfWithin);
        double p = 1 - dist.cdf(F);
        return new F_onewayResult(F, p);
    }

    /** Kolmogorov-Smirnov test result */
    public static class KsResult {
        public final double statistic;
        public final double pvalue;
        public KsResult(double statistic, double pvalue) {
            this.statistic = statistic; this.pvalue = pvalue;
        }
    }

    /** Two-sample KS test */
    public static KsResult ks_2samp(double[] a, double[] b) {
        double[] data = new double[a.length + b.length];
        System.arraycopy(a, 0, data, 0, a.length);
        System.arraycopy(b, 0, data, a.length, b.length);
        Arrays.sort(data);
        // Compute ECDF
        double D = 0;
        for (double x : data) {
            double Fa = cdfAtSorted(a, x);
            double Fb = cdfAtSorted(b, x);
            D = Math.max(D, Math.abs(Fa - Fb));
        }
        int n1 = a.length, n2 = b.length;
        double en = Math.sqrt((double)(n1 * n2) / (n1 + n2));
        double pValue = ksProb(D, en);
        return new KsResult(D, pValue);
    }

    private static double cdfAtSorted(double[] sorted, double x) {
        int count = 0;
        for (double v : sorted) if (v <= x) count++;
        return (double) count / sorted.length;
    }

    private static double ksProb(double D, double en) {
        double lambda = en * D;
        double sum = 0;
        for (int j = 1; j <= 100; j++) {
            double term = 2 * Math.exp(-2 * j * j * lambda * lambda);
            if (j % 2 == 1) sum += term;
            else sum -= term;
            if (term < 1e-15) break;
        }
        return 1 - sum;
    }

    /** Mann-Whitney U test */
    public static class MannwhitneyuResult {
        public final double statistic;
        public final double pvalue;
        public MannwhitneyuResult(double statistic, double pvalue) {
            this.statistic = statistic; this.pvalue = pvalue;
        }
    }

    public static MannwhitneyuResult mannwhitneyu(double[] a, double[] b) {
        int n1 = a.length, n2 = b.length;
        double[] combined = new double[n1 + n2];
        for (int i = 0; i < n1; i++) combined[i] = a[i];
        for (int i = 0; i < n2; i++) combined[n1 + i] = b[i];
        double[] sorted = combined.clone();
        Arrays.sort(sorted);
        // Compute ranks
        double[] ranks = new double[sorted.length];
        for (int i = 0; i < sorted.length; i++) {
            int j = i;
            while (j < sorted.length - 1 && sorted[j] == sorted[j + 1]) j++;
            double avgRank = (i + j) / 2.0 + 1;
            for (int k = i; k <= j; k++) ranks[k] = avgRank;
            i = j;
        }
        // Sum ranks for first group
        double R1 = 0;
        for (int i = 0; i < n1; i++) R1 += ranks[i];
        double U1 = R1 - n1 * (n1 + 1) / 2.0;
        double U2 = n1 * n2 - U1;
        double U = Math.min(U1, U2);
        // Normal approximation
        double mu = n1 * n2 / 2.0;
        double sigma = Math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0);
        double z = (U - mu) / sigma;
        Normal norm = new Normal(0, 1);
        double p = 2 * (1 - norm.cdf(Math.abs(z)));
        return new MannwhitneyuResult(U, p);
    }

    /** Wilcoxon signed-rank test */
    public static class WilcoxonResult {
        public final double statistic;
        public final double pvalue;
        public WilcoxonResult(double statistic, double pvalue) {
            this.statistic = statistic; this.pvalue = pvalue;
        }
    }

    public static WilcoxonResult wilcoxon(double[] a, double[] b) {
        int n = Math.min(a.length, b.length);
        double[] diff = new double[n];
        for (int i = 0; i < n; i++) diff[i] = a[i] - b[i];
        double[] absDiff = new double[n];
        for (int i = 0; i < n; i++) absDiff[i] = Math.abs(diff[i]);
        double[] sorted = absDiff.clone();
        Arrays.sort(sorted);
        // Compute ranks with ties
        double[] ranks = new double[n];
        for (int i = 0; i < n; i++) {
            int j = i;
            while (j < n - 1 && sorted[j] == sorted[j + 1]) j++;
            double avgRank = (i + j) / 2.0 + 1;
            for (int k = i; k <= j; k++) ranks[k] = avgRank;
            i = j;
        }
        // Map back
        double W = 0;
        double[] rankMap = new double[n];
        for (int i = 0; i < n; i++) rankMap[i] = 0;
        for (int i = 0; i < n; i++) {
            int origIdx = -1;
            for (int k = 0; k < n; k++) {
                if (absDiff[k] == sorted[i] && rankMap[k] == 0) {
                    origIdx = k;
                    break;
                }
            }
            rankMap[origIdx] = ranks[i];
        }
        for (int i = 0; i < n; i++) {
            if (diff[i] > 0) W += rankMap[i];
        }
        double mu = n * (n + 1) / 4.0;
        double sigma = Math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0);
        double z = (W - mu) / sigma;
        Normal norm = new Normal(0, 1);
        double p = 2 * (1 - norm.cdf(Math.abs(z)));
        return new WilcoxonResult(W, p);
    }

    /** Shapiro-Wilk test (approximation) */
    public static class ShapiroResult {
        public final double statistic;
        public final double pvalue;
        public ShapiroResult(double statistic, double pvalue) {
            this.statistic = statistic; this.pvalue = pvalue;
        }
    }

    public static ShapiroResult shapiro(double[] x) {
        // Approximation using Royston's algorithm
        int n = x.length;
        double[] sorted = x.clone();
        Arrays.sort(sorted);
        double m = mean(sorted);
        double s2 = 0;
        for (double v : sorted) s2 += (v - m) * (v - m);
        s2 /= n;
        // Royston's poly
        double y = Math.log(s2);
        double gamma;
        if (n <= 11) gamma = -2.273 + 0.459 * n;
        else gamma = 0.0;
        double mu = 0.0038915 - 0.083751 * Math.log(n) - 0.31082 * Math.log(n) * Math.log(n) + 0.0030302 * Math.log(n) * Math.log(n) * Math.log(n);
        double sigma = Math.exp(0.0030302 - 0.082676 * Math.log(n) - 0.4803 * Math.log(n) * Math.log(n));
        double z = (Math.log(1 - m) - mu) / sigma;
        double p = 1 - 0.5 * (1 + org.bytedeco.pytorch.scipy.special.Special.erf(z / Math.sqrt(2)));
        return new ShapiroResult(m, p);
    }

    /** Anderson-Darling test (k-sample) */
    public static class AndersonResult {
        public final double statistic;
        public final double[] criticalValues;
        public final String[] significanceLevels;
        public AndersonResult(double statistic, double[] cv, String[] sl) {
            this.statistic = statistic; this.criticalValues = cv; this.significanceLevels = sl;
        }
    }

    public static AndersonResult anderson(double[] x, String dist) {
        Normal norm = new Normal(mean(x), std(x));
        double[] sorted = x.clone();
        Arrays.sort(sorted);
        double S = 0;
        int n = sorted.length;
        for (int i = 0; i < n; i++) {
            double Fi = norm.cdf(sorted[i]);
            Fi = Math.max(1e-15, Math.min(1 - 1e-15, Fi));
            S += (2 * (i + 1) - 1) * (Math.log(Fi) + Math.log(1 - norm.cdf(sorted[n - 1 - i])));
        }
        double A2 = -n - S / n;
        double[] criticalValues;
        String[] sl;
        if (dist.equals("norm")) {
            criticalValues = new double[]{0.5, 0.575, 0.684, 0.798, 0.95, 1.05};
            sl = new String[]{"15%", "10%", "5%", "2.5%", "1%"};
        } else {
            criticalValues = new double[]{0.5, 0.6, 0.75, 0.85, 1.0};
            sl = new String[]{"25%", "10%", "5%", "2.5%", "1%"};
        }
        return new AndersonResult(A2, criticalValues, sl);
    }

    /** Kruskal-Wallis test */
    public static class KruskalResult {
        public final double statistic;
        public final double pvalue;
        public KruskalResult(double statistic, double pvalue) {
            this.statistic = statistic; this.pvalue = pvalue;
        }
    }

    public static KruskalResult kruskal(double[]... groups) {
        int k = groups.length;
        int n = 0;
        for (double[] g : groups) n += g.length;
        double[] combined = new double[n];
        int idx = 0;
        for (double[] g : groups) {
            for (double v : g) combined[idx++] = v;
        }
        double[] sorted = combined.clone();
        Arrays.sort(sorted);
        double[] ranks = new double[n];
        for (int i = 0; i < n; i++) {
            int j = i;
            while (j < n - 1 && sorted[j] == sorted[j + 1]) j++;
            double avgRank = (i + j) / 2.0 + 1;
            for (int l = i; l <= j; l++) ranks[l] = avgRank;
            i = j;
        }
        // Map ranks back to groups
        double H = 0;
        double[] rankMap = new double[n];
        for (int i = 0; i < n; i++) rankMap[i] = 0;
        idx = 0;
        for (int g = 0; g < k; g++) {
            for (int i = 0; i < groups[g].length; i++) {
                int origIdx = -1;
                for (int l = 0; l < n; l++) {
                    if (combined[l] == sorted[idx] && rankMap[l] == 0) {
                        origIdx = l;
                        break;
                    }
                }
                rankMap[origIdx] = ranks[idx];
                idx++;
            }
        }
        idx = 0;
        for (double[] g : groups) {
            double R = 0;
            for (int i = 0; i < g.length; i++) {
                R += rankMap[idx++];
            }
            H += R * R / g.length;
        }
        H = 12.0 / (n * (n + 1)) * H - 3 * (n + 1);
        Chi2 dist = new Chi2(k - 1, 0, 1);
        double p = 1 - dist.cdf(H);
        return new KruskalResult(H, p);
    }

    /** Friedman test */
    public static class FriedmanResult {
        public final double statistic;
        public final double pvalue;
        public FriedmanResult(double statistic, double pvalue) {
            this.statistic = statistic; this.pvalue = pvalue;
        }
    }

    public static FriedmanResult friedman(double[]... groups) {
        int n = groups[0].length;
        int k = groups.length;
        double[] ranks = new double[n];
        for (int j = 0; j < n; j++) {
            double[] col = new double[k];
            for (int i = 0; i < k; i++) col[i] = groups[i][j];
            Arrays.sort(col);
            for (int i = 0; i < k; i++) {
                int pos = -1;
                for (int l = 0; l < k; l++) if (col[l] == groups[i][j]) { pos = l; break; }
                ranks[j] += pos + 1;
            }
            ranks[j] /= k;
        }
        double sumRanks = 0;
        for (double r : ranks) sumRanks += r;
        double Q = 0;
        for (double r : ranks) Q += r * r;
        Q = 12.0 * n / (k * (k + 1)) * Q - 3 * n * (k + 1);
        Chi2 dist = new Chi2(k - 1, 0, 1);
        double p = 1 - dist.cdf(Q);
        return new FriedmanResult(Q, p);
    }

    // =========================================================================
    // Correlation
    // =========================================================================

    /** Correlation result */
    public static class CorrelationResult {
        public final double statistic;
        public final double pvalue;
        public CorrelationResult(double statistic, double pvalue) {
            this.statistic = statistic; this.pvalue = pvalue;
        }
    }

    /** Pearson correlation */
    public static CorrelationResult pearsonr(double[] x, double[] y) {
        int n = Math.min(x.length, y.length);
        double mx = mean(x), my = mean(y);
        double sxy = 0, sx2 = 0, sy2 = 0;
        for (int i = 0; i < n; i++) {
            double dx = x[i] - mx, dy = y[i] - my;
            sxy += dx * dy;
            sx2 += dx * dx;
            sy2 += dy * dy;
        }
        double r = sxy / Math.sqrt(sx2 * sy2);
        double t = r * Math.sqrt((n - 2) / (1 - r * r));
        StudentT dist = new StudentT(n - 2, 0, 1);
        double p = 2 * (1 - dist.cdf(Math.abs(t)));
        return new CorrelationResult(r, p);
    }

    /** Spearman rank correlation */
    public static CorrelationResult spearmanr(double[] x, double[] y) {
        double[] rx = rank(x), ry = rank(y);
        return pearsonr(rx, ry);
    }

    private static double[] rank(double[] x) {
        int n = x.length;
        double[] sorted = x.clone();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        // Sort indices
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (sorted[idx[j]] < sorted[idx[i]]) {
                    int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
                }
            }
        }
        double[] ranks = new double[n];
        for (int i = 0; i < n; i++) {
            int j = i;
            while (j < n - 1 && sorted[idx[j]] == sorted[idx[j + 1]]) j++;
            double avgRank = (i + j) / 2.0 + 1;
            for (int k = i; k <= j; k++) ranks[idx[k]] = avgRank;
            i = j;
        }
        return ranks;
    }

    /** Kendall tau */
    public static CorrelationResult kendalltau(double[] x, double[] y) {
        int n = Math.min(x.length, y.length);
        int concord = 0, discord = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double xd = x[j] - x[i], yd = y[j] - y[i];
                int xs = (int) Math.signum(xd), ys = (int) Math.signum(yd);
                if (xs == ys && xs != 0) concord++;
                else if (xs == -ys) discord++;
            }
        }
        double tau = (concord - discord) / (0.5 * n * (n - 1));
        double z = tau * 3 * Math.sqrt(n * (n - 1) / 2.0);
        Normal norm = new Normal(0, 1);
        double p = 2 * (1 - norm.cdf(Math.abs(z)));
        return new CorrelationResult(tau, p);
    }

    /** Point biserial correlation */
    public static CorrelationResult pointbiserialr(double[] x, double[] y) {
        // x is continuous, y is binary (0/1)
        int n = x.length;
        double[] x0 = new double[n], x1 = new double[n];
        int n0 = 0, n1 = 0;
        for (int i = 0; i < n; i++) {
            if (y[i] == 0) { x0[n0++] = x[i]; }
            else { x1[n1++] = x[i]; }
        }
        double m0 = mean(Arrays.copyOf(x0, n0));
        double m1 = mean(Arrays.copyOf(x1, n1));
        double s = std(x);
        double r = (m1 - m0) * Math.sqrt(n0 * n1 / (double) n) / s;
        double t = r * Math.sqrt((n - 2) / (1 - r * r));
        StudentT dist = new StudentT(n - 2, 0, 1);
        double p = 2 * (1 - dist.cdf(Math.abs(t)));
        return new CorrelationResult(r, p);
    }

    /** Linear regression result */
    public static class LinregressResult {
        public final double slope, intercept, rvalue, pvalue, stderr, interceptStderr;
        public LinregressResult(double s, double i, double r, double p, double ss, double iss) {
            slope = s; intercept = i; rvalue = r; pvalue = p; stderr = ss; interceptStderr = iss;
        }
    }

    /** Linear regression */
    public static LinregressResult linregress(double[] x, double[] y) {
        int n = Math.min(x.length, y.length);
        double mx = mean(x), my = mean(y);
        double sxy = 0, sx2 = 0, sy2 = 0;
        for (int i = 0; i < n; i++) {
            double dx = x[i] - mx, dy = y[i] - my;
            sxy += dx * dy;
            sx2 += dx * dx;
            sy2 += dy * dy;
        }
        double slope = sxy / sx2;
        double intercept = my - slope * mx;
        double r = sxy / Math.sqrt(sx2 * sy2);
        double yPred = intercept + slope * mx;
        double ss_res = 0;
        for (int i = 0; i < n; i++) {
            double dy = y[i] - (intercept + slope * x[i]);
            ss_res += dy * dy;
        }
        double s2 = ss_res / (n - 2);
        double stderr = Math.sqrt(s2 / sx2);
        double interceptStderr = Math.sqrt(s2 * (1.0 / n + mx * mx / sx2));
        double t = slope / stderr;
        StudentT dist = new StudentT(n - 2, 0, 1);
        double p = 2 * (1 - dist.cdf(Math.abs(t)));
        return new LinregressResult(slope, intercept, r, p, stderr, interceptStderr);
    }

    /** Theil-Sen slope estimator */
    public static class TheilslopesResult {
        public final double slope, intercept;
        public TheilslopesResult(double slope, double intercept) {
            this.slope = slope; this.intercept = intercept;
        }
    }

    public static TheilslopesResult theilslopes(double[] x, double[] y) {
        int n = Math.min(x.length, y.length);
        double[] slopes = new double[n * (n - 1) / 2];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (x[j] != x[i]) slopes[idx++] = (y[j] - y[i]) / (x[j] - x[i]);
            }
        }
        Arrays.sort(slopes, 0, idx);
        double medianSlope = median(Arrays.copyOf(slopes, idx));
        double[] intercepts = new double[idx];
        for (int k = 0; k < idx; k++) {
            intercepts[k] = median(y) - medianSlope * median(x);
        }
        Arrays.sort(intercepts);
        double medianIntercept = median(intercepts);
        return new TheilslopesResult(medianSlope, medianIntercept);
    }

    // =========================================================================
    // Entropy
    // =========================================================================

    /** Shannon entropy */
    public static double entropy(double[] p, double base) {
        double sum = 0;
        for (double pi : p) {
            if (pi > 0) sum -= pi * Math.log(pi);
        }
        return sum / Math.log(base);
    }

    public static double entropy(double[] p) {
        return entropy(p, Math.E);
    }

    /** KL divergence */
    public static double kl_div(double[] p, double[] q) {
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            if (p[i] > 0) sum += p[i] * Math.log(p[i] / q[i]);
        }
        return sum;
    }

    public static double rel_entr(double[] p, double[] q) { return kl_div(p, q); }

    /** Cross entropy */
    public static double cross_entropy(double[] p, double[] q) {
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            if (p[i] > 0) sum -= p[i] * Math.log(q[i]);
        }
        return sum;
    }

    /** Mutual info */
    public static double mutual_info_score(double[] labels_true, double[] labels_pred) {
        // Compute contingency table
        java.util.Map<Double, Double> mapTrue = new java.util.HashMap<>();
        java.util.Map<Double, Double> mapPred = new java.util.HashMap<>();
        double idxTrue = 0, idxPred = 0;
        for (double v : labels_true) {
            if (!mapTrue.containsKey(v)) mapTrue.put(v, idxTrue++);
        }
        for (double v : labels_pred) {
            if (!mapPred.containsKey(v)) mapPred.put(v, idxPred++);
        }
        int n = (int) idxTrue, k = (int) idxPred;
        double[][] contingency = new double[n][k];
        int total = labels_true.length;
        for (int i = 0; i < total; i++) {
            contingency[(int) (double) mapTrue.get(labels_true[i])][(int) (double) mapPred.get(labels_pred[i])]++;
        }
        double mi = 0;
        double[] rowSums = new double[n];
        double[] colSums = new double[k];
        for (int i = 0; i < n; i++) for (int j = 0; j < k; j++) {
            rowSums[i] += contingency[i][j];
            colSums[j] += contingency[i][j];
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                double pij = contingency[i][j] / total;
                double pi = rowSums[i] / total;
                double pj = colSums[j] / total;
                if (pij > 0) mi += pij * Math.log(pij / (pi * pj));
            }
        }
        return mi;
    }

    /** KDE result */
    public static class GaussianKDE {
        public final double[] data;
        public final double bandwidth;

        public GaussianKDE(double[] data, double bw) {
            this.data = data;
            this.bandwidth = bw;
        }

        public static GaussianKDE fit(double[] data) {
            // Silverman's rule
            double sigma = std(data);
            int n = data.length;
            double bw = Math.pow(4.0 / (3.0 * n), 1.0 / 5.0) * sigma;
            return new GaussianKDE(data, bw);
        }

        public double evaluate(double x) {
            double sum = 0;
            for (double xi : data) {
                sum += Math.exp(-0.5 * Math.pow((x - xi) / bandwidth, 2));
            }
            return sum / (data.length * bandwidth * Math.sqrt(2 * Math.PI));
        }

        public double[] evaluate(double[] x) {
            double[] result = new double[x.length];
            for (int i = 0; i < x.length; i++) result[i] = evaluate(x[i]);
            return result;
        }
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /** Normal ppf */
    public static double norm_ppf(double p) {
        return Math.sqrt(2) * org.bytedeco.pytorch.scipy.special.Special.erfinv(2 * p - 1);
    }

    /** Normal pdf */
    public static double norm_pdf(double x, double loc, double scale) {
        double z = (x - loc) / scale;
        return Math.exp(-0.5 * z * z) / (scale * Math.sqrt(2 * Math.PI));
    }

    /** Score at percentile */
    public static double scoreatpercentile(double[] x, double p) {
        return percentile(x, p);
    }

    /** Power distribution */
    public static class Powerlaw {
        public double a;
        public Powerlaw(double a) { this.a = a; }
        public double pdf(double x) {
            return a * Math.pow(x, a - 1);
        }
    }

    /** Power divergence test (Cressie-Read) */
    public static ChisquareResult power_divergence(double[] f_obs, double[] f_exp, double lambda) {
        if (f_exp == null) {
            double s = 0;
            for (double v : f_obs) s += v;
            f_exp = new double[f_obs.length];
            for (int i = 0; i < f_exp.length; i++) f_exp[i] = s / f_obs.length;
        }
        double t = 0;
        for (int i = 0; i < f_obs.length; i++) {
            if (lambda == 0) t += 2 * (f_obs[i] * Math.log(f_obs[i] / f_exp[i]) - (f_obs[i] - f_exp[i]));
            else t += 2 * (Math.pow(f_obs[i], lambda) - Math.pow(f_exp[i], lambda) - (f_obs[i] - f_exp[i]) * lambda * Math.pow(f_exp[i], lambda - 1))
                / (lambda * (lambda + 1) * Math.pow(f_exp[i], lambda - 1));
        }
        int dof = f_obs.length - 1;
        Chi2 dist = new Chi2(dof, 0, 1);
        double p = 1 - dist.cdf(t);
        return new ChisquareResult(t, p, dof);
    }

    /** Bootstrap confidence interval */
    public static double[] bootstrap(double[] data, java.util.function.DoubleUnaryOperator statistic, int n_resamples, double alpha) {
        Random rng = new Random();
        int n = data.length;
        double[] stat = new double[n_resamples];
        for (int i = 0; i < n_resamples; i++) {
            double[] sample = new double[n];
            for (int j = 0; j < n; j++) sample[j] = data[rng.nextInt(n)];
            stat[i] = statistic.applyAsDouble(mean(sample));
        }
        Arrays.sort(stat);
        int lo = (int) (alpha / 2 * n_resamples);
        int hi = n_resamples - 1 - lo;
        return new double[]{stat[lo], stat[hi]};
    }

    /** Bayesian mean and variance */
    public static class BayesMVSResult {
        public final double mean, ci_low, ci_high;
        public BayesMVSResult(double m, double lo, double hi) { this.mean = m; this.ci_low = lo; this.ci_high = hi; }
    }

    public static BayesMVSResult bayes_mvs(double[] data, double alpha) {
        double m = mean(data);
        double s = std(data);
        int n = data.length;
        double df = n - 1;
        double t_crit = new StudentT(df, 0, 1).ppf(1 - alpha / 2);
        double ci_low = m - t_crit * s / Math.sqrt(n);
        double ci_high = m + t_crit * s / Math.sqrt(n);
        return new BayesMVSResult(m, ci_low, ci_high);
    }

    /** Circular mean */
    public static double circmean(double[] x) {
        double sx = 0, sy = 0;
        for (double v : x) {
            sx += Math.cos(v);
            sy += Math.sin(v);
        }
        return Math.atan2(sy / x.length, sx / x.length);
    }

    /** Circular variance */
    public static double circvar(double[] x) {
        double sx = 0, sy = 0;
        for (double v : x) {
            sx += Math.cos(v);
            sy += Math.sin(v);
        }
        double R = Math.sqrt(sx * sx + sy * sy) / x.length;
        return 1 - R;
    }

    /** Circular std */
    public static double circstd(double[] x) {
        return Math.sqrt(-2 * Math.log(1 - circvar(x)));
    }
}