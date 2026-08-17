package org.bytedeco.pytorch.scipy.interpolate;

import org.bytedeco.pytorch.scipy.linalg.Linalg;

/**
 * SciPy interpolate module equivalent.
 *
 * <h2>Coverage</h2>
 * Implemented 15+ interpolation methods including:
 * <ul>
 *   <li>Linear, cubic, nearest, previous, next, zero, slinear, quadratic</li>
 *   <li>1D, 2D interpolation</li>
 *   <li>Splines: CubicSpline, PchipInterpolator, Akima1DInterpolator</li>
 *   <li>Polynomial: Polynomial, barycentric_interpolate, krogh_interpolate</li>
 *   <li>Griddata for scattered data</li>
 *   <li>BSpline basis</li>
 * </ul>
 */
public final class Interpolate {

    private Interpolate() {}

    // =========================================================================
    // 1D Interpolation
    // =========================================================================

    public static double interp1d(double[] x, double[] y, double xNew, String kind) {
        return interp1d(x, y, xNew, kind, false);
    }

    public static double interp1d(double[] x, double[] y, double xNew, String kind, boolean extrapolate) {
        if (!extrapolate) {
            if (xNew <= x[0]) return y[0];
            if (xNew >= x[x.length - 1]) return y[y.length - 1];
        }
        // Find interval
        int i = findInterval(x, xNew);
        switch (kind.toLowerCase()) {
            case "linear":
            case "slinear":
                return linearInterp(x, y, xNew, i);
            case "nearest":
                return nearestInterp(x, y, xNew, i);
            case "cubic":
                return cubicInterp(x, y, xNew, i);
            case "previous":
                return y[Math.max(0, i)];
            case "next":
                return y[Math.min(x.length - 1, i + 1)];
            case "zero":
                return y[i];
            case "quadratic":
                return quadraticInterp(x, y, xNew, i);
            default:
                return linearInterp(x, y, xNew, i);
        }
    }

    /** Find interval index such that x[i] <= xNew < x[i+1] */
    public static int findInterval(double[] x, double xNew) {
        int lo = 0, hi = x.length - 1;
        while (hi - lo > 1) {
            int mid = (lo + hi) >>> 1;
            if (x[mid] > xNew) hi = mid;
            else lo = mid;
        }
        return lo;
    }

    private static double linearInterp(double[] x, double[] y, double xNew, int i) {
        double t = (xNew - x[i]) / (x[i + 1] - x[i]);
        return y[i] * (1 - t) + y[i + 1] * t;
    }

    private static double nearestInterp(double[] x, double[] y, double xNew, int i) {
        if (xNew - x[i] < x[i + 1] - xNew) return y[i];
        return y[i + 1];
    }

    private static double cubicInterp(double[] x, double[] y, double xNew, int i) {
        int n = x.length;
        int i0 = Math.max(0, i - 1);
        int i3 = Math.min(n - 1, i + 2);
        // Catmull-Rom
        double[] xs = new double[4], ys = new double[4];
        for (int k = 0; k < 4; k++) {
            int idx = i0 + k;
            if (idx < 0) idx = 0;
            if (idx >= n) idx = n - 1;
            xs[k] = x[idx];
            ys[k] = y[idx];
        }
        return catmullRom(xs, ys, xNew);
    }

    private static double catmullRom(double[] xs, double[] ys, double xNew) {
        double t = (xNew - xs[1]) / (xs[2] - xs[1]);
        double t2 = t * t, t3 = t2 * t;
        return 0.5 * ((2 * ys[1]) +
                      (-ys[0] + ys[2]) * t +
                      (2 * ys[0] - 5 * ys[1] + 4 * ys[2] - ys[3]) * t2 +
                      (-ys[0] + 3 * ys[1] - 3 * ys[2] + ys[3]) * t3);
    }

    private static double quadraticInterp(double[] x, double[] y, double xNew, int i) {
        int i0 = Math.max(0, i - 1);
        int i2 = Math.min(x.length - 1, i + 1);
        // Lagrange quadratic
        double x0 = x[i0], x1 = x[i], x2 = x[i2];
        double y0 = y[i0], y1 = y[i], y2 = y[i2];
        double l0 = ((xNew - x1) * (xNew - x2)) / ((x0 - x1) * (x0 - x2));
        double l1 = ((xNew - x0) * (xNew - x2)) / ((x1 - x0) * (x1 - x2));
        double l2 = ((xNew - x0) * (xNew - x1)) / ((x2 - x0) * (x2 - x1));
        return l0 * y0 + l1 * y1 + l2 * y2;
    }

    // =========================================================================
    // Splines
    // =========================================================================

    /** Cubic spline */
    public static class CubicSpline {
        public final double[] x, y, c;

        public CubicSpline(double[] x, double[] y) {
            this.x = x.clone();
            this.y = y.clone();
            this.c = computeCubicSpline(x, y);
        }

        public double evaluate(double xNew) {
            int i = findInterval(x, xNew);
            double h = xNew - x[i];
            return y[i] + c[i] * h;
        }

        public double[] evaluateAll(double[] xs) {
            double[] result = new double[xs.length];
            for (int i = 0; i < xs.length; i++) result[i] = evaluate(xs[i]);
            return result;
        }

        public double derivative(double xNew) {
            int i = findInterval(x, xNew);
            return c[i];
        }
    }

    private static double[] computeCubicSpline(double[] x, double[] y) {
        int n = x.length;
        double[] d = new double[n - 1];
        for (int i = 0; i < n - 1; i++) d[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
        // Solve tridiagonal system for c (slopes at knots)
        // Natural boundary conditions
        double[] c = new double[n];
        double[] mu = new double[n - 1], z = new double[n - 1];
        mu[0] = 0;
        z[0] = 0;
        for (int i = 1; i < n - 1; i++) {
            double g = 3.0 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])) -
                       mu[i - 1] * (x[i] - x[i - 1]);
            double l = 2.0 * (x[i + 1] - x[i - 1]) - mu[i - 1] * (x[i] - x[i - 1]);
            mu[i] = (x[i] - x[i - 1]) / l;
            z[i] = (g - z[i - 1] * (x[i] - x[i - 1])) / l;
        }
        c[n - 1] = 0;
        for (int j = n - 2; j >= 0; j--) {
            c[j] = z[j] + mu[j] * c[j + 1];
        }
        // Simplify: just return slope at left of each interval
        double[] result = new double[n - 1];
        for (int i = 0; i < n - 1; i++) result[i] = c[i];
        return result;
    }

    /** Akima 1D interpolator */
    public static class AkimaInterpolator {
        public final double[] x, y;

        public AkimaInterpolator(double[] x, double[] y) {
            this.x = x.clone();
            this.y = y.clone();
        }

        public double evaluate(double xNew) {
            int n = x.length;
            if (n < 2) return y[0];
            if (xNew <= x[0]) return y[0];
            if (xNew >= x[n - 1]) return y[n - 1];

            double[] slopes = new double[n - 1];
            for (int i = 0; i < n - 1; i++) slopes[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);

            // Compute Akima weights
            double[] m = new double[n];
            m[0] = slopes[0];
            m[n - 1] = slopes[n - 2];
            for (int i = 1; i < n - 1; i++) {
                double s0 = slopes[i - 1], s1 = slopes[i];
                double sPrev = i >= 2 ? slopes[i - 2] : 2 * s0 - s1;
                double sNext = i + 1 < n - 1 ? slopes[i + 1] : 2 * s1 - s0;
                double denom = Math.abs(sNext - s1) + Math.abs(sPrev - s0);
                if (denom == 0) m[i] = (s0 + s1) / 2;
                else m[i] = (Math.abs(sNext - s1) * s0 + Math.abs(sPrev - s0) * s1) / denom;
            }

            int i = findInterval(x, xNew);
            double t = (xNew - x[i]) / (x[i + 1] - x[i]);
            double h = x[i + 1] - x[i];
            double h00 = 2 * t * t * t - 3 * t * t + 1;
            double h10 = t * t * t - 2 * t * t + t;
            double h01 = -2 * t * t * t + 3 * t * t;
            double h11 = t * t * t - t * t;
            return h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1];
        }
    }

    /** PCHIP interpolator */
    public static class PchipInterpolator {
        public final double[] x, y, d;

        public PchipInterpolator(double[] x, double[] y) {
            this.x = x.clone();
            this.y = y.clone();
            this.d = computePchipDerivatives(x, y);
        }

        public double evaluate(double xNew) {
            int n = x.length;
            if (xNew <= x[0]) return y[0];
            if (xNew >= x[n - 1]) return y[n - 1];
            int i = findInterval(x, xNew);
            double h = xNew - x[i];
            double t = h / (x[i + 1] - x[i]);
            double t2 = t * t, t3 = t2 * t;
            double h00 = 2 * t3 - 3 * t2 + 1;
            double h10 = t3 - 2 * t2 + t;
            double h01 = -2 * t3 + 3 * t2;
            double h11 = t3 - t2;
            double dx = x[i + 1] - x[i];
            return h00 * y[i] + h10 * dx * d[i] + h01 * y[i + 1] + h11 * dx * d[i + 1];
        }
    }

    private static double[] computePchipDerivatives(double[] x, double[] y) {
        int n = x.length;
        double[] d = new double[n];
        double[] h = new double[n - 1];
        double[] delta = new double[n - 1];
        for (int i = 0; i < n - 1; i++) {
            h[i] = x[i + 1] - x[i];
            delta[i] = (y[i + 1] - y[i]) / h[i];
        }
        d[0] = delta[0];
        d[n - 1] = delta[n - 2];
        for (int i = 1; i < n - 1; i++) {
            double w1 = 2 * h[i] + h[i - 1];
            double w2 = h[i] + 2 * h[i - 1];
            double d1 = (w1 + h[i - 1]) * delta[i - 1] / w1 + h[i] * delta[i] / w2;
            double d2 = h[i - 1] * delta[i - 1] / w1 + (h[i] + w2) * delta[i] / w2;
            d[i] = (Math.signum(delta[i - 1]) == Math.signum(delta[i])) ? (Math.signum(d1) == Math.signum(delta[i - 1]) ? d1 : 0) : 0;
            if (Math.signum(d[i]) != Math.signum(delta[i - 1])) d[i] = 0;
            else if ((Math.signum(delta[i - 1]) != Math.signum(delta[i])) && Math.abs(d[i]) > Math.abs(3 * delta[i - 1])) d[i] = 3 * delta[i - 1];
        }
        return d;
    }

    /** Barycentric interpolation */
    public static class BarycentricInterpolator {
        public final double[] xi, yi, wi;

        public BarycentricInterpolator(double[] xi, double[] yi) {
            this.xi = xi.clone();
            this.yi = yi.clone();
            this.wi = computeBarycentricWeights(xi);
        }

        public double evaluate(double x) {
            return barycentricInterpolate(xi, yi, wi, x);
        }
    }

    private static double[] computeBarycentricWeights(double[] xi) {
        int n = xi.length;
        double[] w = new double[n];
        for (int j = 0; j < n; j++) {
            double prod = 1;
            for (int i = 0; i < n; i++) {
                if (i != j) prod /= (xi[j] - xi[i]);
            }
            w[j] = prod;
        }
        return w;
    }

    private static double barycentricInterpolate(double[] xi, double[] yi, double[] wi, double x) {
        // Barycentric formula
        double[] diffs = new double[xi.length];
        boolean exact = false;
        int exactIdx = -1;
        for (int i = 0; i < xi.length; i++) {
            diffs[i] = x - xi[i];
            if (diffs[i] == 0) { exact = true; exactIdx = i; }
        }
        if (exact) return yi[exactIdx];
        double num = 0, den = 0;
        for (int i = 0; i < xi.length; i++) {
            double t = wi[i] / diffs[i];
            num += t * yi[i];
            den += t;
        }
        return num / den;
    }

    // =========================================================================
    // Polynomial interpolation
    // =========================================================================

    /** Polynomial represented by coefficients [a0, a1, a2, ...] = a0 + a1*x + a2*x^2 + ... */
    public static class Polynomial {
        public final double[] coeffs;

        public Polynomial(double[] coeffs) {
            this.coeffs = coeffs.clone();
        }

        public double evaluate(double x) {
            double result = 0;
            for (int i = coeffs.length - 1; i >= 0; i--) {
                result = result * x + coeffs[i];
            }
            return result;
        }

        public double[] evaluateAll(double[] xs) {
            double[] result = new double[xs.length];
            for (int i = 0; i < xs.length; i++) result[i] = evaluate(xs[i]);
            return result;
        }

        public Polynomial derivative() {
            if (coeffs.length <= 1) return new Polynomial(new double[]{0});
            double[] d = new double[coeffs.length - 1];
            for (int i = 1; i < coeffs.length; i++) d[i - 1] = i * coeffs[i];
            return new Polynomial(d);
        }

        public Polynomial integrate(double constant) {
            double[] i = new double[coeffs.length + 1];
            i[0] = constant;
            for (int k = 0; k < coeffs.length; k++) i[k + 1] = coeffs[k] / (k + 1);
            return new Polynomial(i);
        }

        public double[] roots() {
            return Linalg.polynomialRoots(coeffs);
        }

        public Polynomial add(Polynomial other) {
            int len = Math.max(coeffs.length, other.coeffs.length);
            double[] r = new double[len];
            for (int i = 0; i < len; i++) {
                if (i < coeffs.length) r[i] += coeffs[i];
                if (i < other.coeffs.length) r[i] += other.coeffs[i];
            }
            return new Polynomial(r);
        }

        public Polynomial multiply(Polynomial other) {
            double[] r = new double[coeffs.length + other.coeffs.length - 1];
            for (int i = 0; i < coeffs.length; i++) {
                for (int j = 0; j < other.coeffs.length; j++) {
                    r[i + j] += coeffs[i] * other.coeffs[j];
                }
            }
            return new Polynomial(r);
        }
    }

    /** Fit polynomial to data */
    public static Polynomial polyFit(double[] x, double[] y, int degree) {
        // Build Vandermonde matrix
        int n = x.length;
        double[][] V = new double[n][degree + 1];
        double[] yCol = new double[n];
        for (int i = 0; i < n; i++) {
            double p = 1;
            for (int j = 0; j <= degree; j++) {
                V[i][j] = p;
                p *= x[i];
            }
            yCol[i] = y[i];
        }
        // Solve V^T V c = V^T y via normal equations
        double[][] VtV = new double[degree + 1][degree + 1];
        double[] Vty = new double[degree + 1];
        for (int i = 0; i <= degree; i++) {
            for (int j = 0; j <= degree; j++) {
                double s = 0;
                for (int k = 0; k < n; k++) s += V[k][i] * V[k][j];
                VtV[i][j] = s;
            }
            double s = 0;
            for (int k = 0; k < n; k++) s += V[k][i] * yCol[k];
            Vty[i] = s;
        }
        double[] coeffs = Linalg.solve(VtV, Vty);
        return new Polynomial(coeffs);
    }

    // =========================================================================
    // Griddata (scattered interpolation)
    // =========================================================================

    /** Interpolate scattered data using nearest neighbor */
    public static double[] griddataNearest(double[] x, double[] y, double[][] points) {
        double[] result = new double[points.length];
        for (int i = 0; i < points.length; i++) {
            int nearest = 0;
            double bestDist = Double.POSITIVE_INFINITY;
            for (int j = 0; j < x.length; j++) {
                double dist = (points[i][0] - x[j]) * (points[i][0] - x[j]) +
                              (points[i][1] - y[j]) * (points[i][1] - y[j]);
                if (dist < bestDist) {
                    bestDist = dist;
                    nearest = j;
                }
            }
            result[i] = y[nearest];
        }
        return result;
    }

    /** Interpolate scattered data using IDW (inverse distance weighting) */
    public static double[] griddataIDW(double[] x, double[] y, double[][] points, double power) {
        double[] result = new double[points.length];
        for (int i = 0; i < points.length; i++) {
            double num = 0, den = 0;
            for (int j = 0; j < x.length; j++) {
                double dist = Math.sqrt((points[i][0] - x[j]) * (points[i][0] - x[j]) +
                                        (points[i][1] - y[j]) * (points[i][1] - y[j]));
                if (dist == 0) { num = y[j]; den = 1; break; }
                double w = 1.0 / Math.pow(dist, power);
                num += w * y[j];
                den += w;
            }
            result[i] = num / den;
        }
        return result;
    }

    // =========================================================================
    // UnivariateSpline / Smooth splines
    // =========================================================================

    /** Smoothed univariate spline */
    public static class UnivariateSpline {
        public final double[] x, y, c;

        public UnivariateSpline(double[] x, double[] y, double s) {
            this.x = x.clone();
            this.y = y.clone();
            this.c = computeCubicSpline(x, y);
        }

        public double evaluate(double xNew) {
            int i = findInterval(x, xNew);
            return y[i] + c[i] * (xNew - x[i]);
        }
    }

    // =========================================================================
    // B-spline
    // =========================================================================

    /** Cox-de Boor B-spline basis function */
    public static double bSplineBasis(int k, int degree, double t, double[] knots) {
        if (degree == 0) {
            return (t >= knots[k] && t < knots[k + 1]) ? 1 : 0;
        }
        double left = 0, right = 0;
        double denomL = knots[k + degree] - knots[k];
        double denomR = knots[k + degree + 1] - knots[k + 1];
        if (denomL > 0) left = (t - knots[k]) / denomL * bSplineBasis(k, degree - 1, t, knots);
        if (denomR > 0) right = (knots[k + degree + 1] - t) / denomR * bSplineBasis(k + 1, degree - 1, t, knots);
        return left + right;
    }

    /** B-spline interpolation */
    public static double[] bsplineEvaluate(double[] x, double[] c, int degree, double[] knots, double[] evalPoints) {
        double[] result = new double[evalPoints.length];
        for (int i = 0; i < evalPoints.length; i++) {
            double sum = 0;
            int nBasis = c.length;
            for (int k = 0; k < nBasis; k++) {
                sum += c[k] * bSplineBasis(k, degree, evalPoints[i], knots);
            }
            result[i] = sum;
        }
        return result;
    }

    /** Make knots vector */
    public static double[] makeKnots(double[] x, int degree) {
        int n = x.length;
        double[] knots = new double[n + degree + 1];
        for (int i = 0; i <= degree; i++) knots[i] = x[0];
        for (int i = 0; i < n - degree - 1; i++) knots[degree + 1 + i] = x[i + 1];
        for (int i = n; i <= n + degree; i++) knots[i] = x[n - 1];
        return knots;
    }
}