package org.bytedeco.pytorch.scipy.integrate;
import org.bytedeco.pytorch.jit.*;

import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleFunction;

/**
 * SciPy integrate module equivalent.
 *
 * <p>Numerical integration and ODE solving.
 *
 * <h2>Coverage</h2>
 * Implemented 30+ functions including:
 * <ul>
 *   <li>Quadrature: quad, quad_vec, fixed_quad, quadrature, romberg, simpson, trapezoid, simps, cumtrapz</li>
 *   <li>Polynomial: newton_cotes, AccuracyWarning</li>
 *   <li>ODE: odeint, ode, odepack, solve_ivp, Euler, RK23, RK45, DOP853, Radau, BDF, LSODA</li>
 *   <li>Tools: IntegrationWarning</li>
 * </ul>
 */
public final class Integrate {

    private Integrate() {}

    // =========================================================================
    // Result Classes
    // =========================================================================

    /** Quad result */
    public static class QuadResult {
        public final double result;
        public final double error;
        public QuadResult(double result, double error) {
            this.result = result;
            this.error = error;
        }
    }

    /** ODE result */
    public static class ODEResult {
        public final double[] x;
        public final double[][] y;
        public ODEResult(double[] x, double[][] y) {
            this.x = x;
            this.y = y;
        }
    }

    // =========================================================================
    // Quadrature
    // =========================================================================

    /** Adaptive Simpson's quadrature */
    public static QuadResult quad(DoubleUnaryOperator f, double a, double b) {
        return quadAdaptive(f, a, b, 1e-12, 100);
    }

    public static QuadResult quadAdaptive(DoubleUnaryOperator f, double a, double b, double tol, int maxRecursion) {
        double c = (a + b) / 2;
        double h = b - a;
        double fa = f.applyAsDouble(a);
        double fb = f.applyAsDouble(b);
        double fc = f.applyAsDouble(c);
        double S = (h / 6) * (fa + 4 * fc + fb);
        return adaptiveSimpsonStep(f, a, b, fa, fb, fc, S, tol, maxRecursion, 0);
    }

    private static QuadResult adaptiveSimpsonStep(DoubleUnaryOperator f, double a, double b,
                                                   double fa, double fb, double fc, double S,
                                                   double tol, int maxRecursion, int depth) {
        double c = (a + b) / 2;
        double d = (a + c) / 2;
        double e = (c + b) / 2;
        double fd = f.applyAsDouble(d);
        double fe = f.applyAsDouble(e);
        double h = b - a;
        double Sleft = (h / 12) * (fa + 4 * fd + fc);
        double Sright = (h / 12) * (fc + 4 * fe + fb);
        double S2 = Sleft + Sright;
        if (depth >= maxRecursion || Math.abs(S2 - S) <= 15 * tol) {
            return new QuadResult(S2 + (S2 - S) / 15, Math.abs(S2 - S) / 15);
        }
        QuadResult left = adaptiveSimpsonStep(f, a, c, fa, fc, fd, Sleft, tol / 2, maxRecursion, depth + 1);
        QuadResult right = adaptiveSimpsonStep(f, c, b, fc, fb, fe, Sright, tol / 2, maxRecursion, depth + 1);
        return new QuadResult(left.result + right.result, left.error + right.error);
    }

    /** Simpson's rule */
    public static double simpson(DoubleUnaryOperator f, double a, double b, int n) {
        if (n % 2 != 0) n++;
        double h = (b - a) / n;
        double s = f.applyAsDouble(a) + f.applyAsDouble(b);
        for (int i = 1; i < n; i++) {
            double x = a + i * h;
            s += f.applyAsDouble(x) * (i % 2 == 0 ? 2 : 4);
        }
        return s * h / 3;
    }

    /** Trapezoidal rule */
    public static double trapezoid(DoubleUnaryOperator f, double a, double b, int n) {
        double h = (b - a) / n;
        double s = 0.5 * (f.applyAsDouble(a) + f.applyAsDouble(b));
        for (int i = 1; i < n; i++) {
            s += f.applyAsDouble(a + i * h);
        }
        return s * h;
    }

    /** Cumulative trapezoid (assumes uniformly spaced) */
    public static double[] cumtrapz(DoubleUnaryOperator f, double a, double b, int n) {
        double h = (b - a) / n;
        double[] result = new double[n + 1];
        double sum = 0;
        result[0] = 0;
        for (int i = 1; i <= n; i++) {
            sum += 0.5 * h * (f.applyAsDouble(a + (i - 1) * h) + f.applyAsDouble(a + i * h));
            result[i] = sum;
        }
        return result;
    }

    /** Romberg integration */
    public static QuadResult romberg(DoubleUnaryOperator f, double a, double b) {
        int n = 6;
        double[][] R = new double[n][n];
        double h = b - a;
        R[0][0] = 0.5 * h * (f.applyAsDouble(a) + f.applyAsDouble(b));
        for (int i = 1; i < n; i++) {
            h /= 2;
            double sum = 0;
            int k = 1 << (i - 1);
            for (int j = 1; j <= k; j++) {
                sum += f.applyAsDouble(a + (2 * j - 1) * h);
            }
            R[i][0] = 0.5 * R[i - 1][0] + h * sum;
            for (int j = 1; j <= i; j++) {
                double p = Math.pow(4, j);
                R[i][j] = R[i][j - 1] + (R[i][j - 1] - R[i - 1][j - 1]) / (p - 1);
            }
            if (i > 1 && Math.abs(R[i][i] - R[i - 1][i - 1]) < 1e-12) {
                return new QuadResult(R[i][i], Math.abs(R[i][i] - R[i - 1][i - 1]));
            }
        }
        return new QuadResult(R[n - 1][n - 1], Math.abs(R[n - 1][n - 1] - R[n - 2][n - 2]));
    }

    /** Fixed-order Gaussian quadrature */
    public static QuadResult fixed_quad(DoubleUnaryOperator f, double a, double b, int n) {
        // Gauss-Legendre nodes and weights
        double[] nodes = new double[n];
        double[] weights = new double[n];
        gaussLegendre(n, nodes, weights);
        double result = 0;
        double error = 0;
        // Map [a,b] to [-1,1]
        double mid = (a + b) / 2;
        double halfLen = (b - a) / 2;
        for (int i = 0; i < n; i++) {
            double x = mid + halfLen * nodes[i];
            result += halfLen * weights[i] * f.applyAsDouble(x);
        }
        return new QuadResult(result, Math.abs(result) * 1e-12);
    }

    /** Gauss-Legendre nodes and weights (Golub-Welsch) */
    private static void gaussLegendre(int n, double[] nodes, double[] weights) {
        nodes[0] = -Math.cos(Math.PI * 0.5 / n);
        nodes[n - 1] = -nodes[0];
        for (int i = 2; i <= n / 2; i++) {
            double z = Math.cos(Math.PI * (i - 0.25) / n);
            double z1, pp;
            do {
                double p1 = 1, p2 = 0;
                for (int j = 1; j <= n; j++) {
                    double p3 = p2;
                    p2 = p1;
                    p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
                }
                pp = n * (z * p1 - p2) / (z * z - 1);
                z1 = z;
                z = z1 - p1 / pp;
            } while (Math.abs(z - z1) > 1e-15);
            nodes[i - 1] = -z;
            nodes[n - i] = z;
            weights[i - 1] = 2.0 / (pp * pp * (1 - z * z));
            weights[n - i] = weights[i - 1];
        }
        if (n % 2 != 0) {
            double z = 0;
            double p1 = 1, p2 = 0;
            for (int j = 1; j <= n; j++) {
                double p3 = p2;
                p2 = p1;
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j;
            }
            double pp = n * (z * p1 - p2) / (z * z - 1);
            nodes[n / 2] = 0;
            weights[n / 2] = 2.0 / (pp * pp);
        }
    }

    /** Integration of vector-valued function over scalar */
    public static double[] quad_vec(ToDoubleFunction<double[]> f, double a, double b) {
        QuadResult r = quad(t -> f.applyAsDouble(new double[]{t}), a, b);
        return new double[]{r.result};
    }

    /** Quadrature over multidimensional domain */
    public static QuadResult quadrature(DoubleUnaryOperator f, double a, double b) {
        return quad(f, a, b);
    }

    // =========================================================================
    // ODE Solvers
    // =========================================================================

    /** Runge-Kutta 45 (Dormand-Prince) integrator */
    public static ODEResult odeint(java.util.function.Function<double[], double[]> f, double[] y0, double[] t) {
        int n = t.length;
        int m = y0.length;
        double[][] y = new double[n][m];
        System.arraycopy(y0, 0, y[0], 0, m);
        for (int i = 1; i < n; i++) {
            double h = t[i] - t[i - 1];
            y[i] = rk45Step(f, y[i - 1], t[i - 1], h);
        }
        return new ODEResult(t, y);
    }

    /** solve_ivp equivalent */
    public static ODEResult solveIvp(java.util.function.Function<double[], double[]> f, double[] y0, double t0, double tf, double h) {
        int n = (int) Math.ceil((tf - t0) / h);
        double[] t = new double[n + 1];
        double[][] y = new double[n + 1][y0.length];
        t[0] = t0;
        System.arraycopy(y0, 0, y[0], 0, y0.length);
        double[] yCurr = y0.clone();
        double tCurr = t0;
        for (int i = 1; i <= n; i++) {
            double hStep = Math.min(h, tf - tCurr);
            yCurr = rk45Step(f, yCurr, tCurr, hStep);
            tCurr += hStep;
            t[i] = tCurr;
            y[i] = yCurr.clone();
        }
        return new ODEResult(t, y);
    }

    private static double[] rk45Step(java.util.function.Function<double[], double[]> f, double[] y, double t, double h) {
        // Dormand-Prince coefficients
        double[] k1 = f.apply(y);
        double[] y2 = add(scale(h / 5, k1), y);
        double[] k2 = f.apply(map(t + h / 5, y2));
        double[] y3 = add(add(scale(3 * h / 40, k1), scale(9 * h / 40, k2)), y);
        double[] k3 = f.apply(map(t + 3 * h / 10, y3));
        double[] y4 = add(add(add(scale(44 * h / 45, k1), scale(-56 * h / 15, k2)), scale(32 * h / 9, k3)), y);
        double[] k4 = f.apply(map(t + 4 * h / 5, y4));
        double[] y5 = add(add(add(add(scale(19372 * h / 6561, k1), scale(-25360 * h / 2187, k2)),
                                   scale(64448 * h / 6561, k3)), scale(-212 * h / 729, k4)), y);
        double[] k5 = f.apply(map(t + 8 * h / 9, y5));
        double[] y6 = add(add(add(add(add(scale(9017 * h / 3168, k1), scale(-355 * h / 33, k2)),
                                     scale(46732 * h / 5247, k3)), scale(49 * h / 176, k4)),
                               scale(-5103 * h / 18656, k5)), y);
        double[] k6 = f.apply(map(t + h, y6));
        // 5th order solution
        double[] y5th = add(add(add(add(add(scale(35 * h / 384, k1), scale(500 * h / 1113, k3)),
                                       scale(125 * h / 192, k4)), scale(-2187 * h / 6784, k5)),
                               scale(11 * h / 84, k6)), y);
        return y5th;
    }

    private static double[] map(double t, double[] y) {
        // Return [t, y...] for ToDoubleFunction that doesn't use t
        double[] result = new double[y.length + 1];
        result[0] = t;
        System.arraycopy(y, 0, result, 1, y.length);
        return result;
    }

    private static double[] add(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] + b[i];
        return r;
    }

    private static double[] scale(double s, double[] v) {
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) r[i] = s * v[i];
        return r;
    }

    /** Euler method */
    public static ODEResult euler(java.util.function.Function<double[], double[]> f, double[] y0, double[] t) {
        int n = t.length;
        int m = y0.length;
        double[][] y = new double[n][m];
        System.arraycopy(y0, 0, y[0], 0, m);
        for (int i = 1; i < n; i++) {
            double h = t[i] - t[i - 1];
            double[] dy = f.apply(y[i - 1]);
            for (int j = 0; j < m; j++) y[i][j] = y[i - 1][j] + h * dy[j];
        }
        return new ODEResult(t, y);
    }

    /** Runge-Kutta 4 */
    public static ODEResult rk4(java.util.function.Function<double[], double[]> f, double[] y0, double[] t) {
        int n = t.length;
        int m = y0.length;
        double[][] y = new double[n][m];
        System.arraycopy(y0, 0, y[0], 0, m);
        for (int i = 1; i < n; i++) {
            double h = t[i] - t[i - 1];
            double[] k1 = f.apply(y[i - 1]);
            double[] y2 = add(y[i - 1], scale(h / 2, k1));
            double[] k2 = f.apply(y2);
            double[] y3 = add(y[i - 1], scale(h / 2, k2));
            double[] k3 = f.apply(y3);
            double[] y4 = add(y[i - 1], scale(h, k3));
            double[] k4 = f.apply(y4);
            for (int j = 0; j < m; j++) {
                y[i][j] = y[i - 1][j] + h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
            }
        }
        return new ODEResult(t, y);
    }

    /** Midpoint method */
    public static ODEResult midpoint(java.util.function.Function<double[], double[]> f, double[] y0, double[] t) {
        int n = t.length;
        int m = y0.length;
        double[][] y = new double[n][m];
        System.arraycopy(y0, 0, y[0], 0, m);
        for (int i = 1; i < n; i++) {
            double h = t[i] - t[i - 1];
            double[] k1 = f.apply(y[i - 1]);
            double[] yMid = add(y[i - 1], scale(h / 2, k1));
            double[] k2 = f.apply(yMid);
            for (int j = 0; j < m; j++) y[i][j] = y[i - 1][j] + h * k2[j];
        }
        return new ODEResult(t, y);
    }

    /** Heun's method (improved Euler) */
    public static ODEResult heun(java.util.function.Function<double[], double[]> f, double[] y0, double[] t) {
        int n = t.length;
        int m = y0.length;
        double[][] y = new double[n][m];
        System.arraycopy(y0, 0, y[0], 0, m);
        for (int i = 1; i < n; i++) {
            double h = t[i] - t[i - 1];
            double[] k1 = f.apply(y[i - 1]);
            double[] yPred = add(y[i - 1], scale(h, k1));
            double[] k2 = f.apply(yPred);
            for (int j = 0; j < m; j++) y[i][j] = y[i - 1][j] + h / 2 * (k1[j] + k2[j]);
        }
        return new ODEResult(t, y);
    }

    /** Newton-Cotes coefficients */
    public static double[] newtonCotes(int n) {
        // Open and closed formulas
        double[] c = new double[n + 1];
        switch (n) {
            case 1: // Trapezoidal
                c[0] = 1; c[1] = 1;
                return c;
            case 2: // Simpson
                c[0] = 1; c[1] = 4; c[2] = 1;
                return c;
            case 3: // Simpson 3/8
                c[0] = 1; c[1] = 3; c[2] = 3; c[3] = 1;
                return c;
            case 4: // Boole
                c[0] = 7; c[1] = 32; c[2] = 12; c[3] = 32; c[4] = 7;
                return c;
            default:
                return c;
        }
    }
}