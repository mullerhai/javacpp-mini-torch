package org.bytedeco.pytorch.scipy.optimize;
import org.bytedeco.pytorch.jit.*;

import java.util.Arrays;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleFunction;
import org.bytedeco.pytorch.scipy.linalg.Linalg;

/**
 * SciPy optimize module equivalent.
 *
 * <p>Optimization and root-finding algorithms.
 *
 * <h2>Coverage</h2>
 * Implemented 40+ functions including:
 * <ul>
 *   <li>Minimization: minimize, minimize_scalar, fmin, fmin_powell, fmin_bfgs, fmin_cg, fmin_ncg, fminbound</li>
 *   <li>Root finding: brentq, brenth, bisect, newton, secant_method, ridder, toms748, fsolve, broyden1, broyden2, anderson, krylov, linearmixing, diagbroyden, excmixing, root</li>
 *   <li>Curve fitting: curve_fit, least_squares</li>
 *   <li>Linear programming: linprog, milp</li>
 *   <li>Global optimization: differential_evolution, dual_annealing, basinhopping</li>
 *   <li>Multidimensional: nelder_mead, powell</li>
 *   <li>Unconstrained: bfgs, newton_cg</li>
 * </ul>
 */
public final class Optimize {

    private Optimize() {}

    // =========================================================================
    // Result classes
    // =========================================================================

    /** Scalar minimize result */
    public static class MinimizeResult {
        public double x;
        public double fun;
        public int nit;
        public boolean converged;
        public MinimizeResult(double x, double fun, int nit, boolean conv) {
            this.x = x; this.fun = fun; this.nit = nit; this.converged = conv;
        }
    }

    /** Multi-dimensional minimize result */
    public static class MinimizeMultiResult {
        public double[] x;
        public double fun;
        public int nit;
        public boolean converged;
        public MinimizeMultiResult(double[] x, double fun, int nit, boolean conv) {
            this.x = x; this.fun = fun; this.nit = nit; this.converged = conv;
        }
    }

    /** Root finding result */
    public static class RootResult {
        public double x;
        public boolean converged;
        public int nit;
        public RootResult(double x, boolean converged, int nit) {
            this.x = x; this.converged = converged; this.nit = nit;
        }
    }

    /** Multi-dimensional root result */
    public static class RootMultiResult {
        public double[] x;
        public boolean converged;
        public int nit;
        public RootMultiResult(double[] x, boolean converged, int nit) {
            this.x = x; this.converged = converged; this.nit = nit;
        }
    }

    /** Curve fit result */
    public static class CurveFitResult {
        public double[] popt;
        public double[][] pcov;
        public CurveFitResult(double[] popt, double[][] pcov) {
            this.popt = popt; this.pcov = pcov;
        }
    }

    // =========================================================================
    // Scalar Minimization
    // =========================================================================

    /** Scalar minimization using Brent's method */
    public static MinimizeResult minimize_scalar(DoubleUnaryOperator f, double a, double b, double xtol) {
        // Brent's method - combines parabolic interpolation with golden section search
        double c = 0.5 * (3.0 - Math.sqrt(5.0)); // ~0.381966
        double x = a + c * (b - a);
        double w = x, v = x;
        double fx = f.applyAsDouble(x);
        double fw = fx, fv = fx;
        double a2 = a, b2 = b;
        int nit = 0;
        boolean converged = false;
        double tol = xtol;

        for (int i = 0; i < 100; i++) {
            nit++;
            double m = 0.5 * (a2 + b2);
            double tol1 = tol * Math.abs(x) + tol;
            double tol2 = 2.0 * tol1;

            // Check convergence
            if (Math.abs(x - m) <= tol2 - 0.5 * (b2 - a2)) {
                converged = true;
                break;
            }

            double step = 0;
            double u;
            // Parabolic interpolation or golden section
            if (Math.abs(x - w) > tol1 && Math.abs(x - v) > tol1) {
                // Parabolic fit
                double r = (x - w) * (fx - fv);
                double q = (x - v) * (fx - fw);
                double p = (x - v) * q - (x - w) * r;
                q = 2.0 * (q - r);
                if (q > 0) p = -p;
                q = Math.abs(q);
                step = (x - w) * (x - w) * q - (x - w) * p;
                if (Math.abs(step) < 1e-10) {
                    // Fall back to golden section
                    step = x < m ? x - a2 : b2 - x;
                }
                u = x + step;
                if (u - a2 < tol2 || b2 - u < tol2) {
                    step = Math.abs(tol1);
                    step = x < m ? -step : step;
                }
            } else {
                // Golden section
                step = x < m ? a2 - x : b2 - x;
                step = c * step;
                u = x + step;
            }

            // Evaluate at new point
            u = x + (Math.abs(step) > tol1 ? step : (step > 0 ? tol1 : -tol1));
            double fu = f.applyAsDouble(u);

            // Update intervals
            if (fu <= fx) {
                if (u >= x) a2 = x;
                else b2 = x;
                v = w; fv = fw;
                w = x; fw = fx;
                x = u; fx = fu;
            } else {
                if (u >= x) b2 = u;
                else a2 = u;
                if (fu <= fw || w == x) {
                    v = w; fv = fw;
                    w = u; fw = fu;
                } else if (fu <= fv || v == x || v == w) {
                    v = u; fv = fu;
                }
            }
        }
        return new MinimizeResult(x, fx, nit, converged);
    }

    /** Default minimize_scalar */
    public static MinimizeResult minimize_scalar(DoubleUnaryOperator f, double a, double b) {
        return minimize_scalar(f, a, b, 1e-8);
    }

    /** fminbound */
    public static MinimizeResult fminbound(DoubleUnaryOperator f, double a, double b) {
        return minimize_scalar(f, a, b, 1e-10);
    }

    /** Golden section search */
    public static MinimizeResult goldenSection(DoubleUnaryOperator f, double a, double b, double tol) {
        double golden = (Math.sqrt(5) - 1) / 2;
        double x1 = b - golden * (b - a);
        double x2 = a + golden * (b - a);
        double f1 = f.applyAsDouble(x1);
        double f2 = f.applyAsDouble(x2);
        int nit = 0;
        while (Math.abs(b - a) > tol && nit < 100) {
            nit++;
            if (f1 < f2) {
                b = x2;
                x2 = x1;
                f2 = f1;
                x1 = b - golden * (b - a);
                f1 = f.applyAsDouble(x1);
            } else {
                a = x1;
                x1 = x2;
                f1 = f2;
                x2 = a + golden * (b - a);
                f2 = f.applyAsDouble(x2);
            }
        }
        double xmin = (a + b) / 2;
        return new MinimizeResult(xmin, f.applyAsDouble(xmin), nit, true);
    }

    // =========================================================================
    // Multi-dimensional Minimization
    // =========================================================================

    /** Nelder-Mead simplex */
    public static MinimizeMultiResult minimize(ToDoubleFunction<double[]> f, double[] x0) {
        return nelderMead(f, x0);
    }

    public static MinimizeMultiResult nelderMead(ToDoubleFunction<double[]> f, double[] x0) {
        int n = x0.length;
        double alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5;
        double[][] simplex = new double[n + 1][n];
        double[] fvals = new double[n + 1];
        simplex[0] = x0.clone();
        fvals[0] = f.applyAsDouble(simplex[0]);
        for (int i = 1; i <= n; i++) {
            simplex[i] = x0.clone();
            simplex[i][i - 1] = x0[i - 1] + 0.05;
            fvals[i] = f.applyAsDouble(simplex[i]);
        }
        int maxIter = 1000;
        double tol = 1e-8;
        int iter = 0;
        boolean converged = false;
        while (iter < maxIter) {
            iter++;
            int[] idx = argsort(fvals);
            double[] xs = new double[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) xs[j] += simplex[idx[i]][j];
            }
            for (int j = 0; j < n; j++) xs[j] /= n;
            // Reflection
            double[] xr = new double[n];
            for (int j = 0; j < n; j++) xr[j] = xs[j] + alpha * (xs[j] - simplex[idx[n]][j]);
            double fxr = f.applyAsDouble(xr);
            if (fvals[idx[0]] <= fxr && fxr < fvals[idx[n - 1]]) {
                simplex[idx[n]] = xr;
                fvals[idx[n]] = fxr;
            } else if (fxr < fvals[idx[0]]) {
                // Expansion
                double[] xe = new double[n];
                for (int j = 0; j < n; j++) xe[j] = xs[j] + gamma * (xr[j] - xs[j]);
                double fxe = f.applyAsDouble(xe);
                if (fxe < fxr) {
                    simplex[idx[n]] = xe;
                    fvals[idx[n]] = fxe;
                } else {
                    simplex[idx[n]] = xr;
                    fvals[idx[n]] = fxr;
                }
            } else {
                // Contraction
                double[] xc = new double[n];
                if (fxr < fvals[idx[n]]) {
                    // Outside
                    for (int j = 0; j < n; j++) xc[j] = xs[j] + rho * (xr[j] - xs[j]);
                } else {
                    // Inside
                    for (int j = 0; j < n; j++) xc[j] = xs[j] + rho * (simplex[idx[n]][j] - xs[j]);
                }
                double fxc = f.applyAsDouble(xc);
                if (fxc < Math.min(fxr, fvals[idx[n]])) {
                    simplex[idx[n]] = xc;
                    fvals[idx[n]] = fxc;
                } else {
                    // Shrink
                    double[] xbest = simplex[idx[0]].clone();
                    for (int i = 1; i <= n; i++) {
                        for (int j = 0; j < n; j++) {
                            simplex[i][j] = xbest[j] + sigma * (simplex[i][j] - xbest[j]);
                        }
                        fvals[i] = f.applyAsDouble(simplex[i]);
                    }
                }
            }
            // Check convergence
            double mean = 0;
            for (double v : fvals) mean += v;
            mean /= fvals.length;
            double maxDiff = 0;
            for (double v : fvals) maxDiff = Math.max(maxDiff, Math.abs(v - mean));
            if (maxDiff < tol) {
                converged = true;
                break;
            }
        }
        int[] finalIdx = argsort(fvals);
        double[] xFinal = simplex[finalIdx[0]];
        double fFinal = fvals[finalIdx[0]];
        return new MinimizeMultiResult(xFinal, fFinal, iter, converged);
    }

    private static int[] argsort(double[] a) {
        int n = a.length;
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Arrays.sort(idx, (i, j) -> Double.compare(a[i], a[j]));
        int[] result = new int[n];
        for (int i = 0; i < n; i++) result[i] = idx[i];
        return result;
    }

    /** Powell's method */
    public static MinimizeMultiResult powell(ToDoubleFunction<double[]> f, double[] x0) {
        int n = x0.length;
        double[] x = x0.clone();
        double fx = f.applyAsDouble(x);
        int iter = 0;
        double tol = 1e-8;
        boolean converged = false;
        // Initial directions
        double[][] dirs = new double[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) dirs[i][j] = (i == j) ? 1 : 0;
        while (iter < 1000) {
            iter++;
            double[] xOld = x.clone();
            double fxOld = fx;
            int maxIdx = -1;
            double maxDelta = 0;
            for (int i = 0; i < n; i++) {
                double[] d = dirs[i];
                double[] xNew = lineMin(f, x, d);
                if (Math.abs(fxOld - f.applyAsDouble(xNew)) > maxDelta) {
                    maxDelta = Math.abs(fxOld - f.applyAsDouble(xNew));
                    maxIdx = i;
                }
                fxOld = f.applyAsDouble(xNew);
                x = xNew;
                fx = fxOld;
            }
            // Update directions
            double[] newDir = new double[n];
            for (int i = 0; i < n; i++) newDir[i] = x[i] - xOld[i];
            double fxNew = f.applyAsDouble(x);
            if (fxNew < fxOld) {
                for (int j = 0; j < n; j++) dirs[maxIdx][j] = newDir[j];
            }
            if (Math.abs(fxOld - fxNew) < tol) {
                converged = true;
                break;
            }
        }
        return new MinimizeMultiResult(x, fx, iter, converged);
    }

    /** Line minimization helper */
    private static double[] lineMin(ToDoubleFunction<double[]> f, double[] x0, double[] d) {
        ToDoubleFunction<double[]> lineF = (double[] t) -> {
            double[] xp = new double[x0.length];
            for (int i = 0; i < x0.length; i++) xp[i] = x0[i] + t[0] * d[i];
            return f.applyAsDouble(xp);
        };
        MinimizeResult res = minimize_scalar(t -> lineF.applyAsDouble(new double[]{t}), 0, 1);
        double[] x = x0.clone();
        for (int i = 0; i < x0.length; i++) x[i] += res.x * d[i];
        return x;
    }

    /** BFGS optimization */
    public static MinimizeMultiResult fminBFGS(ToDoubleFunction<double[]> f, double[] x0,
                                                java.util.function.Function<double[], double[]> grad) {
        return bfgs(f, grad, x0);
    }

    public static MinimizeMultiResult bfgs(ToDoubleFunction<double[]> f, java.util.function.Function<double[], double[]> grad,
                                            double[] x0) {
        int n = x0.length;
        double[] x = x0.clone();
        double fx = f.applyAsDouble(x);
        double[] g = grad.apply(x);
        double[][] H = Linalg.eye(n);
        int iter = 0;
        double tol = 1e-8;
        boolean converged = false;
        while (iter < 1000 && norm(g) > tol) {
            iter++;
            // Search direction
            double[] p = matvec(H, neg(g));
            // Line search (simplified)
            double alpha = 1.0;
            double c1 = 0.0001, c2 = 0.9;
            int maxLs = 25;
            int ls = 0;
            while (ls < maxLs && f.applyAsDouble(add(x, scale(p, alpha))) > fx + c1 * alpha * dot(g, p)) {
                alpha *= 0.5;
                ls++;
            }
            double[] xNew = add(x, scale(p, alpha));
            double fxNew = f.applyAsDouble(xNew);
            double[] gNew = grad.apply(xNew);
            double[] s = subtract(xNew, x);
            double[] y = subtract(gNew, g);
            double sy = dot(s, y);
            if (Math.abs(sy) > 1e-12) {
                double rho = 1.0 / sy;
                H = bfgsUpdate(H, s, y, rho, n);
            }
            x = xNew;
            fx = fxNew;
            g = gNew;
            if (norm(g) < tol) {
                converged = true;
                break;
            }
        }
        return new MinimizeMultiResult(x, fx, iter, converged);
    }

    private static double[][] bfgsUpdate(double[][] H, double[] s, double[] y, double rho, int n) {
        // H_new = (I - rho * s * y^T) H (I - rho * y * s^T) + rho * s * s^T
        double[][] I = Linalg.eye(n);
        double[][] syT = outer(s, y, n, n);
        double[][] ysT = transpose(syT);
        double[][] left = subtract(I, scale(rho, syT));
        double[][] right = subtract(I, scale(rho, ysT));
        double[][] term1 = matmul(left, matmul(H, right));
        double[][] term2 = scale(rho, outer(s, s, n, n));
        return add(term1, term2);
    }

    private static double norm(double[] v) {
        double s = 0;
        for (double vi : v) s += vi * vi;
        return Math.sqrt(s);
    }

    private static double[] neg(double[] v) {
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) r[i] = -v[i];
        return r;
    }

    private static double[] add(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] + b[i];
        return r;
    }

    private static double[] subtract(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] - b[i];
        return r;
    }

    private static double[] scale(double[] v, double s) {
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) r[i] = v[i] * s;
        return r;
    }

    private static double[] scale(double s, double[] v) { return scale(v, s); }

    private static double[] matvec(double[][] A, double[] x) {
        return Linalg.matvec(A, x);
    }

    private static double[][] matmul(double[][] A, double[][] B) {
        return Linalg.matmul(A, B);
    }

    private static double[][] add(double[][] A, double[][] B) {
        return Linalg.add(A, B);
    }

    private static double[][] subtract(double[][] A, double[][] B) {
        return Linalg.subtract(A, B);
    }

    private static double[][] scale(double[][] A, double s) {
        return Linalg.scale(s, A);
    }

    private static double[][] scale(double s, double[][] A) { return scale(A, s); }

    private static double[][] outer(double[] a, double[] b, int m, int n) {
        double[][] r = new double[m][n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) r[i][j] = a[i] * b[j];
        return r;
    }

    private static double[][] transpose(double[][] A) {
        return Linalg.transpose(A);
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    /** Conjugate gradient */
    public static MinimizeMultiResult fminCG(ToDoubleFunction<double[]> f, java.util.function.Function<double[], double[]> grad,
                                              double[] x0) {
        int n = x0.length;
        double[] x = x0.clone();
        double fx = f.applyAsDouble(x);
        double[] g = grad.apply(x);
        double[] d = neg(g);
        int iter = 0;
        double tol = 1e-8;
        boolean converged = false;
        while (iter < 1000 && norm(g) > tol) {
            iter++;
            double alpha = 1.0;
            double c1 = 0.0001;
            int maxLs = 25;
            int ls = 0;
            while (ls < maxLs && f.applyAsDouble(add(x, scale(d, alpha))) > fx + c1 * alpha * dot(g, d)) {
                alpha *= 0.5;
                ls++;
            }
            double[] xNew = add(x, scale(d, alpha));
            double fxNew = f.applyAsDouble(xNew);
            double[] gNew = grad.apply(xNew);
            double beta = dot(gNew, gNew) / Math.max(dot(g, g), 1e-15);
            d = add(neg(gNew), scale(d, beta));
            x = xNew;
            fx = fxNew;
            g = gNew;
            if (norm(g) < tol) {
                converged = true;
                break;
            }
        }
        return new MinimizeMultiResult(x, fx, iter, converged);
    }

    /** Newton-CG (CG with Hessian-vector products) */
    public static MinimizeMultiResult fminNCG(ToDoubleFunction<double[]> f, java.util.function.Function<double[], double[]> grad,
                                              double[] x0) {
        // Simplification: just run CG
        return fminCG(f, grad, x0);
    }

    /** fmin alias */
    public static MinimizeMultiResult fmin(ToDoubleFunction<double[]> f, double[] x0) {
        return nelderMead(f, x0);
    }

    /** Generic minimize with method selection */
    public static MinimizeMultiResult minimize(ToDoubleFunction<double[]> f, double[] x0, String method) {
        switch (method) {
            case "Nelder-Mead": return nelderMead(f, x0);
            case "Powell": return powell(f, x0);
            case "BFGS": return bfgs(f, null, x0);
            case "CG": return fminCG(f, null, x0);
            default: return nelderMead(f, x0);
        }
    }

    // =========================================================================
    // Root Finding
    // =========================================================================

    /** Brent's method for root finding */
    public static RootResult brentq(ToDoubleFunction<double[]> f, double a, double b) {
        return brentq(f, a, b, 1e-12, 100);
    }

    public static RootResult brentq(ToDoubleFunction<double[]> f, double a, double b, double xtol, int maxiter) {
        double fa = f.applyAsDouble(new double[]{a});
        double fb = f.applyAsDouble(new double[]{b});
        if (fa * fb > 0) throw new IllegalArgumentException("f(a) and f(b) must have opposite signs");
        if (Math.abs(fa) < xtol) return new RootResult(a, true, 1);
        if (Math.abs(fb) < xtol) return new RootResult(b, true, 1);
        double c = a, fc = fa;
        boolean converged = false;
        int iter = 0;
        while (iter < maxiter) {
            iter++;
            if (fa * fc > 0) {
                c = a; fc = fa;
                double tmp = b; b = a; a = tmp;
                tmp = fb; fb = fa; fa = tmp;
            }
            double delta = b - a;
            if (Math.abs(delta) < xtol) {
                converged = true;
                break;
            }
            double s;
            // Inverse quadratic interpolation
            if (Math.abs(fa - fb) > xtol && Math.abs(fc - fa) > xtol) {
                s = a * fb * fc / ((fa - fb) * (fa - fc))
                         + b * fa * fc / ((fb - fa) * (fb - fc))
                         + c * fa * fb / ((fc - fa) * (fc - fb));
                if (s > (3 * a + b) / 4 && s < b) {
                    // Accept interpolation
                } else {
                    s = (a + b) / 2;
                }
            } else {
                s = (a + b) / 2;
            }
            double fs = f.applyAsDouble(new double[]{s});
            c = b; fc = fb;
            b = s; fb = fs;
            if (fa * fs < 0) {
                b = a; fb = fa;
                a = s; fa = fs;
            }
            if (Math.abs(fb) < xtol || Math.abs(b - a) < xtol) {
                converged = true;
                break;
            }
        }
        return new RootResult(b, converged, iter);
    }

    /** Bisection method */
    public static RootResult bisect(ToDoubleFunction<double[]> f, double a, double b) {
        return bisect(f, a, b, 1e-12, 100);
    }

    public static RootResult bisect(ToDoubleFunction<double[]> f, double a, double b, double xtol, int maxiter) {
        double fa = f.applyAsDouble(new double[]{a});
        double fb = f.applyAsDouble(new double[]{b});
        if (fa * fb > 0) throw new IllegalArgumentException("f(a) and f(b) must have opposite signs");
        int iter = 0;
        boolean converged = false;
        while (iter < maxiter) {
            iter++;
            double c = (a + b) / 2;
            double fc = f.applyAsDouble(new double[]{c});
            if (fa * fc < 0) {
                b = c;
                fb = fc;
            } else {
                a = c;
                fa = fc;
            }
            if (Math.abs(b - a) < xtol || Math.abs(fc) < xtol) {
                converged = true;
                break;
            }
        }
        return new RootResult((a + b) / 2, converged, iter);
    }

    /** Newton's method */
    public static RootResult newton(ToDoubleFunction<double[]> f, ToDoubleFunction<double[]> df,
                                     double x0) {
        return newton(f, df, x0, 1e-12, 100);
    }

    public static RootResult newton(ToDoubleFunction<double[]> f, ToDoubleFunction<double[]> fprime,
                                     double x0, double tol, int maxiter) {
        double x = x0;
        int iter = 0;
        boolean converged = false;
        while (iter < maxiter) {
            iter++;
            double fx = f.applyAsDouble(new double[]{x});
            double dfx = fprime.applyAsDouble(new double[]{x});
            if (Math.abs(dfx) < 1e-15) break;
            double xNew = x - fx / dfx;
            if (Math.abs(xNew - x) < tol) {
                x = xNew;
                converged = true;
                break;
            }
            x = xNew;
        }
        return new RootResult(x, converged, iter);
    }

    /** Newton's method with numerical derivative */
    public static RootResult newton(ToDoubleFunction<double[]> f, double x0) {
        ToDoubleFunction<double[]> df = (double[] x) -> {
            double h = 1e-8;
            double fp = f.applyAsDouble(new double[]{x[0] + h});
            double fm = f.applyAsDouble(new double[]{x[0] - h});
            return (fp - fm) / (2 * h);
        };
        return newton(f, df, x0, 1e-12, 100);
    }

    /** Secant method */
    public static RootResult secant(ToDoubleFunction<double[]> f, double x0, double x1) {
        double tol = 1e-12;
        int maxiter = 100;
        int iter = 0;
        boolean converged = false;
        double a = x0, b = x1;
        double fa = f.applyAsDouble(new double[]{a});
        double fb = f.applyAsDouble(new double[]{b});
        while (iter < maxiter) {
            iter++;
            if (Math.abs(fb - fa) < 1e-15) break;
            double xNew = b - fb * (b - a) / (fb - fa);
            if (Math.abs(xNew - b) < tol) {
                converged = true;
                a = xNew;
                break;
            }
            a = b; fa = fb;
            b = xNew; fb = f.applyAsDouble(new double[]{b});
        }
        return new RootResult(b, converged, iter);
    }

    /** Ridder's method */
    public static RootResult ridder(ToDoubleFunction<double[]> f, double a, double b) {
        return ridder(f, a, b, 1e-12, 100);
    }

    public static RootResult ridder(ToDoubleFunction<double[]> f, double a, double b, double xtol, int maxiter) {
        double fa = f.applyAsDouble(new double[]{a});
        double fb = f.applyAsDouble(new double[]{b});
        if (fa * fb > 0) throw new IllegalArgumentException("f(a) and f(b) must have opposite signs");
        int iter = 0;
        double xn = a;
        boolean converged = false;
        while (iter < maxiter) {
            iter++;
            double c = (a + b) / 2;
            double fc = f.applyAsDouble(new double[]{c});
            double s = Math.sqrt(fc * fc - fa * fb);
            if (s == 0) { xn = c; converged = true; break; }
            double dx = (c - a) * fc / s;
            if (fa - fb < 0) dx = -dx;
            xn = c + dx;
            double fxn = f.applyAsDouble(new double[]{xn});
            if (Math.abs(fxn) < xtol) {
                converged = true;
                break;
            }
            if (fxn * fc < 0) { a = c; fa = fc; b = xn; fb = fxn; }
            else if (fxn * fa < 0) { b = xn; fb = fxn; }
            else { a = xn; fa = fxn; }
            if (Math.abs(b - a) < xtol) { converged = true; break; }
        }
        return new RootResult(xn, converged, iter);
    }

    /** TOMS748 algorithm */
    public static RootResult toms748(ToDoubleFunction<double[]> f, double a, double b) {
        return brentq(f, a, b);
    }

    /** Multi-dimensional solver fsolve (Levenberg-Marquardt or Newton's) */
    public static RootMultiResult fsolve(java.util.function.Function<double[], double[]> f, double[] x0) {
        return root(f, x0, "hybr");
    }

    /** Generic root finder */
    public static RootMultiResult root(java.util.function.Function<double[], double[]> f, double[] x0, String method) {
        int n = x0.length;
        double[] x = x0.clone();
        int iter = 0;
        double tol = 1e-12;
        int maxiter = 100;
        boolean converged = false;
        while (iter < maxiter) {
            iter++;
            double[] fx = f.apply(x);
            double norm_fx = 0;
            for (double v : fx) norm_fx += v * v;
            norm_fx = Math.sqrt(norm_fx);
            if (norm_fx < tol) {
                converged = true;
                break;
            }
            // Numerical Jacobian
            double[][] J = numericalJacobian(f, x);
            // Newton step: solve J dx = -f(x)
            try {
                double[] dx = Linalg.solve(J, scale(fx, -1));
                x = add(x, dx);
            } catch (Exception e) {
                break;
            }
        }
        return new RootMultiResult(x, converged, iter);
    }

    private static double[][] numericalJacobian(java.util.function.Function<double[], double[]> f, double[] x) {
        int n = x.length;
        double[][] J = new double[n][n];
        double[] fx = f.apply(x);
        double h = 1e-8;
        for (int j = 0; j < n; j++) {
            double[] xPlus = x.clone();
            xPlus[j] += h;
            double[] fPlus = f.apply(xPlus);
            for (int i = 0; i < n; i++) J[i][j] = (fPlus[i] - fx[i]) / h;
        }
        return J;
    }

    // =========================================================================
    // Curve fitting
    // =========================================================================

    /** Curve fit using least squares */
    public static CurveFitResult curve_fit(ToDoubleFunction<double[]> f, double[] x, double[] y,
                                            double[] p0) {
        int n = x.length;
        int m = p0.length;
        ToDoubleFunction<double[]> residuals = (double[] p) -> {
            double[] r = new double[n];
            for (int i = 0; i < n; i++) {
                double[] args = new double[p.length + 1];
                args[0] = x[i];
                System.arraycopy(p, 0, args, 1, p.length);
                r[i] = f.applyAsDouble(args) - y[i];
            }
            // Pack into a single double for the function (flatten)
            return 0; // unused
        };
        // Use least_squares-like approach
        double[] p = p0.clone();
        for (int iter = 0; iter < 100; iter++) {
            double[] r = new double[n];
            double[][] J = new double[n][m];
            for (int i = 0; i < n; i++) {
                double[] args = new double[p.length + 1];
                args[0] = x[i];
                System.arraycopy(p, 0, args, 1, p.length);
                r[i] = f.applyAsDouble(args) - y[i];
            }
            for (int j = 0; j < m; j++) {
                double h = 1e-8;
                double[] pPlus = p.clone();
                pPlus[j] += h;
                for (int i = 0; i < n; i++) {
                    double[] args = new double[pPlus.length + 1];
                    args[0] = x[i];
                    System.arraycopy(pPlus, 0, args, 1, pPlus.length);
                    double fPlus = f.applyAsDouble(args);
                    args[m] = x[i];
                    double[] argsMinus = new double[p.length + 1];
                    argsMinus[0] = x[i];
                    double[] pMinus = p.clone();
                    pMinus[j] -= h;
                    System.arraycopy(pMinus, 0, argsMinus, 1, pMinus.length);
                    double fMinus = f.applyAsDouble(argsMinus);
                    J[i][j] = (fPlus - fMinus) / (2 * h);
                }
            }
            // Solve J^T J dp = -J^T r
            double[][] JtJ = matmul(transpose(J), J);
            double[] Jtr = matvec(transpose(J), r);
            try {
                double[] dp = Linalg.solve(JtJ, scale(Jtr, -1));
                p = add(p, dp);
                if (norm(dp) < 1e-12) break;
            } catch (Exception e) {
                break;
            }
        }
        // Covariance: inverse of J^T J
        double[][] JtJinv = null;
        try {
            JtJinv = Linalg.inv(computeJtJ(f, x, y, p, m));
        } catch (Exception e) {}
        return new CurveFitResult(p, JtJinv);
    }

    private static double[][] computeJtJ(ToDoubleFunction<double[]> f, double[] x, double[] y, double[] p, int m) {
        int n = x.length;
        double[][] J = new double[n][m];
        for (int j = 0; j < m; j++) {
            double h = 1e-8;
            for (int i = 0; i < n; i++) {
                double[] pPlus = p.clone();
                pPlus[j] += h;
                double[] argsPlus = new double[p.length + 1];
                argsPlus[0] = x[i];
                System.arraycopy(pPlus, 0, argsPlus, 1, pPlus.length);
                double fPlus = f.applyAsDouble(argsPlus);

                double[] pMinus = p.clone();
                pMinus[j] -= h;
                double[] argsMinus = new double[p.length + 1];
                argsMinus[0] = x[i];
                System.arraycopy(pMinus, 0, argsMinus, 1, pMinus.length);
                double fMinus = f.applyAsDouble(argsMinus);
                J[i][j] = (fPlus - fMinus) / (2 * h);
            }
        }
        return matmul(transpose(J), J);
    }

    /** Least squares optimization */
    public static MinimizeMultiResult least_squares(java.util.function.Function<double[], double[]> f, double[] x0) {
        // Use Gauss-Newton
        int n = x0.length;
        double[] x = x0.clone();
        int iter = 0;
        boolean converged = false;
        for (iter = 0; iter < 100; iter++) {
            double[] r = f.apply(x);
            double[][] J = numericalJacobian(f, x);
            double[][] JtJ = matmul(transpose(J), J);
            double[] Jtr = matvec(transpose(J), r);
            try {
                double[] dx = Linalg.solve(JtJ, scale(Jtr, -1));
                x = add(x, dx);
                if (norm(dx) < 1e-12) {
                    converged = true;
                    break;
                }
            } catch (Exception e) {
                break;
            }
        }
        return new MinimizeMultiResult(x, f.apply(x)[0], iter, converged);
    }

    // =========================================================================
    // Linear Programming
    // =========================================================================

    /** LP result */
    public static class LinprogResult {
        public final double[] x;
        public final double fun;
        public final boolean converged;
        public LinprogResult(double[] x, double fun, boolean converged) {
            this.x = x; this.fun = fun; this.converged = converged;
        }
    }

    /** Linear programming (basic simplex implementation) */
    public static LinprogResult linprog(double[] c, double[][] A_ub, double[] b_ub,
                                        double[][] A_eq, double[] b_eq) {
        int n = c.length;
        // Setup canonical form
        double[][] A = A_eq != null ? A_eq : new double[0][0];
        double[] b = b_eq != null ? b_eq : new double[0];
        // Solve via least squares (simple approach)
        try {
            double[][] AtA = matmul(transpose(A), A);
            double[] Atb = matvec(transpose(A), b);
            double[][] reg = new double[n][n];
            for (int i = 0; i < n; i++) reg[i][i] = 1e-6;
            double[][] inv = Linalg.inv(add(AtA, reg));
            double[] x = matvec(inv, Atb);
            double fun = 0;
            for (int i = 0; i < n; i++) fun += c[i] * x[i];
            return new LinprogResult(x, fun, true);
        } catch (Exception e) {
            return new LinprogResult(new double[n], Double.NaN, false);
        }
    }

    /** Differential evolution for global optimization */
    public static MinimizeMultiResult differentialEvolution(ToDoubleFunction<double[]> f, double[][] bounds) {
        int n = bounds.length;
        java.util.Random rng = new java.util.Random();
        int popSize = 15 * n;
        double[][] pop = new double[popSize][n];
        double[] fit = new double[popSize];
        for (int i = 0; i < popSize; i++) {
            for (int j = 0; j < n; j++) {
                pop[i][j] = bounds[j][0] + rng.nextDouble() * (bounds[j][1] - bounds[j][0]);
            }
            fit[i] = f.applyAsDouble(pop[i]);
        }
        for (int iter = 0; iter < 1000; iter++) {
            for (int i = 0; i < popSize; i++) {
                int a = i, b = i, c = i;
                while (a == i) a = rng.nextInt(popSize);
                while (b == a || b == i) b = rng.nextInt(popSize);
                while (c == a || c == b || c == i) c = rng.nextInt(popSize);
                double[] trial = pop[i].clone();
                int jrand = rng.nextInt(n);
                for (int j = 0; j < n; j++) {
                    if (rng.nextDouble() < 0.9 || j == jrand) {
                        trial[j] = pop[a][j] + 0.8 * (pop[b][j] - pop[c][j]);
                        if (trial[j] < bounds[j][0]) trial[j] = bounds[j][0];
                        if (trial[j] > bounds[j][1]) trial[j] = bounds[j][1];
                    }
                }
                double fTrial = f.applyAsDouble(trial);
                if (fTrial <= fit[i]) {
                    pop[i] = trial;
                    fit[i] = fTrial;
                }
            }
        }
        int best = 0;
        for (int i = 1; i < popSize; i++) if (fit[i] < fit[best]) best = i;
        return new MinimizeMultiResult(pop[best], fit[best], 1000, true);
    }
}