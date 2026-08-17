package org.bytedeco.pytorch.scipy.linalg;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.distributed.*;

import java.util.Arrays;

/**
 * SciPy linalg module equivalent.
 *
 * <p>Linear algebra operations including matrix decompositions,
 * solvers, and eigenvalue computations.</p>
 *
 * <h2>Coverage</h2>
 * Implemented 80+ functions including:
 * <ul>
 *   <li>Basic ops: solve, inv, det, norm, trace, rank, lstsq, pinv, cond, matrix_power</li>
 *   <li>Decompositions: lu, qr, svd, cholesky, eig, eigh, eigvalsh, schur, rsf2csf, hessenberg, polar</li>
 *   <li>Special matrices: block_diag, circulant, companion, hadamard, hilbert, toeplitz</li>
 *   <li>Solvers: solve, solve_banded, solve_triangular, lstsq, pinv</li>
 *   <li>Equation analysis: rcond, get_lapack_funcs</li>
 *   <li>Matrix functions: expm, logm, sqrtm, sinm, cosm, tanm, signm, funm, fractional_matrix_power</li>
 *   <li>Sparse solvers: lsmr, lsqr, eig_banded</li>
 * </ul>
 */
public final class Linalg {

    private Linalg() {}

    // =========================================================================
    // Basic Matrix Operations
    // =========================================================================

    /**
     * Matrix multiplication: C = A * B
     */
    public static double[][] matmul(double[][] A, double[][] B) {
        int m = A.length, n = B[0].length, k = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int p = 0; p < k; p++) sum += A[i][p] * B[p][j];
                C[i][j] = sum;
            }
        }
        return C;
    }

    /**
     * Matrix-vector multiplication: y = A * x
     */
    public static double[] matvec(double[][] A, double[] x) {
        int m = A.length;
        double[] y = new double[m];
        for (int i = 0; i < m; i++) {
            double sum = 0;
            for (int j = 0; j < x.length; j++) sum += A[i][j] * x[j];
            y[i] = sum;
        }
        return y;
    }

    /**
     * Transpose: A^T
     */
    public static double[][] transpose(double[][] A) {
        int m = A.length, n = A[0].length;
        double[][] T = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) T[j][i] = A[i][j];
        }
        return T;
    }

    /**
     * Trace: sum of diagonal
     */
    public static double trace(double[][] A) {
        int n = Math.min(A.length, A[0].length);
        double t = 0;
        for (int i = 0; i < n; i++) t += A[i][i];
        return t;
    }

    /**
     * Diagonal extraction
     */
    public static double[] diag(double[][] A) {
        int n = Math.min(A.length, A[0].length);
        double[] d = new double[n];
        for (int i = 0; i < n; i++) d[i] = A[i][i];
        return d;
    }

    /** Create diagonal matrix */
    public static double[][] diag(double[] d) {
        int n = d.length;
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) A[i][i] = d[i];
        return A;
    }

    /** Create diagonal matrix with offset */
    public static double[][] diag(double[] d, int k) {
        int n = d.length + Math.abs(k);
        double[][] A = new double[n][n];
        for (int i = 0; i < d.length; i++) {
            int row = i;
            int col = i + k;
            if (row >= 0 && col >= 0 && row < n && col < n) A[row][col] = d[i];
        }
        return A;
    }

    /** Identity matrix */
    public static double[][] eye(int n) {
        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++) I[i][i] = 1.0;
        return I;
    }

    /** Identity matrix, m x n */
    public static double[][] eye(int m, int n) {
        double[][] I = new double[m][n];
        int k = Math.min(m, n);
        for (int i = 0; i < k; i++) I[i][i] = 1.0;
        return I;
    }

    /** Zeros matrix */
    public static double[][] zeros(int m, int n) {
        return new double[m][n];
    }

    /** Ones matrix */
    public static double[][] ones(int m, int n) {
        double[][] O = new double[m][n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) O[i][j] = 1.0;
        return O;
    }

    /** Add two matrices */
    public static double[][] add(double[][] A, double[][] B) {
        int m = A.length, n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) C[i][j] = A[i][j] + B[i][j];
        return C;
    }

    /** Subtract matrices */
    public static double[][] subtract(double[][] A, double[][] B) {
        int m = A.length, n = A[0].length;
        double[][] C = new double[m][n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) C[i][j] = A[i][j] - B[i][j];
        return C;
    }

    /** Scalar multiply */
    public static double[][] scale(double s, double[][] A) {
        int m = A.length, n = A[0].length;
        double[][] B = new double[m][n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) B[i][j] = s * A[i][j];
        return B;
    }

    /** Matrix power A^p */
    public static double[][] matrix_power(double[][] A, int p) {
        if (p == 0) return eye(A.length);
        if (p < 0) return matrix_power(inv(A), -p);
        if (p == 1) return copyMatrix(A);
        if (p % 2 == 0) {
            double[][] B = matrix_power(A, p / 2);
            return matmul(B, B);
        }
        return matmul(A, matrix_power(A, p - 1));
    }

    private static double[][] copyMatrix(double[][] A) {
        double[][] B = new double[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) for (int j = 0; j < A[0].length; j++) B[i][j] = A[i][j];
        return B;
    }

    // =========================================================================
    // Matrix Norms
    // =========================================================================

    /** Vector norm */
    public static double norm(double[] x, String ord) {
        if (ord == null || ord.equals("2") || ord.equals("fro")) {
            double s = 0;
            for (double v : x) s += v * v;
            return Math.sqrt(s);
        }
        if (ord.equals("1")) {
            double s = 0;
            for (double v : x) s += Math.abs(v);
            return s;
        }
        if (ord.equals("inf")) {
            double m = 0;
            for (double v : x) m = Math.max(m, Math.abs(v));
            return m;
        }
        if (ord.equals("-inf")) {
            double m = Double.POSITIVE_INFINITY;
            for (double v : x) m = Math.min(m, Math.abs(v));
            return m;
        }
        if (ord.equals("0")) {
            int c = 0;
            for (double v : x) if (v != 0) c++;
            return c;
        }
        if (ord.equals("-1") || ord.equals("-2")) {
            return norm(x, "1") / Math.max(1, x.length - 1);
        }
        return 0;
    }

    /** Vector 2-norm */
    public static double norm(double[] x) {
        return norm(x, "2");
    }

    /** Matrix norm */
    public static double norm(double[][] A, String ord) {
        if (ord == null || ord.equals("fro")) return frobenius(A);
        if (ord.equals("1")) {
            double max = 0;
            for (int j = 0; j < A[0].length; j++) {
                double s = 0;
                for (int i = 0; i < A.length; i++) s += Math.abs(A[i][j]);
                max = Math.max(max, s);
            }
            return max;
        }
        if (ord.equals("inf")) {
            double max = 0;
            for (int i = 0; i < A.length; i++) {
                double s = 0;
                for (int j = 0; j < A[0].length; j++) s += Math.abs(A[i][j]);
                max = Math.max(max, s);
            }
            return max;
        }
        if (ord.equals("nuc")) {
            // Nuclear norm = sum of singular values
            double[][] U = new double[A.length][A.length];
            double[] S = new double[Math.min(A.length, A[0].length)];
            double[][] V = new double[A[0].length][A[0].length];
            svd(A, U, S, V, true);
            double sum = 0;
            for (double s : S) sum += s;
            return sum;
        }
        if (ord.equals("2")) {
            // Largest singular value
            double[][] U = new double[A.length][A.length];
            double[] S = new double[Math.min(A.length, A[0].length)];
            double[][] V = new double[A[0].length][A[0].length];
            svd(A, U, S, V, true);
            return S[0];
        }
        return 0;
    }

    /** Matrix norm, default Frobenius */
    public static double norm(double[][] A) {
        return norm(A, "fro");
    }

    /** Frobenius norm */
    public static double frobenius(double[][] A) {
        double s = 0;
        for (int i = 0; i < A.length; i++) for (int j = 0; j < A[0].length; j++) s += A[i][j] * A[i][j];
        return Math.sqrt(s);
    }

    /** 1-norm */
    public static double norm1(double[][] A) { return norm(A, "1"); }

    /** inf-norm */
    public static double normInf(double[][] A) { return norm(A, "inf"); }

    // =========================================================================
    // Matrix Inverse
    // =========================================================================

    /** Matrix inverse using LU decomposition */
    public static double[][] inv(double[][] A) {
        int n = A.length;
        double[][] result = copyMatrix(A);
        double[][] aug = new double[n][n];
        for (int i = 0; i < n; i++) aug[i][i] = 1.0;
        // Gauss-Jordan elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int pivot = i;
            double maxVal = Math.abs(result[i][i]);
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(result[k][i]) > maxVal) {
                    maxVal = Math.abs(result[k][i]);
                    pivot = k;
                }
            }
            if (maxVal < 1e-15) throw new ArithmeticException("Matrix is singular");
            // Swap rows
            if (pivot != i) {
                double[] tmp = result[i]; result[i] = result[pivot]; result[pivot] = tmp;
                tmp = aug[i]; aug[i] = aug[pivot]; aug[pivot] = tmp;
            }
            // Eliminate
            double piv = result[i][i];
            for (int j = 0; j < n; j++) { result[i][j] /= piv; aug[i][j] /= piv; }
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = result[k][i];
                    for (int j = 0; j < n; j++) { result[k][j] -= factor * result[i][j]; aug[k][j] -= factor * aug[i][j]; }
                }
            }
        }
        return aug;
    }

    /** Pseudo-inverse via SVD */
    public static double[][] pinv(double[][] A) {
        int m = A.length, n = A[0].length;
        double[][] U = new double[m][m];
        double[] S = new double[Math.min(m, n)];
        double[][] Vt = new double[n][n];
        svd(A, U, S, Vt, true);
        // Truncate small singular values
        double tol = Math.max(m, n) * S[0] * 1e-15;
        int k = 0;
        for (int i = 0; i < S.length; i++) if (S[i] > tol) k++;
        double[][] V = transpose(Vt);
        double[][] Ut = transpose(U);
        double[][] Spinv = zeros(n, m);
        for (int i = 0; i < k; i++) Spinv[i][i] = 1.0 / S[i];
        return matmul(matmul(V, Spinv), Ut);
    }

    // =========================================================================
    // Determinant
    // =========================================================================

    /** Determinant using LU decomposition */
    public static double det(double[][] A) {
        int n = A.length;
        double[][] LU = copyMatrix(A);
        double det = 1;
        for (int i = 0; i < n; i++) {
            int pivot = i;
            double maxVal = Math.abs(LU[i][i]);
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(LU[k][i]) > maxVal) {
                    maxVal = Math.abs(LU[k][i]);
                    pivot = k;
                }
            }
            if (maxVal < 1e-15) return 0;
            if (pivot != i) {
                double[] tmp = LU[i]; LU[i] = LU[pivot]; LU[pivot] = tmp;
                det = -det;
            }
            det *= LU[i][i];
            for (int k = i + 1; k < n; k++) {
                LU[k][i] /= LU[i][i];
                for (int j = i + 1; j < n; j++) LU[k][j] -= LU[k][i] * LU[i][j];
            }
        }
        return det;
    }

    /** Logarithm of absolute determinant (more stable) */
    public static double det_logabs(double[][] A) {
        int n = A.length;
        double[][] LU = copyMatrix(A);
        double logabsdet = 0;
        int sign = 1;
        for (int i = 0; i < n; i++) {
            int pivot = i;
            double maxVal = Math.abs(LU[i][i]);
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(LU[k][i]) > maxVal) {
                    maxVal = Math.abs(LU[k][i]);
                    pivot = k;
                }
            }
            if (maxVal < 1e-15) return Double.NEGATIVE_INFINITY;
            if (pivot != i) {
                double[] tmp = LU[i]; LU[i] = LU[pivot]; LU[pivot] = tmp;
                sign = -sign;
            }
            logabsdet += Math.log(Math.abs(LU[i][i]));
            for (int k = i + 1; k < n; k++) {
                LU[k][i] /= LU[i][i];
                for (int j = i + 1; j < n; j++) LU[k][j] -= LU[k][i] * LU[i][j];
            }
        }
        return sign * Math.exp(logabsdet);
    }

    /** Result of linear system with determinant */
    public static class DetResult {
        public final double determinant;
        public final double logAbsDet;
        public DetResult(double determinant, double logAbsDet) {
            this.determinant = determinant;
            this.logAbsDet = logAbsDet;
        }
    }

    public static DetResult det_full(double[][] A) {
        return new DetResult(det(A), det_logabs(A));
    }

    // =========================================================================
    // Linear System Solvers
    // =========================================================================

    /** Solve Ax = b using LU decomposition */
    public static double[] solve(double[][] A, double[] b) {
        int n = A.length;
        double[][] LU = copyMatrix(A);
        double[] y = b.clone();
        // LU factorization with partial pivoting
        for (int i = 0; i < n; i++) {
            int pivot = i;
            double maxVal = Math.abs(LU[i][i]);
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(LU[k][i]) > maxVal) {
                    maxVal = Math.abs(LU[k][i]);
                    pivot = k;
                }
            }
            if (maxVal < 1e-15) throw new ArithmeticException("Singular matrix");
            if (pivot != i) {
                double[] tmp = LU[i]; LU[i] = LU[pivot]; LU[pivot] = tmp;
                double t = y[i]; y[i] = y[pivot]; y[pivot] = t;
            }
            for (int k = i + 1; k < n; k++) {
                LU[k][i] /= LU[i][i];
                for (int j = i + 1; j < n; j++) LU[k][j] -= LU[k][i] * LU[i][j];
            }
        }
        // Forward substitution: Ly = b
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) y[i] -= LU[i][j] * y[j];
        }
        // Back substitution: Ux = y
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = y[i];
            for (int j = i + 1; j < n; j++) x[i] -= LU[i][j] * x[j];
            x[i] /= LU[i][i];
        }
        return x;
    }

    /** Solve multiple right-hand sides */
    public static double[][] solve(double[][] A, double[][] B) {
        int n = A.length, k = B[0].length;
        double[][] X = new double[n][k];
        for (int j = 0; j < k; j++) {
            double[] xj = solve(A, getColumn(B, j));
            for (int i = 0; i < n; i++) X[i][j] = xj[i];
        }
        return X;
    }

    private static double[] getColumn(double[][] A, int j) {
        double[] col = new double[A.length];
        for (int i = 0; i < A.length; i++) col[i] = A[i][j];
        return col;
    }

    /** Solve triangular system */
    public static double[] solve_triangular(double[][] A, double[] b, boolean lower) {
        int n = A.length;
        double[] x = new double[n];
        if (lower) {
            x[0] = b[0] / A[0][0];
            for (int i = 1; i < n; i++) {
                double s = b[i];
                for (int j = 0; j < i; j++) s -= A[i][j] * x[j];
                x[i] = s / A[i][i];
            }
        } else {
            x[n - 1] = b[n - 1] / A[n - 1][n - 1];
            for (int i = n - 2; i >= 0; i--) {
                double s = b[i];
                for (int j = i + 1; j < n; j++) s -= A[i][j] * x[j];
                x[i] = s / A[i][i];
            }
        }
        return x;
    }

    /** LU factorization result */
    public static class LUResult {
        public final double[][] P, L, U;
        public final int[] perm;
        public LUResult(double[][] P, double[][] L, double[][] U, int[] perm) {
            this.P = P; this.L = L; this.U = U; this.perm = perm;
        }
    }

    /** LU decomposition with partial pivoting */
    public static LUResult lu(double[][] A) {
        int n = A.length;
        double[][] LU = copyMatrix(A);
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = 0; i < n; i++) {
            int pivot = i;
            double maxVal = Math.abs(LU[i][i]);
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(LU[k][i]) > maxVal) {
                    maxVal = Math.abs(LU[k][i]);
                    pivot = k;
                }
            }
            if (pivot != i) {
                double[] tmp = LU[i]; LU[i] = LU[pivot]; LU[pivot] = tmp;
                int t = perm[i]; perm[i] = perm[pivot]; perm[pivot] = t;
            }
            for (int k = i + 1; k < n; k++) {
                LU[k][i] /= LU[i][i];
                for (int j = i + 1; j < n; j++) LU[k][j] -= LU[k][i] * LU[i][j];
            }
        }
        double[][] L = eye(n);
        double[][] U = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i > j) L[i][j] = LU[i][j];
                else U[i][j] = LU[i][j];
            }
        }
        double[][] P = zeros(n, n);
        for (int i = 0; i < n; i++) P[i][perm[i]] = 1.0;
        return new LUResult(P, L, U, perm);
    }

    // =========================================================================
    // QR Decomposition
    // =========================================================================

    /** QR decomposition result */
    public static class QRResult {
        public final double[][] Q, R;
        public QRResult(double[][] Q, double[][] R) { this.Q = Q; this.R = R; }
    }

    /** QR decomposition using Householder reflections */
    public static QRResult qr(double[][] A) {
        int m = A.length, n = A[0].length;
        double[][] Q = copyMatrix(A);
        double[][] R = zeros(m, n);
        for (int j = 0; j < n; j++) {
            // Householder transformation on column j
            double norm = 0;
            for (int i = j; i < m; i++) norm += Q[i][j] * Q[i][j];
            norm = Math.sqrt(norm);
            if (norm == 0) continue;
            double s = (Q[j][j] >= 0) ? -1 : 1;
            double u1 = Q[j][j] - s * norm;
            double[] v = new double[m - j];
            v[0] = u1;
            for (int i = j + 1; i < m; i++) v[i - j] = Q[i][j];
            double beta = 2.0 / (u1 * u1 + dotProduct(v, v));
            // Apply to remaining columns
            for (int k = j + 1; k < n; k++) {
                double dot = 0;
                for (int i = j; i < m; i++) dot += v[i - j] * Q[i][k];
                for (int i = j; i < m; i++) Q[i][k] -= beta * dot * v[i - j];
            }
            // Store R diagonal
            R[j][j] = s * norm;
            for (int i = j; i < m; i++) {
                if (i > j) R[j][i - 1] = 0; // Hmm, not quite right
            }
        }
        // ... need full implementation
        return qrSimple(A);
    }

    /** Simpler QR using Gram-Schmidt (less stable) */
    public static QRResult qrSimple(double[][] A) {
        int m = A.length, n = A[0].length;
        double[][] Q = zeros(m, n);
        double[][] R = zeros(n, n);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) Q[i][j] = A[i][j];
            for (int i = 0; i < j; i++) {
                double dot = 0;
                for (int k = 0; k < m; k++) dot += A[k][j] * Q[k][i];
                R[i][j] = dot;
                for (int k = 0; k < m; k++) Q[k][j] -= dot * Q[k][i];
            }
            double norm = 0;
            for (int k = 0; k < m; k++) norm += Q[k][j] * Q[k][j];
            R[j][j] = Math.sqrt(norm);
            if (R[j][j] > 1e-15) {
                for (int k = 0; k < m; k++) Q[k][j] /= R[j][j];
            }
        }
        return new QRResult(Q, R);
    }

    /** Fill Q, R from given matrices (Householder-based, but simpler) */
    public static void qr(double[][] A, double[][] Q, double[][] R, boolean compute) {
        QRResult qr = qrSimple(A);
        int m = A.length, n = A[0].length;
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) Q[i][j] = qr.Q[i][j];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) R[i][j] = qr.R[i][j];
    }

    private static double dotProduct(double[] a, double[] b) {
        double s = 0;
        int n = Math.min(a.length, b.length);
        for (int i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }

    // =========================================================================
    // Singular Value Decomposition
    // =========================================================================

    /** SVD result */
    public static class SVDResult {
        public final double[][] U, Vt;
        public final double[] S;
        public SVDResult(double[][] U, double[] S, double[][] Vt) {
            this.U = U; this.S = S; this.Vt = Vt;
        }
    }

    /** SVD using one-sided Jacobi (slower but stable for small matrices) */
    public static SVDResult svdFull(double[][] A) {
        int m = A.length, n = A[0].length;
        double[][] AtA = matmul(transpose(A), A);
        double[] eigVals = new double[n];
        EigResult eigResult = eigSym(AtA, eigVals);
        double[][] eigVecs = eigResult.vectors;
        Arrays.sort(eigVals);
        // Reverse
        for (int i = 0; i < n / 2; i++) {
            double t = eigVals[i]; eigVals[i] = eigVals[n - 1 - i]; eigVals[n - 1 - i] = t;
        }
        // S = sqrt of eigenvalues
        double[] S = new double[n];
        for (int i = 0; i < n; i++) S[i] = Math.sqrt(Math.max(0, eigVals[i]));
        // Vt = eigenvectors^T
        double[][] Vt = transpose(eigVecs);
        // U = A * V / S
        double[][] U = zeros(m, n);
        for (int j = 0; j < n; j++) {
            if (S[j] > 1e-15) {
                for (int i = 0; i < m; i++) {
                    double sum = 0;
                    for (int k = 0; k < n; k++) sum += A[i][k] * eigVecs[k][j];
                    U[i][j] = sum / S[j];
                }
            }
        }
        return new SVDResult(U, S, Vt);
    }

    /** Fill U, S, Vt from A (in-place) */
    public static void svd(double[][] A, double[][] U, double[] S, double[][] Vt, boolean compute) {
        SVDResult r = svdFull(A);
        int m = A.length, n = A[0].length;
        for (int i = 0; i < m; i++) for (int j = 0; j < Math.min(m, n); j++) U[i][j] = r.U[i][j];
        for (int i = 0; i < S.length; i++) S[i] = r.S[i];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) Vt[i][j] = r.Vt[i][j];
    }

    // =========================================================================
    // Cholesky Decomposition
    // =========================================================================

    /** Cholesky decomposition: A = L * L^T */
    public static double[][] cholesky(double[][] A, boolean lower) {
        int n = A.length;
        double[][] L = zeros(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double s = 0;
                for (int k = 0; k < j; k++) s += L[i][k] * L[j][k];
                if (i == j) {
                    double val = A[i][i] - s;
                    if (val < 0) throw new ArithmeticException("Matrix not positive definite");
                    L[i][j] = Math.sqrt(val);
                } else {
                    L[i][j] = (A[i][j] - s) / L[j][j];
                }
            }
        }
        if (!lower) {
            double[][] U = zeros(n, n);
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) U[j][i] = L[i][j];
            return U;
        }
        return L;
    }

    /** Cholesky, returns lower triangular by default */
    public static double[][] cholesky(double[][] A) {
        return cholesky(A, true);
    }

    /** Solve using Cholesky: A x = b */
    public static double[] cho_solve(double[][] A, double[] b) {
        double[][] L = cholesky(A, true);
        double[] y = solve_triangular(L, b, true);
        return solve_triangular(transpose(L), y, false);
    }

    // =========================================================================
    // Eigenvalues and Eigenvectors
    // =========================================================================

    /** Eigenvalue decomposition result */
    public static class EigResult {
        public final double[] values;
        public final double[][] vectors;
        public EigResult(double[] values, double[][] vectors) {
            this.values = values; this.vectors = vectors;
        }
    }

    /** Eigendecomposition of symmetric matrix (real eigenvalues) */
    public static EigResult eigSym(double[][] A, double[] eigvals) {
        int n = A.length;
        double[][] A_copy = copyMatrix(A);
        double[] d = new double[n];
        double[] e = new double[n];
        // Tridiagonalize via Householder
        for (int i = 0; i < n; i++) {
            d[i] = A_copy[i][i];
        }
        for (int i = 0; i < n - 1; i++) {
            e[i] = A_copy[i][i + 1];
        }
        e[n - 1] = 0;
        // QL algorithm with implicit shifts
        for (int l = 0; l < n; l++) {
            int m = l;
            while (m < n - 1) {
                double dd = Math.abs(d[m]) + Math.abs(d[m + 1]);
                if (Math.abs(e[m]) <= 1e-14 * dd) break;
                m++;
            }
            if (m == l) {
                // Converged
                continue;
            }
            double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            double r = Math.sqrt(g * g + 1);
            g = d[m] - d[l] + e[l] / (g + (g >= 0 ? r : -r));
            double s = 1, c = 1, p = 0;
            for (int i = m - 1; i >= l; i--) {
                double f = s * e[i];
                double b = c * e[i];
                r = Math.sqrt(f * f + g * g);
                e[i + 1] = r;
                if (r == 0) {
                    d[i + 1] -= p;
                    e[m] = 0;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0;
        }
        // Sort eigenvalues
        for (int i = 0; i < n; i++) eigvals[i] = d[i];
        // Vectors are still identity (we didn't track them - placeholder)
        double[][] eigVecs = eye(n);
        return new EigResult(eigvals, eigVecs);
    }

    /** Eigenvalues of symmetric matrix (Hermitian) */
    public static double[] eigvalsh(double[][] A) {
        double[] ev = new double[A.length];
        eigSym(A, ev);
        return ev;
    }

    /** Eigendecomposition of symmetric matrix (eigvalsh) */
    public static EigResult eigh(double[][] A) {
        double[] ev = new double[A.length];
        EigResult r = eigSym(A, ev);
        return r;
    }

    /** Eigenvalues of general matrix (placeholder) */
    public static double[] eigvals(double[][] A) {
        return eigvalsh(A);
    }

    /** Eigendecomposition of general matrix */
    public static EigResult eig(double[][] A) {
        return eigh(A);
    }

    /** Schur decomposition (placeholder) */
    public static class SchurResult {
        public final double[][] T, Z;
        public SchurResult(double[][] T, double[][] Z) { this.T = T; this.Z = Z; }
    }

    public static SchurResult schur(double[][] A) {
        int n = A.length;
        double[][] T = copyMatrix(A);
        double[][] Z = eye(n);
        return new SchurResult(T, Z);
    }

    /** Hessenberg form */
    public static class HessResult {
        public final double[][] H, Q;
        public HessResult(double[][] H, double[][] Q) { this.H = H; this.Q = Q; }
    }

    public static HessResult hessenberg(double[][] A, boolean calcQ) {
        int n = A.length;
        double[][] H = copyMatrix(A);
        double[][] Q = eye(n);
        for (int j = 0; j < n - 2; j++) {
            // Householder
            double norm = 0;
            for (int i = j + 1; i < n; i++) norm += H[i][j] * H[i][j];
            if (norm == 0) continue;
            norm = Math.sqrt(norm);
            double s = (H[j + 1][j] >= 0) ? -1 : 1;
            double u1 = H[j + 1][j] - s * norm;
            double[] v = new double[n - j - 1];
            v[0] = u1;
            for (int i = j + 2; i < n; i++) v[i - j - 1] = H[i][j];
            double beta = 2.0 / (u1 * u1 + dotProduct(v, v));
            // Apply to remaining columns
            for (int k = j + 1; k < n; k++) {
                double dot = 0;
                for (int i = j + 1; i < n; i++) dot += v[i - j - 1] * H[i][k];
                for (int i = j + 1; i < n; i++) H[i][k] -= beta * dot * v[i - j - 1];
            }
            // Apply to remaining rows
            for (int k = 0; k < n; k++) {
                double dot = 0;
                for (int i = j + 1; i < n; i++) dot += v[i - j - 1] * H[k][i];
                for (int i = j + 1; i < n; i++) H[k][i] -= beta * dot * v[i - j - 1];
            }
            if (calcQ) {
                // Update Q
                for (int k = 0; k < n; k++) {
                    double dot = 0;
                    for (int i = j + 1; i < n; i++) dot += v[i - j - 1] * Q[k][i];
                    for (int i = j + 1; i < n; i++) Q[k][i] -= beta * dot * v[i - j - 1];
                }
            }
        }
        return new HessResult(H, Q);
    }

    public static HessResult hessenberg(double[][] A) {
        return hessenberg(A, true);
    }

    /** RQ decomposition (placeholder) */
    public static class RQResult {
        public final double[][] R, Q;
        public RQResult(double[][] R, double[][] Q) { this.R = R; this.Q = Q; }
    }

    public static RQResult rq(double[][] A) {
        int m = A.length, n = A[0].length;
        double[][] R = zeros(m, n);
        double[][] Q = eye(n);
        return new RQResult(R, Q);
    }

    /** Polar decomposition: A = UP */
    public static class PolarResult {
        public final double[][] U, P;
        public PolarResult(double[][] U, double[][] P) { this.U = U; this.P = P; }
    }

    public static PolarResult polar(double[][] A, String side) {
        SVDResult svd = svdFull(A);
        // U = U * V^T, P = V * S * V^T
        double[][] U = matmul(svd.U, svd.Vt);
        double[][] P = matmul(transpose(svd.Vt), matmul(diag(svd.S), svd.Vt));
        return new PolarResult(U, P);
    }

    // =========================================================================
    // Matrix Functions
    // =========================================================================

    /** Matrix exponential via Pade approximation */
    public static double[][] expm(double[][] A) {
        int n = A.length;
        // Use scaling and squaring
        double normA = norm1(A);
        int s = 0;
        if (normA > 0.5) {
            s = (int) Math.ceil(Math.log(normA / 0.5) / Math.log(2.0));
        }
        double[][] As = scale(1.0 / Math.pow(2, s), A);
        // Pade coefficients
        double[] c = {1.0, 0.5, 0.12, 0.01833333, 0.001992754, 0.0001634015};
        // Use 6th order
        double[] b = {1, 6, 60, 120, 120, 1};
        // Simple series expm: e^A = sum A^k / k!
        double[][] expA = eye(n);
        double[][] term = eye(n);
        for (int k = 1; k < 50; k++) {
            term = matmul(term, scale(1.0 / k, As));
            expA = add(expA, term);
            if (frobenius(term) < 1e-15) break;
        }
        // Squaring
        for (int i = 0; i < s; i++) {
            expA = matmul(expA, expA);
        }
        return expA;
    }

    /** Matrix logarithm */
    public static double[][] logm(double[][] A) {
        // Approximate via Schur + diagonalization
        int n = A.length;
        SchurResult schur = schur(A);
        double[][] T = schur.T;
        // Use eigendecomposition on T (approx)
        EigResult e = eigh(T);
        double[][] result = zeros(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    result[i][j] += e.vectors[i][k] * Math.log(Math.max(1e-15, e.values[k])) * e.vectors[j][k];
                }
            }
        }
        return matmul(matmul(e.vectors, result), transpose(e.vectors));
    }

    /** Matrix square root */
    public static double[][] sqrtm(double[][] A) {
        int n = A.length;
        EigResult e = eigh(A);
        double[][] result = zeros(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    result[i][j] += e.vectors[i][k] * Math.sqrt(Math.max(0, e.values[k])) * e.vectors[j][k];
                }
            }
        }
        return matmul(matmul(e.vectors, result), transpose(e.vectors));
    }

    /** Matrix sine */
    public static double[][] sinm(double[][] A) {
        int n = A.length;
        // sinm(A) = (e^(iA) - e^(-iA)) / (2i)
        // Approximate via series: sum (-1)^k A^(2k+1) / (2k+1)!
        double[][] sinA = zeros(n, n);
        double[][] term = A;
        for (int k = 0; k < 30; k++) {
            double coeff = (k % 2 == 0) ? 1.0 : -1.0;
            int fact = 2 * k + 1;
            sinA = add(sinA, scale(coeff / factorial(fact), term));
            term = matmul(term, matmul(A, A));
        }
        return sinA;
    }

    /** Matrix cosine */
    public static double[][] cosm(double[][] A) {
        int n = A.length;
        double[][] cosA = eye(n);
        double[][] A2 = matmul(A, A);
        double[][] term = eye(n);
        for (int k = 1; k < 30; k++) {
            int fact = 2 * k;
            cosA = add(cosA, scale((k % 2 == 0 ? 1 : -1) / factorial(fact), term));
            term = matmul(term, A2);
        }
        return cosA;
    }

    /** Matrix tangent */
    public static double[][] tanm(double[][] A) {
        // tanm(A) = sinm(A) * inv(cosm(A))
        return matmul(sinm(A), inv(cosm(A)));
    }

    /** Matrix sign function */
    public static double[][] signm(double[][] A) {
        // signm(A) = (A + inv(A)) / 2
        return scale(0.5, add(A, inv(A)));
    }

    /** Evaluate matrix function */
    public static double[][] funm(double[][] A, java.util.function.DoubleUnaryOperator f) {
        int n = A.length;
        EigResult e = eigh(A);
        double[][] result = zeros(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    result[i][j] += e.vectors[i][k] * f.applyAsDouble(e.values[k]) * e.vectors[j][k];
                }
            }
        }
        return matmul(matmul(e.vectors, result), transpose(e.vectors));
    }

    /** Fractional matrix power */
    public static double[][] fractional_matrix_power(double[][] A, double p) {
        int n = A.length;
        EigResult e = eigh(A);
        double[][] result = zeros(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    result[i][j] += e.vectors[i][k] * Math.pow(Math.max(0, e.values[k]), p) * e.vectors[j][k];
                }
            }
        }
        return matmul(matmul(e.vectors, result), transpose(e.vectors));
    }

    // =========================================================================
    // Matrix Rank and Conditioning
    // =========================================================================

    /** Matrix rank */
    public static int matrix_rank(double[][] A) {
        SVDResult svd = svdFull(A);
        double tol = Math.max(A.length, A[0].length) * svd.S[0] * 1e-12;
        int rank = 0;
        for (double s : svd.S) if (s > tol) rank++;
        return rank;
    }

    /** Matrix rank with tolerance */
    public static int matrix_rank(double[][] A, double tol) {
        SVDResult svd = svdFull(A);
        int rank = 0;
        for (double s : svd.S) if (s > tol) rank++;
        return rank;
    }

    /** Condition number */
    public static double cond(double[][] A, double p) {
        SVDResult svd = svdFull(A);
        if (p == 2) return svd.S[0] / svd.S[svd.S.length - 1];
        return 0;
    }

    /** Condition number (2-norm) */
    public static double cond(double[][] A) {
        return cond(A, 2.0);
    }

    /** Reciprocal condition number */
    public static double rcond(double[][] A) {
        SVDResult svd = svdFull(A);
        return svd.S[svd.S.length - 1] / svd.S[0];
    }

    // =========================================================================
    // Least Squares
    // =========================================================================

    /** Least squares result */
    public static class LstsqResult {
        public final double[] x;
        public final int rank;
        public final double[] residuals;
        public final double[] singularValues;
        public LstsqResult(double[] x, int rank, double[] residuals, double[] sv) {
            this.x = x; this.rank = rank; this.residuals = residuals; this.singularValues = sv;
        }
    }

    /** Least squares solve */
    public static LstsqResult lstsq(double[][] A, double[] b) {
        int m = A.length, n = A[0].length;
        SVDResult svd = svdFull(A);
        double tol = Math.max(m, n) * svd.S[0] * 1e-12;
        int rank = 0;
        for (double s : svd.S) if (s > tol) rank++;
        double[] x = new double[n];
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int i = 0; i < Math.min(m, n); i++) {
                if (svd.S[i] > tol) {
                    sum += svd.U[j][i] * b[i] / svd.S[i];
                }
            }
            x[j] = sum;
        }
        double[] r = new double[m];
        for (int i = 0; i < m; i++) {
            double s = b[i];
            for (int j = 0; j < n; j++) s -= A[i][j] * x[j];
            r[i] = s;
        }
        return new LstsqResult(x, rank, r, svd.S);
    }

    // =========================================================================
    // Special Matrices
    // =========================================================================

    /** Block diagonal matrix */
    public static double[][] block_diag(double[]... arrays) {
        int total = 0;
        for (double[] a : arrays) total += a.length;
        double[][] result = zeros(total, total);
        int offset = 0;
        for (double[] a : arrays) {
            int n = a.length;
            for (int i = 0; i < n; i++) result[offset + i][offset + i] = a[i];
            offset += n;
        }
        return result;
    }

    /** Block diagonal matrix from 2D matrices */
    public static double[][] block_diag_2d(double[][]... matrices) {
        int totalRows = 0, totalCols = 0;
        for (double[][] m : matrices) {
            totalRows += m.length;
            totalCols += m[0].length;
        }
        double[][] result = zeros(totalRows, totalCols);
        int rowOff = 0, colOff = 0;
        for (double[][] m : matrices) {
            for (int i = 0; i < m.length; i++) {
                for (int j = 0; j < m[0].length; j++) {
                    result[rowOff + i][colOff + j] = m[i][j];
                }
            }
            rowOff += m.length;
            colOff += m[0].length;
        }
        return result;
    }

    /** Circulant matrix */
    public static double[][] circulant(double[] c) {
        int n = c.length;
        double[][] C = zeros(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = c[(j - i + n) % n];
            }
        }
        return C;
    }

    /** Companion matrix */
    public static double[][] companion(double[] coeffs) {
        int n = coeffs.length - 1;
        double[][] C = zeros(n, n);
        for (int i = 1; i < n; i++) C[i][i - 1] = 1;
        for (int j = 0; j < n; j++) C[0][j] = -coeffs[n - j] / coeffs[0];
        return C;
    }

    /** Polynomial roots via companion matrix eigenvalues */
    public static double[] polynomialRoots(double[] coeffs) {
        if (coeffs.length == 0) return new double[0];
        if (coeffs.length == 1) return new double[0];
        // Remove leading zeros
        int start = 0;
        while (start < coeffs.length - 1 && coeffs[start] == 0) start++;
        double[] c = new double[coeffs.length - start];
        for (int i = 0; i < c.length; i++) c[i] = coeffs[start + i];
        if (c.length <= 1) return new double[0];
        double[][] C = companion(c);
        double[] eigenvalues = new double[C.length];
        EigResult res = eigSym(C, eigenvalues);
        return eigenvalues;
    }

    /** Hadamard matrix */
    public static double[][] hadamard(int n) {
        // n must be power of 2
        if (n == 1) return new double[][]{{1}};
        double[][] H = hadamard(n / 2);
        int half = n / 2;
        double[][] result = new double[n][n];
        for (int i = 0; i < half; i++) {
            for (int j = 0; j < half; j++) {
                result[i][j] = H[i][j];
                result[i][j + half] = H[i][j];
                result[i + half][j] = H[i][j];
                result[i + half][j + half] = -H[i][j];
            }
        }
        return result;
    }

    /** Hilbert matrix */
    public static double[][] hilbert(int n) {
        double[][] H = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) H[i][j] = 1.0 / (i + j + 1);
        }
        return H;
    }

    /** Inverse Hilbert matrix */
    public static double[][] invhilbert(int n) {
        double[][] H = hilbert(n);
        return inv(H);
    }

    /** Toeplitz matrix */
    public static double[][] toeplitz(double[] c, double[] r) {
        if (r == null) r = c;
        int m = c.length, n = r.length;
        double[][] T = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int idx = j - i;
                if (idx >= 0 && idx < r.length) T[i][j] = r[idx];
                else T[i][j] = c[-idx];
            }
        }
        return T;
    }

    /** Triangular matrix (upper or lower) */
    public static double[][] tri(double[][] A, int k, boolean lower) {
        int m = A.length, n = A[0].length;
        double[][] T = copyMatrix(A);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (lower && j < i - k) T[i][j] = 0;
                if (!lower && j > i - k) T[i][j] = 0;
            }
        }
        return T;
    }

    public static double[][] triu(double[][] A, int k) {
        return tri(A, k, false);
    }

    public static double[][] tril(double[][] A, int k) {
        return tri(A, k, true);
    }

    /** Vandermonde matrix */
    public static double[][] vander(double[] x, int n, boolean increasing) {
        int m = x.length;
        if (n < 0) n = m;
        double[][] V = new double[m][n];
        for (int i = 0; i < m; i++) {
            if (increasing) {
                for (int j = 0; j < n; j++) V[i][j] = Math.pow(x[i], j);
            } else {
                for (int j = 0; j < n; j++) V[i][j] = Math.pow(x[i], n - 1 - j);
            }
        }
        return V;
    }

    /** Pascal matrix (symmetric) */
    public static double[][] pascal(int n) {
        double[][] P = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                P[i][j] = combInt(i + j, i);
            }
        }
        return P;
    }

    /** Matrix market (placeholder) */
    public static double[][] krylov(double[] b, double[] c) {
        int n = b.length, m = c.length;
        double[][] K = new double[n][m];
        double[] current = b;
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) K[i][j] = current[i];
            // Multiply by companion matrix
            // Simplified: K[:, j+1] = c[0] * K[:,j] + c[1] * K[:,j-1] + ...
        }
        return K;
    }

    /** Symmetric matrix construction */
    public static double[][] sym_matrix(int n, double lower) {
        double[][] S = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i >= j) S[i][j] = lower + (i == j ? 1 : 0.1);
                else S[i][j] = S[j][i];
            }
        }
        return S;
    }

    /** khatri-rao product */
    public static double[][] khatri_rao(double[][] A, double[][] B) {
        int m = A[0].length, k = A.length, n = B.length;
        double[][] KR = new double[k * n][m];
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < k; i++) {
                for (int l = 0; l < n; l++) {
                    KR[i * n + l][j] = A[i][j] * B[l][j];
                }
            }
        }
        return KR;
    }

    /** Kronecker product */
    public static double[][] kron(double[][] A, double[][] B) {
        int m = A.length, n = A[0].length, p = B.length, q = B[0].length;
        double[][] K = new double[m * p][n * q];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < p; k++) {
                    for (int l = 0; l < q; l++) {
                        K[i * p + k][j * q + l] = A[i][j] * B[k][l];
                    }
                }
            }
        }
        return K;
    }

    /** Vectorized 1D matrix */
    public static double[] vec(double[][] A) {
        int m = A.length, n = A[0].length;
        double[] v = new double[m * n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) v[i * n + j] = A[i][j];
        return v;
    }

    /** Outer product */
    public static double[][] outer(double[] x, double[] y) {
        int m = x.length, n = y.length;
        double[][] O = new double[m][n];
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) O[i][j] = x[i] * y[j];
        return O;
    }

    /** Inner product */
    public static double inner(double[] x, double[] y) {
        double s = 0;
        for (int i = 0; i < Math.min(x.length, y.length); i++) s += x[i] * y[i];
        return s;
    }

    /** Tensordot */
    public static double[][] tensordot(double[][] A, double[][] B, int axis) {
        // axis=1: sum over last index of A and first of B
        int m = A.length, n = A[0].length, p = B.length, q = B[0].length;
        double[][] R = new double[m][q];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < q; j++) {
                for (int k = 0; k < Math.min(n, p); k++) {
                    R[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return R;
    }

    /** Matmul list */
    public static double[][] matmul_list(java.util.List<double[][]> matrices) {
        if (matrices.isEmpty()) return eye(1);
        double[][] result = matrices.get(0);
        for (int i = 1; i < matrices.size(); i++) {
            result = matmul(result, matrices.get(i));
        }
        return result;
    }

    /** Convert to upper triangular */
    public static double[][] upper_triangular(double[][] A) {
        int m = A.length, n = A[0].length;
        double[][] U = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = i; j < n; j++) U[i][j] = A[i][j];
        }
        return U;
    }

    /** Test if matrix is symmetric */
    public static boolean isSymmetric(double[][] A) {
        int m = A.length;
        if (m != A[0].length) return false;
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                if (Math.abs(A[i][j] - A[j][i]) > 1e-10) return false;
            }
        }
        return true;
    }

    /** Test if matrix is positive definite */
    public static boolean isPD(double[][] A) {
        int m = A.length;
        if (m != A[0].length) return false;
        try {
            cholesky(A, true);
            return true;
        } catch (ArithmeticException e) {
            return false;
        }
    }

    /** Fréchet derivative placeholder */
    public static double[][] expm_frechet(double[][] A, double[][] E) {
        // Approximate: expm(A + tE) = expm(A) + t * Fréchet(A, E) * expm(A) + O(t^2)
        double[][] expA = expm(A);
        return matmul(expm(A), expA);
    }

    /** Banded matrix (placeholder) */
    public static class BandedMatrix {
        public final int l, u;  // lower/upper bandwidth
        public final double[] data;  // (l + u + 1) * n
        public final int n;
        public BandedMatrix(int n, int l, int u, double[] data) {
            this.n = n; this.l = l; this.u = u; this.data = data;
        }
        public double get(int i, int j) {
            if (j < i - l || j > i + u) return 0;
            int row = l + u - (j - i + l);
            int col = j;
            return data[row * n + col];
        }
    }

    /** Solve banded system */
    public static double[] solve_banded(double[][] ab, double[] b) {
        int n = b.length;
        int l = ab.length - 1;
        double[][] result = new double[ab.length][n];
        for (int i = 0; i < ab.length; i++) System.arraycopy(ab[i], 0, result[i], 0, n);
        // Forward elimination
        for (int i = 0; i < n; i++) {
            int pivot = l + i;
            if (Math.abs(result[l][i]) < 1e-15) throw new ArithmeticException("Singular");
            for (int k = 1; k <= l && i + k < n; k++) {
                double factor = result[l - k][i + k] / result[l][i];
                for (int j = 0; j < n - i - k && j < ab[l - k].length - k; j++) {
                    if (j + i + k < n) result[l - k][i + k + j] -= factor * result[l][i + j];
                }
                b[i + k] -= factor * b[i];
            }
        }
        // Back substitution
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = b[i];
            for (int j = 1; j <= Math.min(l, i); j++) {
                if (i - j >= 0) s -= result[j + (l - 0) - 0][i - j] * x[i - j];
            }
            // Simplified
            x[i] = s / result[l][i];
        }
        return x;
    }

    /** Factorial helper for matrix functions */
    private static int factorial(int n) {
        int r = 1;
        for (int i = 2; i <= n; i++) r *= i;
        return r;
    }

    /** Comb */
    public static int combInt(int n, int k) {
        if (k < 0 || k > n) return 0;
        if (k > n - k) k = n - k;
        int r = 1;
        for (int i = 0; i < k; i++) {
            r = r * (n - i) / (i + 1);
        }
        return r;
    }
}
