/**
 * SciPy4J: Enterprise-Grade SciPy Implementation for Java.
 *
 * <p>SciPy4J is a pure-Java implementation of Python's SciPy scientific computing stack,
 * targeting feature parity with SciPy v1.18. It leverages existing JavaCPP-PyTorch
 * infrastructure (Tensor, DataFrame, distribution/, plot/, graphx/) to provide:</p>
 *
 * <h2>Submodules</h2>
 * <ul>
 *   <li>{@link org.bytedeco.pytorch.scipy.special} — 200+ special functions
 *       (Bessel, gamma, beta, error functions, hypergeometric, elliptic, orthogonal polynomials)</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.stats} — Full statistical distributions,
 *       hypothesis tests, entropy, correlation, KDE</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.optimize} — Root-finding, linear/nonlinear
 *       programming, curve fitting, least squares</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.integrate} — Numerical integration
 *       (quad, romberg, quadrature) and ODE solvers</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.interpolate} — 1D/2D/ND interpolation,
 *       splines (CubicSpline, UnivariateSpline, BarycentricInterpolator)</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.linalg} — Dense/sparse linear algebra,
 *       matrix decomposition, eigenvalues</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.fft} — Discrete Fourier transforms</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.signal} — Signal processing
 *       (filter design, convolution, spectral analysis)</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.spatial} — Spatial algorithms
 *       (Delaunay, ConvexHull, KDTree, distance metrics)</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.sparse} — Sparse matrix representations
 *       (CSR, CSC, COO, LIL, DIA, BSR)</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.ndimage} — N-dimensional image processing</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.cluster} — Hierarchical clustering, k-means</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.constants} — Physical and mathematical constants</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.datasets} — Load built-in datasets</li>
 *   <li>{@link org.bytedeco.pytorch.scipy.io} — Scientific data format I/O</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * import static org.bytedeco.pytorch.scipy.SciPy.*;
 *
 * // Special functions
 * double result = SciPy.special().erf(0.5);
 *
 * // Statistics
 * double pValue = SciPy.stats().ttest(x, y);
 *
 * // Optimization
 * MinimizeResult r = SciPy.optimize().minimize(func, x0);
 *
 * // Integration
 * double integral = SciPy.integrate().quad(expFunc, 0, 1);
 * }</pre>
 *
 * <h2>Integration with Other Modules</h2>
 * <ul>
 *   <li>{@link org.bytedeco.pytorch.plot.matplot.Matplotlib} — Visualization</li>
 *   <li>{@link org.bytedeco.pytorch.plot.seaborn.Seaborn} — Statistical visualization</li>
 *   <li>{@link org.bytedeco.pytorch.tensor.Tensor} — Tensor operations</li>
 *   <li>{@link org.bytedeco.pytorch.dataframe.DataFrame} — DataFrame bridging</li>
 *   <li>{@link org.bytedeco.pytorch.graphx.GraphX} — Graph algorithms</li>
 * </ul>
 *
 * @see <a href="https://github.com/scipy/scipy">SciPy Reference</a>
 */
package org.bytedeco.pytorch.scipy;
