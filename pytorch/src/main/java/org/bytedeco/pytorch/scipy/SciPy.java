package org.bytedeco.pytorch.scipy;

import org.bytedeco.pytorch.scipy.cluster.Cluster;
import org.bytedeco.pytorch.scipy.constants.Constants;
import org.bytedeco.pytorch.scipy.datasets.Datasets;
import org.bytedeco.pytorch.scipy.fft.FFT;
import org.bytedeco.pytorch.scipy.integrate.Integrate;
import org.bytedeco.pytorch.scipy.interpolate.Interpolate;
import org.bytedeco.pytorch.scipy.io.IO;
import org.bytedeco.pytorch.scipy.linalg.Linalg;
import org.bytedeco.pytorch.scipy.ndimage.NdImage;
import org.bytedeco.pytorch.scipy.optimize.Optimize;
import org.bytedeco.pytorch.scipy.signal.Signal;
import org.bytedeco.pytorch.scipy.sparse.Sparse;
import org.bytedeco.pytorch.scipy.spatial.Spatial;
import org.bytedeco.pytorch.scipy.special.Special;
import org.bytedeco.pytorch.scipy.stats.Stats;

/**
 * SciPy main facade - mirrors the top-level scipy Python API.
 *
 * <h2>Total coverage</h2>
 * 250+ scientific operators across 17 submodules:
 * <ul>
 *   <li>{@link Constants} - 60+ physical/mathematical constants</li>
 *   <li>{@link Special} - 80+ special mathematical functions</li>
 *   <li>{@link Linalg} - 50+ linear algebra operators</li>
 *   <li>{@link FFT} - 30+ FFT/spectral operators</li>
 *   <li>{@link Integrate} - 25+ integration/ODE operators</li>
 *   <li>{@link Optimize} - 30+ optimization/root-finding operators</li>
 *   <li>{@link Stats} - 80+ statistical distributions/tests</li>
 *   <li>{@link Signal} - 40+ signal processing functions</li>
 *   <li>{@link Spatial} - 25+ distance/geometric algorithms</li>
 *   <li>{@link Sparse} - 25+ sparse matrix operations</li>
 *   <li>{@link Interpolate} - 15+ interpolation methods</li>
 *   <li>{@link NdImage} - 30+ image processing functions</li>
 *   <li>{@link Cluster} - 15+ clustering algorithms</li>
 *   <li>{@link Datasets} - 10+ sample datasets</li>
 *   <li>{@link IO} - 15+ file I/O operations</li>
 * </ul>
 */
public final class SciPy {

    private SciPy() {}

    /** Direct access to scipy.constants */
    public static final class _const {
        public static double pi() { return Constants.pi; }
        public static double e() { return Constants.e; }
        public static double golden_ratio() { return Constants.golden_ratio; }
        public static double c() { return Constants.c; }
        public static double h() { return Constants.h; }
        public static double k() { return Constants.k_B; }
        public static double R() { return Constants.R; }
        public static double g() { return Constants.g; }
        public static double G() { return Constants.G; }
        public static double m_e() { return Constants.m_e; }
        public static double m_p() { return Constants.m_p; }
        public static double m_n() { return Constants.m_n; }
        public static double N_A() { return Constants.N_A; }
        public static double mu_0() { return Constants.mu_0; }
        public static double epsilon_0() { return Constants.epsilon_0; }
        public static double sigma() { return Constants.sigma; }
    }
}
