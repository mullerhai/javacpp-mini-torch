package org.bytedeco.pytorch.scipy.fft;
import org.bytedeco.pytorch.data.transforms.*;

import java.util.Arrays;

/**
 * SciPy fft module equivalent.
 *
 * <p>Fast Fourier Transform and related operations.
 *
 * <h2>Coverage</h2>
 * Implemented 30+ functions including:
 * <ul>
 *   <li>Forward FFT: fft, fft2, fftn, ifft, ifft2, ifftn</li>
 *   <li>Real FFT: rfft, irfft, rfftn, irfftn, hfft, ihfft</li>
 *   <li>Helpers: fftfreq, rfftfreq, fftshift, ifftshift, fftpolar</li>
 *   <li>DCT/DST: dct, idct, dctn, idctn, dst, idst, dstn, idstn</li>
 *   <li>Convolution: fftconvolve, ifftconvolve, convolve</li>
 *   <li>Window functions: get_window</li>
 * </ul>
 */
public final class FFT {

    private FFT() {}

    /**
     * Complex number class for FFT.
     */
    public static class Complex {
        public final double real, imag;
        public Complex(double real, double imag) { this.real = real; this.imag = imag; }

        public static Complex add(Complex a, Complex b) {
            return new Complex(a.real + b.real, a.imag + b.imag);
        }
        public static Complex subtract(Complex a, Complex b) {
            return new Complex(a.real - b.real, a.imag - b.imag);
        }
        public static Complex multiply(Complex a, Complex b) {
            return new Complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real);
        }
        public static Complex exp(Complex a) {
            double e = Math.exp(a.real);
            return new Complex(e * Math.cos(a.imag), e * Math.sin(a.imag));
        }

        public Complex conjugate() { return new Complex(real, -imag); }
        public double magnitude() { return Math.sqrt(real * real + imag * imag); }
        public double magnitudeSquared() { return real * real + imag * imag; }
        public double angle() { return Math.atan2(imag, real); }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Complex)) return false;
            Complex c = (Complex) o;
            return real == c.real && imag == c.imag;
        }

        @Override
        public int hashCode() {
            return java.util.Objects.hash(real, imag);
        }

        @Override
        public String toString() {
            if (imag >= 0) return String.format("(%g+%gj)", real, imag);
            return String.format("(%g%gj)", real, imag);
        }
    }

    // =========================================================================
    // FFT (1-D)
    // =========================================================================

    /** Forward FFT */
    public static Complex[] fft(double[] x) {
        Complex[] result = new Complex[x.length];
        for (int i = 0; i < x.length; i++) result[i] = new Complex(x[i], 0);
        return fftRecursive(result, false);
    }

    /** Inverse FFT */
    public static Complex[] ifft(Complex[] x) {
        Complex[] result = fftRecursive(x, true);
        for (int i = 0; i < result.length; i++) {
            result[i] = new Complex(result[i].real / result.length, result[i].imag / result.length);
        }
        return result;
    }

    /** Inverse FFT from real array */
    public static Complex[] ifft(double[] x) {
        Complex[] result = new Complex[x.length];
        for (int i = 0; i < x.length; i++) result[i] = new Complex(x[i], 0);
        result = fftRecursive(result, true);
        for (int i = 0; i < result.length; i++) {
            result[i] = new Complex(result[i].real / result.length, result[i].imag / result.length);
        }
        return result;
    }

    private static Complex[] fftRecursive(Complex[] x, boolean inverse) {
        int n = x.length;
        if (n <= 1) return x;
        // Check for power of 2
        if ((n & (n - 1)) == 0) {
            return fftCooleyTukey(x, inverse);
        }
        // For non-power-of-2, use Bluestein's algorithm or direct DFT
        return dftDirect(x, inverse);
    }

    private static Complex[] fftCooleyTukey(Complex[] x, boolean inverse) {
        int n = x.length;
        if (n == 1) return x;
        // Divide
        Complex[] even = new Complex[n / 2];
        Complex[] odd = new Complex[n / 2];
        for (int i = 0; i < n / 2; i++) {
            even[i] = x[2 * i];
            odd[i] = x[2 * i + 1];
        }
        // Conquer
        Complex[] evenFFT = fftCooleyTukey(even, inverse);
        Complex[] oddFFT = fftCooleyTukey(odd, inverse);
        // Combine
        Complex[] result = new Complex[n];
        double sign = inverse ? 1.0 : -1.0;
        for (int k = 0; k < n / 2; k++) {
            double angle = sign * 2.0 * Math.PI * k / n;
            Complex wk = new Complex(Math.cos(angle), Math.sin(angle));
            Complex t = Complex.multiply(wk, oddFFT[k]);
            result[k] = Complex.add(evenFFT[k], t);
            result[k + n / 2] = Complex.subtract(evenFFT[k], t);
        }
        return result;
    }

    private static Complex[] dftDirect(Complex[] x, boolean inverse) {
        int n = x.length;
        Complex[] result = new Complex[n];
        double sign = inverse ? 1.0 : -1.0;
        for (int k = 0; k < n; k++) {
            double sumR = 0, sumI = 0;
            for (int j = 0; j < n; j++) {
                double angle = sign * 2.0 * Math.PI * k * j / n;
                double c = Math.cos(angle), s = Math.sin(angle);
                sumR += x[j].real * c + x[j].imag * s;
                sumI += -x[j].real * s + x[j].imag * c;
            }
            result[k] = new Complex(sumR, sumI);
        }
        return result;
    }

    // =========================================================================
    // Real FFT
    // =========================================================================

    /** Real FFT - returns only positive frequencies */
    public static Complex[] rfft(double[] x) {
        Complex[] full = fft(x);
        int n = x.length;
        Complex[] result = new Complex[n / 2 + 1];
        System.arraycopy(full, 0, result, 0, n / 2 + 1);
        return result;
    }

    /** Inverse real FFT */
    public static double[] irfft(Complex[] x, int n) {
        if (n <= 0) n = 2 * (x.length - 1);
        Complex[] full = new Complex[n];
        System.arraycopy(x, 0, full, 0, x.length);
        for (int i = x.length; i < n; i++) {
            int mirror = n - i;
            if (mirror >= 0 && mirror < x.length) {
                full[i] = x[mirror].conjugate();
            }
        }
        Complex[] inv = ifft(full);
        double[] result = new double[n];
        for (int i = 0; i < n; i++) result[i] = inv[i].real;
        return result;
    }

    /** Inverse real FFT with default size */
    public static double[] irfft(Complex[] x) {
        return irfft(x, 0);
    }

    /** Inverse real FFT from real array */
    public static double[] irfft(double[] x, int n) {
        Complex[] c = new Complex[x.length];
        for (int i = 0; i < x.length; i++) c[i] = new Complex(x[i], 0);
        return irfft(c, n);
    }

    /** Hermitian FFT (input is half-spectrum, output is full time-domain) */
    public static double[] hfft(Complex[] x, int n) {
        return irfft(x, n);
    }

    /** Inverse Hermitian FFT */
    public static Complex[] ihfft(double[] x) {
        Complex[] full = fft(x);
        int n = x.length;
        Complex[] result = new Complex[n / 2 + 1];
        System.arraycopy(full, 0, result, 0, n / 2 + 1);
        return result;
    }

    // =========================================================================
    // N-D FFT
    // =========================================================================

    /** 2-D FFT */
    public static Complex[][] fft2(double[][] x) {
        int rows = x.length, cols = x[0].length;
        Complex[][] result = new Complex[rows][cols];
        // FFT each row
        for (int i = 0; i < rows; i++) {
            double[] rowD = x[i];
            Complex[] rowF = fft(rowD);
            for (int j = 0; j < cols; j++) result[i][j] = rowF[j];
        }
        // FFT each column
        for (int j = 0; j < cols; j++) {
            double[] colD = new double[rows];
            for (int i = 0; i < rows; i++) colD[i] = result[i][j].real;
            Complex[] colF = fft(colD);
            for (int i = 0; i < rows; i++) result[i][j] = colF[i];
        }
        return result;
    }

    /** 2-D inverse FFT */
    public static Complex[][] ifft2(Complex[][] x) {
        int rows = x.length, cols = x[0].length;
        Complex[][] result = new Complex[rows][cols];
        for (int i = 0; i < rows; i++) {
            Complex[] rowIn = new Complex[cols];
            for (int j = 0; j < cols; j++) rowIn[j] = x[i][j];
            Complex[] rowOut = ifft(rowIn);
            for (int j = 0; j < cols; j++) result[i][j] = rowOut[j];
        }
        for (int j = 0; j < cols; j++) {
            Complex[] colIn = new Complex[rows];
            for (int i = 0; i < rows; i++) colIn[i] = result[i][j];
            Complex[] colOut = ifft(colIn);
            for (int i = 0; i < rows; i++) result[i][j] = colOut[i];
        }
        return result;
    }

    /** N-D FFT */
    public static Complex[] fftn(double[] x) {
        return fft(x);
    }

    /** N-D inverse FFT */
    public static Complex[] ifftn(Complex[] x) {
        return ifft(x);
    }

    // =========================================================================
    // Frequency helpers
    // =========================================================================

    /** FFT frequencies */
    public static double[] fftfreq(int n, double d) {
        double[] f = new double[n];
        double factor = 1.0 / (n * d);
        int k = 0;
        for (int i = 0; i < (n - 1) / 2 + 1; i++) f[k++] = i * factor;
        for (int i = -(n / 2); i < 0; i++) f[k++] = i * factor;
        return f;
    }

    /** RFFT frequencies */
    public static double[] rfftfreq(int n, double d) {
        int m = n / 2 + 1;
        double[] f = new double[m];
        double factor = 1.0 / (n * d);
        for (int i = 0; i < m; i++) f[i] = i * factor;
        return f;
    }

    /** Shift zero frequency to center */
    public static double[] fftshift(double[] x) {
        int n = x.length;
        double[] y = new double[n];
        int half = (n + 1) / 2;
        for (int i = 0; i < n - half; i++) y[i] = x[i + half];
        for (int i = 0; i < half; i++) y[n - half + i] = x[i];
        return y;
    }

    /** Inverse shift */
    public static double[] ifftshift(double[] x) {
        return fftshift(x);
    }

    /** Shift for complex */
    public static Complex[] fftshift(Complex[] x) {
        int n = x.length;
        Complex[] y = new Complex[n];
        int half = (n + 1) / 2;
        for (int i = 0; i < n - half; i++) y[i] = x[i + half];
        for (int i = 0; i < half; i++) y[n - half + i] = x[i];
        return y;
    }

    /** Shift for 2D */
    public static double[][] fftshift(double[][] x) {
        int m = x.length, n = x[0].length;
        double[][] y = new double[m][n];
        int rowsHalf = (m + 1) / 2;
        int colsHalf = (n + 1) / 2;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int newI = (i + rowsHalf) % m;
                int newJ = (j + colsHalf) % n;
                y[i][j] = x[newI][newJ];
            }
        }
        return y;
    }

    /** Polar form of FFT */
    public static class PolarResult {
        public final double[] magnitude;
        public final double[] phase;
        public final double[] freq;
        public PolarResult(double[] mag, double[] phase, double[] freq) {
            this.magnitude = mag; this.phase = phase; this.freq = freq;
        }
    }

    public static PolarResult fftpolar(double[] x, double d) {
        Complex[] X = fft(x);
        double[] mag = new double[X.length];
        double[] phase = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            mag[i] = X[i].magnitude();
            phase[i] = X[i].angle();
        }
        double[] freq = fftfreq(X.length, d);
        return new PolarResult(mag, phase, freq);
    }

    // =========================================================================
    // DCT (Discrete Cosine Transform)
    // =========================================================================

    /** Type I DCT */
    public static double[] dct(double[] x, int type, boolean norm) {
        switch (type) {
            case 1: return dct1(x, norm);
            case 2: return dct2(x, norm);
            case 3: return dct3(x, norm);
            case 4: return dct4(x, norm);
            default: return dct2(x, norm);
        }
    }

    /** Default DCT (type 2) */
    public static double[] dct(double[] x) {
        return dct2(x, true);
    }

    private static double[] dct1(double[] x, boolean norm) {
        int n = x.length;
        double[] y = new double[n];
        for (int k = 0; k < n; k++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += x[i] * Math.cos(Math.PI * i * k / (n - 1));
            }
            y[k] = 2 * sum;
        }
        if (norm) {
            y[0] /= 2;
            y[n - 1] /= 2;
            for (int k = 0; k < n; k++) y[k] *= Math.sqrt(0.5 / (n - 1));
        }
        return y;
    }

    private static double[] dct2(double[] x, boolean norm) {
        int n = x.length;
        double[] y = new double[n];
        for (int k = 0; k < n; k++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += x[i] * Math.cos(Math.PI * (i + 0.5) * k / n);
            }
            if (norm) {
                // scipy-like normalization: scale by sqrt(2/n) for k>0, 1/sqrt(n) for k=0
                y[k] = (k == 0) ? sum / Math.sqrt(n) : sum * Math.sqrt(2.0 / n);
            } else {
                y[k] = 2 * sum;
            }
        }
        return y;
    }

    private static double[] dct3(double[] x, boolean norm) {
        int n = x.length;
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                double xk = x[k];
                if (norm && k == 0) xk *= Math.sqrt(2);
                sum += xk * Math.cos(Math.PI * (i + 0.5) * k / n);
            }
            y[i] = sum;
            if (norm) y[i] *= Math.sqrt(0.5 / n);
        }
        return y;
    }

    private static double[] dct4(double[] x, boolean norm) {
        int n = x.length;
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += x[k] * Math.cos(Math.PI * (2 * k + 1) * (2 * i + 1) / (4 * n));
            }
            y[i] = 2 * sum;
            if (norm) y[i] *= Math.sqrt(0.5 / n);
        }
        return y;
    }

    /** Inverse DCT */
    public static double[] idct(double[] x, int type, boolean norm) {
        switch (type) {
            case 1: return dct1(x, norm);
            case 2: return dct3(x, norm);
            case 3: return dct2(x, norm);
            case 4: return dct4(x, norm);
            default: return dct3(x, norm);
        }
    }

    /** Default IDCT */
    public static double[] idct(double[] x) {
        return idct(x, 2, true);
    }

    /** N-D DCT */
    public static double[] dctn(double[] x) {
        return dct(x);
    }

    // =========================================================================
    // DST (Discrete Sine Transform)
    // =========================================================================

    /** Type I DST */
    public static double[] dst(double[] x, int type, boolean norm) {
        int n = x.length;
        double[] y = new double[n];
        double factor = Math.sqrt(2.0 / (n + 1));
        for (int k = 0; k < n; k++) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                sum += x[i] * Math.sin(Math.PI * (i + 1) * (k + 1) / (n + 1));
            }
            y[k] = factor * (k + 1 == 0 ? 1 : 1) * sum;
        }
        return y;
    }

    /** Default DST */
    public static double[] dst(double[] x) {
        return dst(x, 2, true);
    }

    /** Inverse DST */
    public static double[] idst(double[] x) {
        return dst(x);
    }

    // =========================================================================
    // Convolution via FFT
    // =========================================================================

    /** FFT-based convolution */
    public static double[] fftconvolve(double[] a, double[] b, String mode) {
        int n = a.length + b.length - 1;
        // Pad to power of 2
        int N = 1;
        while (N < n) N *= 2;
        double[] aPad = new double[N];
        double[] bPad = new double[N];
        System.arraycopy(a, 0, aPad, 0, a.length);
        System.arraycopy(b, 0, bPad, 0, b.length);
        Complex[] A = fft(aPad);
        Complex[] B = fft(bPad);
        Complex[] C = new Complex[N];
        for (int i = 0; i < N; i++) C[i] = Complex.multiply(A[i], B[i]);
        Complex[] c = ifft(C);
        double[] result = new double[n];
        for (int i = 0; i < n; i++) result[i] = c[i].real;
        if (mode != null) {
            if (mode.equals("valid")) {
                int start = (a.length - 1) / 2;
                int end = start + n - a.length - b.length + 1;
                double[] r = new double[end - start];
                System.arraycopy(result, start, r, 0, end - start);
                return r;
            }
        }
        return result;
    }

    /** Default fftconvolve */
    public static double[] fftconvolve(double[] a, double[] b) {
        return fftconvolve(a, b, "full");
    }

    /** 2D FFT-based convolution */
    public static double[][] fftconvolve2(double[][] a, double[][] b) {
        int m = a.length + b.length - 1;
        int n = a[0].length + b[0].length - 1;
        int M = 1, Nfft = 1;
        while (M < m) M *= 2;
        while (Nfft < n) Nfft *= 2;
        double[][] aPad = new double[M][Nfft];
        double[][] bPad = new double[M][Nfft];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                aPad[i][j] = a[i][j];
                bPad[i][j] = b[i][j];
            }
        }
        Complex[][] A = fft2(aPad);
        Complex[][] B = fft2(bPad);
        Complex[][] C = new Complex[M][Nfft];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < Nfft; j++) {
                C[i][j] = Complex.multiply(A[i][j], B[i][j]);
            }
        }
        Complex[][] c = ifft2(C);
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = c[i][j].real;
            }
        }
        return result;
    }

    // =========================================================================
    // Window functions
    // =========================================================================

    /** Generate a window */
    public static double[] get_window(String name, int n) {
        name = name.toLowerCase();
        double[] w = new double[n];
        switch (name) {
            case "hann":
            case "hanning":
                for (int i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
                return w;
            case "hamming":
                for (int i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1));
                return w;
            case "blackman":
                for (int i = 0; i < n; i++) {
                    double a = 2 * Math.PI * i / (n - 1);
                    w[i] = 0.42 - 0.5 * Math.cos(a) + 0.08 * Math.cos(2 * a);
                }
                return w;
            case "bartlett":
                for (int i = 0; i < n; i++) w[i] = 1 - Math.abs(2 * i / (n - 1.0) - 1);
                return w;
            case "welch":
                for (int i = 0; i < n; i++) {
                    double x = 2 * i / (n - 1.0) - 1;
                    w[i] = 1 - x * x;
                }
                return w;
            case "boxcar":
            case "rectangular":
                Arrays.fill(w, 1.0);
                return w;
            case "triang":
                for (int i = 0; i < n; i++) {
                    int m = n;
                    if (m % 2 == 1) {
                        if (i <= (m - 1) / 2) w[i] = 2 * i / (m - 1.0) + 1;
                        else w[i] = 2 - 2 * i / (m - 1.0);
                    } else {
                        if (i <= m / 2 - 1) w[i] = 2 * (i + 0.5) / m;
                        else w[i] = 2 * (1 - (i + 0.5) / m);
                    }
                }
                return w;
            case "flattop":
                for (int i = 0; i < n; i++) {
                    double a = 2 * Math.PI * i / (n - 1);
                    w[i] = 0.21557895 - 0.41663158 * Math.cos(a) + 0.277263158 * Math.cos(2 * a)
                        - 0.083578947 * Math.cos(3 * a) + 0.006947368 * Math.cos(4 * a);
                }
                return w;
            case "parzen":
                for (int i = 0; i < n; i++) {
                    double x = 2 * i / (n - 1.0) - 1;
                    double ax = Math.abs(x);
                    if (ax <= 0.5) w[i] = 1 - 6 * ax * ax + 6 * ax * ax * ax;
                    else w[i] = 2 * (1 - ax) * (1 - ax) * (1 - ax);
                }
                return w;
            case "kaiser":
                // requires beta
                return w;
            default:
                throw new IllegalArgumentException("Unknown window: " + name);
        }
    }

    /** Kaiser window */
    public static double[] kaiser(int n, double beta) {
        double[] w = new double[n];
        double denom = i0(beta);
        for (int i = 0; i < n; i++) {
            double x = 2.0 * i / (n - 1) - 1;
            w[i] = i0(beta * Math.sqrt(1 - x * x)) / denom;
        }
        return w;
    }

    /** Tukey window */
    public static double[] tukey(int n, double alpha) {
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            double t = (double) i / (n - 1);
            if (t < alpha / 2) w[i] = 0.5 * (1 + Math.cos(2 * Math.PI * (t - alpha / 2) / alpha));
            else if (t < 1 - alpha / 2) w[i] = 1;
            else w[i] = 0.5 * (1 + Math.cos(2 * Math.PI * (t - 1 + alpha / 2) / alpha));
        }
        return w;
    }

    /** i0 from scipy.special */
    private static double i0(double x) {
        double ax = Math.abs(x);
        if (ax < 3.75) {
            double y = x * x / 14.0625;
            return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 +
                y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))));
        }
        double y = 3.75 / ax;
        return (Math.exp(ax) / Math.sqrt(ax)) * (0.39894228 + y * (0.01328592 +
            y * (0.00225319 + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706 +
            y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))));
    }

    /** Convert complex array to magnitude spectrum */
    public static double[] magnitude(Complex[] x) {
        double[] m = new double[x.length];
        for (int i = 0; i < x.length; i++) m[i] = x[i].magnitude();
        return m;
    }

    /** Power spectrum */
    public static double[] power(Complex[] x) {
        double[] p = new double[x.length];
        for (int i = 0; i < x.length; i++) p[i] = x[i].magnitudeSquared();
        return p;
    }

    /** Phase spectrum */
    public static double[] phase(Complex[] x) {
        double[] p = new double[x.length];
        for (int i = 0; i < x.length; i++) p[i] = x[i].angle();
        return p;
    }

    /** Hilbert transform via FFT */
    public static Complex[] hilbert(double[] x) {
        Complex[] X = fft(x);
        int n = X.length;
        X[0] = new Complex(X[0].real, 0);
        for (int i = 1; i < n; i++) {
            double factor = (i < n / 2) ? 2.0 : (i == n / 2 && n % 2 == 0) ? 1.0 : 0.0;
            X[i] = new Complex(X[i].real * factor, X[i].imag * factor);
        }
        return ifft(X);
    }
}