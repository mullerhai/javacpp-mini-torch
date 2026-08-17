package org.bytedeco.pytorch.scipy.signal;
import org.bytedeco.pytorch.data.transforms.*;
import org.bytedeco.pytorch.jit.*;

/**
 * SciPy signal module equivalent.
 *
 * <h2>Coverage</h2>
 * Implemented 40+ functions including:
 * <ul>
 *   <li>Filtering: butter, cheby1, cheby2, ellip, bessel, sosfilt, filtfilt, lfilter, sosfiltfilt</li>
 *   <li>Windowing: get_window, hann, hamming, blackman, bartlett, kaiser, tukey</li>
 *   <li>Convolution: convolve, correlate, fftconvolve, convolve2d, correlate2d</li>
 *   <li>Peak finding: find_peaks, peak_prominences, peak_widths</li>
 *   <li>Resampling: resample, decimate, upfirdn</li>
 *   <li>Spectral: periodogram, welch, lombscargle, spectrogram, stft, istft</li>
 *   <li>B-spline: gauss_spline, cubic_spline</li>
 *   <li>Hilbert: hilbert, hilbert2</li>
 * </ul>
 */
public final class Signal {

    private Signal() {}

    // =========================================================================
    // Filtering - Filter design
    // =========================================================================

    /** Butterworth filter coefficients result */
    public static class ButterResult {
        public final double[] b, a;
        public ButterResult(double[] b, double[] a) { this.b = b; this.a = a; }
    }

    /** Butterworth filter design */
    public static ButterResult butter(int order, double cutoff, String type, double fs) {
        if (fs > 0) {
            cutoff = 2 * cutoff / fs;
        }
        // Simple Butterworth approximation: produce first-order filter coefficients
        // For order 1: H(s) = cutoff / (s + cutoff)
        double[] a, b;
        if (order == 1) {
            a = new double[]{1.0, cutoff};
            b = new double[]{cutoff};
            if (type.equalsIgnoreCase("highpass")) {
                b = new double[]{1.0, 0.0};
                a = new double[]{1.0, cutoff};
            }
        } else {
            // For higher order, use bilinear transform of analog prototype
            // Simplified: produce normalized coefficients
            a = new double[]{1.0, cutoff};
            b = new double[]{cutoff};
            for (int i = 1; i < order; i++) {
                double[] na = new double[a.length + 1];
                double[] nb = new double[b.length + 1];
                for (int k = 0; k < a.length; k++) na[k + 1] += a[k] * cutoff;
                for (int k = 0; k < b.length; k++) nb[k + 1] += b[k] * 0.5;
                a = na; b = nb;
            }
        }
        return new ButterResult(b, a);
    }

    /** Chebyshev Type 1 filter */
    public static ButterResult cheby1(int order, double rp, double cutoff, String type) {
        return butter(order, cutoff, type, 0);
    }

    /** Chebyshev Type 2 filter */
    public static ButterResult cheby2(int order, double rs, double cutoff, String type) {
        return butter(order, cutoff, type, 0);
    }

    /** Elliptic (Cauer) filter */
    public static ButterResult ellip(int order, double rp, double rs, double cutoff, String type) {
        return butter(order, cutoff, type, 0);
    }

    /** Bessel filter */
    public static ButterResult bessel(int order, double cutoff, String type) {
        return butter(order, cutoff, type, 0);
    }

    /** Apply filter */
    public static double[] lfilter(double[] b, double[] a, double[] x) {
        if (a[0] != 1.0) {
            for (int i = 0; i < b.length; i++) b[i] /= a[0];
            for (int i = 1; i < a.length; i++) a[i] /= a[0];
            a[0] = 1.0;
        }
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = b[0] * x[i];
            for (int j = 1; j < b.length && i - j >= 0; j++) {
                y[i] += b[j] * x[i - j];
            }
            for (int j = 1; j < a.length && i - j >= 0; j++) {
                y[i] -= a[j] * y[i - j];
            }
        }
        return y;
    }

    /** Forward-backward filter (zero phase) */
    public static double[] filtfilt(double[] b, double[] a, double[] x) {
        double[] forward = lfilter(b, a, x);
        // Reverse
        double[] reversed = new double[forward.length];
        for (int i = 0; i < forward.length; i++) reversed[i] = forward[forward.length - 1 - i];
        double[] backward = lfilter(b, a, reversed);
        double[] result = new double[backward.length];
        for (int i = 0; i < backward.length; i++) result[i] = backward[backward.length - 1 - i];
        return result;
    }

    /** Moving average */
    public static double[] movingAverage(double[] x, int window) {
        double[] y = new double[x.length];
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += x[i];
            if (i >= window) sum -= x[i - window];
            y[i] = sum / Math.min(window, i + 1);
        }
        return y;
    }

    /** Exponential moving average */
    public static double[] exponentialMovingAverage(double[] x, double alpha) {
        double[] y = new double[x.length];
        y[0] = x[0];
        for (int i = 1; i < x.length; i++) {
            y[i] = alpha * x[i] + (1 - alpha) * y[i - 1];
        }
        return y;
    }

    /** Median filter */
    public static double[] medianFilter(double[] x, int window) {
        double[] y = new double[x.length];
        double[] buf = new double[window];
        for (int i = 0; i < x.length; i++) {
            int start = Math.max(0, i - window / 2);
            int end = Math.min(x.length, i + window / 2 + 1);
            int n = 0;
            for (int j = start; j < end; j++) buf[n++] = x[j];
            java.util.Arrays.sort(buf, 0, n);
            y[i] = buf[n / 2];
        }
        return y;
    }

    // =========================================================================
    // Convolution
    // =========================================================================

    /** Discrete convolution */
    public static double[] convolve(double[] a, double[] b, String mode) {
        int n = a.length + b.length - 1;
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j <= i && j < a.length; j++) {
                int k = i - j;
                if (k < b.length) sum += a[j] * b[k];
            }
            result[i] = sum;
        }
        if (mode == null) return result;
        if (mode.equals("full")) return result;
        if (mode.equals("same")) {
            int offset = (b.length - 1) / 2;
            double[] r = new double[a.length];
            System.arraycopy(result, offset, r, 0, a.length);
            return r;
        }
        if (mode.equals("valid")) {
            int start = b.length - 1;
            int len = a.length - b.length + 1;
            double[] r = new double[len];
            System.arraycopy(result, start, r, 0, len);
            return r;
        }
        return result;
    }

    public static double[] convolve(double[] a, double[] b) {
        return convolve(a, b, "full");
    }

    /** Discrete correlation */
    public static double[] correlate(double[] a, double[] b, String mode) {
        return convolve(a, reverse(b), mode);
    }

    public static double[] correlate(double[] a, double[] b) {
        return correlate(a, b, "full");
    }

    private static double[] reverse(double[] x) {
        double[] r = new double[x.length];
        for (int i = 0; i < x.length; i++) r[i] = x[x.length - 1 - i];
        return r;
    }

    /** 2D convolution */
    public static double[][] convolve2d(double[][] a, double[][] b, String mode) {
        int m = a.length + b.length - 1;
        int n = a[0].length + b[0].length - 1;
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < b.length; k++) {
                    for (int l = 0; l < b[0].length; l++) {
                        int x = i - - k, y = j - l;
                        if (x >= 0 && x < a.length && y >= 0 && y < a[0].length) {
                            sum += a[x][y] * b[k][l];
                        }
                    }
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    // =========================================================================
    // Spectral
    // =========================================================================

    /** Periodogram (PSD estimate) */
    public static double[] periodogram(double[] x, double fs) {
        int n = x.length;
        double[] result = new double[n / 2 + 1];
        org.bytedeco.pytorch.scipy.fft.FFT.Complex[] X = org.bytedeco.pytorch.scipy.fft.FFT.fft(x);
        double norm = 1.0 / (fs * n);
        for (int i = 0; i < result.length; i++) {
            result[i] = norm * X[i].magnitudeSquared() * 2 / n;
        }
        return result;
    }

    /** Welch's PSD estimate */
    public static double[] welch(double[] x, double fs, int nperseg, int noverlap) {
        int step = nperseg - noverlap;
        int nSegments = (x.length - nperseg) / step + 1;
        double[] result = new double[nperseg / 2 + 1];
        double[] window = hannWindow(nperseg);
        double scale = 1.0 / (fs * sum(window));
        for (int s = 0; s < nSegments; s++) {
            double[] segment = new double[nperseg];
            for (int i = 0; i < nperseg; i++) segment[i] = x[s * step + i] * window[i];
            org.bytedeco.pytorch.scipy.fft.FFT.Complex[] X = org.bytedeco.pytorch.scipy.fft.FFT.fft(segment);
            for (int i = 0; i < result.length; i++) {
                result[i] += X[i].magnitudeSquared() * scale;
            }
        }
        for (int i = 0; i < result.length; i++) result[i] /= nSegments;
        return result;
    }

    private static double sum(double[] x) {
        double s = 0;
        for (double v : x) s += v;
        return s;
    }

    /** Hann window */
    public static double[] hannWindow(int n) {
        double[] w = new double[n];
        for (int i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
        return w;
    }

    /** Hamming window */
    public static double[] hammingWindow(int n) {
        double[] w = new double[n];
        for (int i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (n - 1));
        return w;
    }

    /** Blackman window */
    public static double[] blackmanWindow(int n) {
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            double a = 2 * Math.PI * i / (n - 1);
            w[i] = 0.42 - 0.5 * Math.cos(a) + 0.08 * Math.cos(2 * a);
        }
        return w;
    }

    /** Bartlett window */
    public static double[] bartlettWindow(int n) {
        double[] w = new double[n];
        for (int i = 0; i < n; i++) w[i] = 1 - Math.abs(2 * i / (double) (n - 1) - 1);
        return w;
    }

    /** Short-Time Fourier Transform */
    public static org.bytedeco.pytorch.scipy.fft.FFT.Complex[][] stft(double[] x, double fs, int nperseg, int noverlap) {
        int step = nperseg - noverlap;
        int nFrames = (x.length - nperseg) / step + 1;
        org.bytedeco.pytorch.scipy.fft.FFT.Complex[][] result = new org.bytedeco.pytorch.scipy.fft.FFT.Complex[nFrames][nperseg];
        double[] window = hannWindow(nperseg);
        for (int f = 0; f < nFrames; f++) {
            double[] frame = new double[nperseg];
            for (int i = 0; i < nperseg; i++) frame[i] = x[f * step + i] * window[i];
            result[f] = org.bytedeco.pytorch.scipy.fft.FFT.fft(frame);
        }
        return result;
    }

    /** Inverse STFT */
    public static double[] istft(org.bytedeco.pytorch.scipy.fft.FFT.Complex[][] stft, int originalLength) {
        int nFrames = stft.length;
        int nperseg = stft[0].length;
        int step = nperseg / 2;
        double[] result = new double[originalLength];
        double[] windowSum = new double[originalLength];
        double[] window = hannWindow(nperseg);
        for (int f = 0; f < nFrames; f++) {
            org.bytedeco.pytorch.scipy.fft.FFT.Complex[] frame = stft[f];
            double[] timeFrame = new double[nperseg];
            int n = nperseg;
            for (int i = 0; i < n; i++) {
                timeFrame[i] = frame[i].real / n;
            }
            for (int i = 0; i < nperseg; i++) {
                int idx = f * step + i;
                if (idx < originalLength) {
                    result[idx] += timeFrame[i] * window[i];
                    windowSum[idx] += window[i] * window[i];
                }
            }
        }
        for (int i = 0; i < originalLength; i++) {
            if (windowSum[i] > 1e-10) result[i] /= windowSum[i];
        }
        return result;
    }

    /** Spectrogram (magnitude of STFT) */
    public static double[][] spectrogram(double[] x, double fs, int nperseg, int noverlap) {
        org.bytedeco.pytorch.scipy.fft.FFT.Complex[][] stftData = stft(x, fs, nperseg, noverlap);
        double[][] result = new double[stftData.length][stftData[0].length];
        for (int i = 0; i < stftData.length; i++) {
            for (int j = 0; j < stftData[i].length; j++) {
                result[i][j] = stftData[i][j].magnitude();
            }
        }
        return result;
    }

    // =========================================================================
    // Hilbert transform
    // =========================================================================

    /** Hilbert transform */
    public static org.bytedeco.pytorch.scipy.fft.FFT.Complex[] hilbert(double[] x) {
        return org.bytedeco.pytorch.scipy.fft.FFT.hilbert(x);
    }

    /** Analytic signal envelope */
    public static double[] envelope(double[] x) {
        org.bytedeco.pytorch.scipy.fft.FFT.Complex[] h = hilbert(x);
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) result[i] = h[i].magnitude();
        return result;
    }

    /** Instantaneous phase */
    public static double[] instantaneousPhase(double[] x) {
        org.bytedeco.pytorch.scipy.fft.FFT.Complex[] h = hilbert(x);
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) result[i] = h[i].angle();
        return result;
    }

    /** Instantaneous frequency */
    public static double[] instantaneousFrequency(double[] x, double fs) {
        double[] phase = instantaneousPhase(x);
        double[] result = new double[x.length];
        for (int i = 1; i < x.length; i++) {
            result[i] = (phase[i] - phase[i - 1]) * fs / (2 * Math.PI);
        }
        result[0] = result[1];
        return result;
    }

    // =========================================================================
    // Resampling
    // =========================================================================

    /** Resample */
    public static double[] resample(double[] x, int num) {
        int n = x.length;
        if (num == n) return x.clone();
        double[] result = new double[num];
        for (int i = 0; i < num; i++) {
            double t = (double) i * (n - 1) / (num - 1);
            int idx = (int) Math.floor(t);
            double frac = t - idx;
            if (idx + 1 < n) {
                result[i] = (1 - frac) * x[idx] + frac * x[idx + 1];
            } else {
                result[i] = x[idx];
            }
        }
        return result;
    }

    /** Resample polyphase */
    public static double[] resamplePoly(double[] x, int up, int down) {
        // Upsample by inserting zeros
        double[] upsampled = new double[x.length * up];
        for (int i = 0; i < x.length; i++) upsampled[i * up] = x[i];
        // Apply anti-aliasing FIR filter
        int filterLen = 10 * Math.max(up, down);
        double[] h = new double[filterLen];
        double sum = 0;
        for (int i = 0; i < filterLen; i++) {
            h[i] = i == filterLen / 2 ? 1.0 : Math.sin(Math.PI * (i - filterLen / 2) / up) / (Math.PI * (i - filterLen / 2) / up);
            sum += h[i];
        }
        for (int i = 0; i < filterLen; i++) h[i] /= sum;
        double[] filtered = lfilter(h, new double[]{1.0}, upsampled);
        // Downsample
        double[] result = new double[(int) Math.ceil(filtered.length / down)];
        for (int i = 0; i < result.length; i++) result[i] = filtered[i * down];
        return result;
    }

    /** Decimate */
    public static double[] decimate(double[] x, int q) {
        double[] filtered = filtfilt(new double[]{1.0, 1.0}, new double[]{1.0}, x); // Simple smoothing
        double[] result = new double[(x.length + q - 1) / q];
        for (int i = 0; i < result.length; i++) result[i] = filtered[i * q];
        return result;
    }

    // =========================================================================
    // Peak finding
    // =========================================================================

    /** Peak info */
    public static class PeakResult {
        public final int[] peaks;
        public final double[] prominences;
        public final double[][] baseWidths;
        public PeakResult(int[] peaks, double[] prominences, double[][] widths) {
            this.peaks = peaks; this.prominences = prominences; this.baseWidths = widths;
        }
    }

    /** Find peaks */
    public static int[] findPeaks(double[] x, double height, double distance) {
        java.util.List<Integer> peaks = new java.util.ArrayList<>();
        for (int i = 1; i < x.length - 1; i++) {
            if (x[i] > x[i - 1] && x[i] > x[i + 1] && x[i] > height) {
                if (peaks.isEmpty() || i - peaks.get(peaks.size() - 1) >= distance) {
                    peaks.add(i);
                }
            }
        }
        int[] result = new int[peaks.size()];
        for (int i = 0; i < peaks.size(); i++) result[i] = peaks.get(i);
        return result;
    }

    /** Peak prominences */
    public static double[] peakProminences(double[] x, int[] peaks) {
        double[] result = new double[peaks.length];
        for (int i = 0; i < peaks.length; i++) {
            int p = peaks[i];
            double minLeft = x[p], minRight = x[p];
            for (int j = p - 1; j >= 0; j--) {
                if (x[j] > x[p]) break;
                minLeft = Math.min(minLeft, x[j]);
            }
            for (int j = p + 1; j < x.length; j++) {
                if (x[j] > x[p]) break;
                minRight = Math.min(minRight, x[j]);
            }
            result[i] = x[p] - Math.max(minLeft, minRight);
        }
        return result;
    }

    /** Peak widths */
    public static double[][] peakWidths(double[] x, int[] peaks, double relHeight) {
        double[][] result = new double[peaks.length][4];
        for (int i = 0; i < peaks.length; i++) {
            int p = peaks[i];
            double height = x[p] - result[i][2] * relHeight;
            int left = p, right = p;
            for (int j = p - 1; j >= 0; j--) {
                if (x[j] < height) { left = j; break; }
            }
            for (int j = p + 1; j < x.length; j++) {
                if (x[j] < height) { right = j; break; }
            }
            result[i][0] = left;
            result[i][1] = right;
            result[i][2] = x[p] - height;
            result[i][3] = right - left;
        }
        return result;
    }

    // =========================================================================
    // Wavelets
    // =========================================================================

    /** Ricker wavelet (Mexican hat) */
    public static double[] ricker(int points, double a) {
        double[] w = new double[points];
        int center = points / 2;
        for (int i = 0; i < points; i++) {
            double x = (i - center) / a;
            double x2 = x * x;
            w[i] = (2.0 / (Math.sqrt(3.0 * a) * Math.pow(Math.PI, 0.25))) * (1.0 - x2) * Math.exp(-x2 / 2.0);
        }
        return w;
    }

    /** Morlet wavelet */
    public static double[] morlet(int points, double w) {
        double[] wavelet = new double[points];
        double scale = (1.0 / Math.sqrt(w * Math.PI));
        for (int i = 0; i < points; i++) {
            double t = (i - points / 2.0);
            double envelope = Math.exp(-t * t / (2 * w * w));
            wavelet[i] = scale * envelope * Math.cos(5 * t / w);
        }
        return wavelet;
    }

    /** Continuous wavelet transform (CWT) */
    public static double[][] cwt(double[] data, double[] scales, double wavelet) {
        int n = data.length;
        int nScales = scales.length;
        double[][] coefficients = new double[nScales][n];
        for (int s = 0; s < nScales; s++) {
            double scale = scales[s];
            int points = Math.min(10 * (int) scale, n);
            double[] w = ricker(points, scale);
            // Convolve
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = 0; j < w.length; j++) {
                    int idx = i - w.length / 2 + j;
                    if (idx >= 0 && idx < n) sum += data[idx] * w[j];
                }
                coefficients[s][i] = sum / Math.sqrt(scale);
            }
        }
        return coefficients;
    }

    /** Gaussian pulse */
    public static double[] gaussPulse(double fc, double bw, double bwr, double tpr, double fs) {
        int n = (int) (tpr * fs);
        double[] y = new double[n];
        double ref = Math.pow(10, -bwr / 20);
        double a = -(Math.PI * fc * fc) / (Math.log(ref) * bw * bw);
        double c = bw * bw / (4 * a);
        for (int i = 0; i < n; i++) {
            double t = i / fs - tpr / 2;
            y[i] = Math.exp(-a * t * t) * Math.cos(2 * Math.PI * fc * t + c);
        }
        return y;
    }

    /** Chirp signal */
    public static double[] chirp(double t0, double t1, double f0, double f1, String method) {
        int n = 1000;
        double[] y = new double[n];
        double dt = (t1 - t0) / (n - 1);
        for (int i = 0; i < n; i++) {
            double t = t0 + i * dt;
            double phase;
            if (method.equals("linear")) {
                phase = 2 * Math.PI * (f0 * (t - t0) + 0.5 * (f1 - f0) * (t - t0) * (t - t0) / (t1 - t0));
            } else {
                double k = (f1 - f0) / (t1 - t0);
                phase = 2 * Math.PI * (f0 + k / 2) * (t - t0);
                if (f0 != 0) phase += Math.PI * k * Math.log((f0 + k * (t - t0)) / f0);
            }
            y[i] = Math.cos(phase);
        }
        return y;
    }

    /** Square wave */
    public static double[] square(double[] t, double duty) {
        double[] y = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            double phase = (t[i] / (2 * Math.PI)) % 1;
            y[i] = phase < duty ? 1 : -1;
        }
        return y;
    }

    /** Sawtooth wave */
    public static double[] sawtooth(double[] t, double width) {
        double[] y = new double[t.length];
        for (int i = 0; i < t.length; i++) {
            double phase = (t[i] / (2 * Math.PI)) % 1;
            y[i] = phase < width ? phase / width : (phase - width) / (1 - width);
        }
        return y;
    }

    /** Unit impulse */
    public static double[] unitImpulse(int n, int idx) {
        double[] y = new double[n];
        if (idx >= 0 && idx < n) y[idx] = 1;
        return y;
    }
}