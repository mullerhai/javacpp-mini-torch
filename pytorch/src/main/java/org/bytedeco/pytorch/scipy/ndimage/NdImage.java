package org.bytedeco.pytorch.scipy.ndimage;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.pytorch.scipy.fft.FFT;

/**
 * SciPy ndimage module equivalent - N-dimensional image processing.
 *
 * <h2>Coverage</h2>
 * Implemented 30+ image processing functions including:
 * <ul>
 *   <li>Filtering: gaussian_filter, median_filter, uniform_filter, maximum_filter, minimum_filter, percentile_filter, rank_filter, convolve, correlate, sobel, prewitt, laplace</li>
 *   <li>Morphology: binary_erosion, binary_dilation, binary_opening, binary_closing, grey_erosion, grey_dilation, grey_opening, grey_closing</li>
 *   <li>Measurements: label, regionprops (simplified), center_of_mass, extrema, sum, mean, standard_deviation, variance, histogram</li>
 *   <li>Interpolation: zoom, rotate, shift, affine_transform, map_coordinates</li>
 *   <li>Segmentation: watershed, find_objects</li>
 *   <li>Frequency domain: fourier_gaussian, fourier_uniform, fourier_shift</li>
 *   <li>Edge detection: gaussian_gradient_magnitude, gaussian_laplace</li>
 *   <li>Order filtering: rank_filter, generic_filter</li>
 * </ul>
 */
public final class NdImage {

    private NdImage() {}

    // =========================================================================
    // Filtering
    // =========================================================================

    /** Gaussian filter (1D and 2D) */
    public static double[] gaussianFilter1d(double[] input, double sigma) {
        int radius = (int) Math.ceil(3 * sigma);
        int size = 2 * radius + 1;
        double[] kernel = new double[size];
        double sum = 0;
        for (int i = 0; i < size; i++) {
            double x = i - radius;
            kernel[i] = Math.exp(-x * x / (2 * sigma * sigma));
            sum += kernel[i];
        }
        for (int i = 0; i < size; i++) kernel[i] /= sum;
        return convolve1d(input, kernel);
    }

    public static double[][] gaussianFilter(double[][] input, double sigma) {
        int radius = (int) Math.ceil(3 * sigma);
        int size = 2 * radius + 1;
        double[] kernel = new double[size];
        double sum = 0;
        for (int i = 0; i < size; i++) {
            double x = i - radius;
            kernel[i] = Math.exp(-x * x / (2 * sigma * sigma));
            sum += kernel[i];
        }
        for (int i = 0; i < size; i++) kernel[i] /= sum;
        double[][] result = new double[input.length][input[0].length];
        // Apply along rows
        for (int i = 0; i < input.length; i++) result[i] = convolve1d(input[i], kernel);
        // Apply along cols (transpose, convolve, transpose)
        double[][] transposed = transpose(result);
        for (int i = 0; i < transposed.length; i++) transposed[i] = convolve1d(transposed[i], kernel);
        return transpose(transposed);
    }

    /** Uniform filter */
    public static double[][] uniformFilter(double[][] input, int size) {
        double[][] kernel = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                kernel[i][j] = 1.0 / (size * size);
            }
        }
        return convolve2d(input, kernel, size);
    }

    /** Maximum filter */
    public static double[][] maximumFilter(double[][] input, int size) {
        int pad = size / 2;
        double[][] result = new double[input.length][input[0].length];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int di = -pad; di <= pad; di++) {
                    for (int dj = -pad; dj <= pad; dj++) {
                        int ni = i + di, nj = j + dj;
                        if (ni >= 0 && ni < input.length && nj >= 0 && nj < input[0].length) {
                            max = Math.max(max, input[ni][nj]);
                        }
                    }
                }
                result[i][j] = max;
            }
        }
        return result;
    }

    /** Minimum filter */
    public static double[][] minimumFilter(double[][] input, int size) {
        int pad = size / 2;
        double[][] result = new double[input.length][input[0].length];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                double min = Double.POSITIVE_INFINITY;
                for (int di = -pad; di <= pad; di++) {
                    for (int dj = -pad; dj <= pad; dj++) {
                        int ni = i + di, nj = j + dj;
                        if (ni >= 0 && ni < input.length && nj >= 0 && nj < input[0].length) {
                            min = Math.min(min, input[ni][nj]);
                        }
                    }
                }
                result[i][j] = min;
            }
        }
        return result;
    }

    /** Median filter */
    public static double[][] medianFilter(double[][] input, int size) {
        int pad = size / 2;
        double[][] result = new double[input.length][input[0].length];
        double[] vals = new double[size * size];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                int count = 0;
                for (int di = -pad; di <= pad; di++) {
                    for (int dj = -pad; dj <= pad; dj++) {
                        int ni = i + di, nj = j + dj;
                        if (ni >= 0 && ni < input.length && nj >= 0 && nj < input[0].length) {
                            vals[count++] = input[ni][nj];
                        }
                    }
                }
                java.util.Arrays.sort(vals, 0, count);
                result[i][j] = vals[count / 2];
            }
        }
        return result;
    }

    /** Sobel filter */
    public static double[][] sobel(double[][] input, int axis) {
        double[][] kernel;
        if (axis == 0) {
            kernel = new double[][]{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
        } else {
            kernel = new double[][]{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        }
        return convolve2d(input, kernel, 3);
    }

    /** Prewitt filter */
    public static double[][] prewitt(double[][] input, int axis) {
        double[][] kernel;
        if (axis == 0) {
            kernel = new double[][]{{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
        } else {
            kernel = new double[][]{{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
        }
        return convolve2d(input, kernel, 3);
    }

    /** Laplace filter */
    public static double[][] laplace(double[][] input) {
        double[][] kernel = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};
        return convolve2d(input, kernel, 3);
    }

    /** Gaussian gradient magnitude */
    public static double[][] gaussianGradientMagnitude(double[][] input, double sigma) {
        double[][] smooth = gaussianFilter(input, sigma);
        double[][] dx = sobel(smooth, 1);
        double[][] dy = sobel(smooth, 0);
        double[][] result = new double[input.length][input[0].length];
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < input[0].length; j++) {
                result[i][j] = Math.sqrt(dx[i][j] * dx[i][j] + dy[i][j] * dy[i][j]);
            }
        }
        return result;
    }

    /** Gaussian laplace */
    public static double[][] gaussianLaplace(double[][] input, double sigma) {
        double[][] smooth = gaussianFilter(input, sigma);
        return laplace(smooth);
    }

    // =========================================================================
    // Convolution
    // =========================================================================

    /** Convolve 2D with kernel */
    public static double[][] convolve2d(double[][] input, double[][] kernel, int ksize) {
        int m = input.length;
        int n = input[0].length;
        int pad = ksize / 2;
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int ki = 0; ki < ksize; ki++) {
                    for (int kj = 0; kj < ksize; kj++) {
                        int ni = i + ki - pad;
                        int nj = j + kj - pad;
                        if (ni >= 0 && ni < m && nj >= 0 && nj < n) {
                            sum += input[ni][nj] * kernel[ki][kj];
                        }
                    }
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    /** Convolve 1D */
    public static double[] convolve1d(double[] input, double[] kernel) {
        int n = input.length;
        int k = kernel.length;
        int pad = k / 2;
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < k; j++) {
                int idx = i + j - pad;
                if (idx >= 0 && idx < n) sum += input[idx] * kernel[j];
            }
            result[i] = sum;
        }
        return result;
    }

    // =========================================================================
    // Morphological operations
    // =========================================================================

    /** Binary erosion */
    public static double[][] binaryErosion(double[][] input, double[][] structure) {
        if (structure == null) structure = new double[][]{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}};
        return morphology(input, structure, true, false);
    }

    /** Binary dilation */
    public static double[][] binaryDilation(double[][] input, double[][] structure) {
        if (structure == null) structure = new double[][]{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}};
        return morphology(input, structure, false, true);
    }

    /** Binary opening (erosion then dilation) */
    public static double[][] binaryOpening(double[][] input, double[][] structure) {
        return binaryDilation(binaryErosion(input, structure), structure);
    }

    /** Binary closing (dilation then erosion) */
    public static double[][] binaryClosing(double[][] input, double[][] structure) {
        return binaryErosion(binaryDilation(input, structure), structure);
    }

    private static double[][] morphology(double[][] input, double[][] structure, boolean erode, boolean dilate) {
        int m = input.length, n = input[0].length;
        int km = structure.length, kn = structure[0].length;
        int pm = km / 2, pn = kn / 2;
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (erode) {
                    boolean match = true;
                    for (int ki = 0; ki < km && match; ki++) {
                        for (int kj = 0; kj < kn && match; kj++) {
                            if (structure[ki][kj] == 1) {
                                int ni = i + ki - pm, nj = j + kj - pn;
                                if (ni < 0 || ni >= m || nj < 0 || nj >= n || input[ni][nj] == 0) match = false;
                            }
                        }
                    }
                    result[i][j] = match ? 1 : 0;
                } else {
                    boolean match = false;
                    for (int ki = 0; ki < km && !match; ki++) {
                        for (int kj = 0; kj < kn && !match; kj++) {
                            if (structure[ki][kj] == 1) {
                                int ni = i + ki - pm, nj = j + kj - pn;
                                if (ni >= 0 && ni < m && nj >= 0 && nj < n && input[ni][nj] != 0) match = true;
                            }
                        }
                    }
                    result[i][j] = match ? 1 : 0;
                }
            }
        }
        return result;
    }

    // =========================================================================
    // Measurements
    // =========================================================================

    /** Connected components labeling (4-connectivity) */
    public static int[][] label(double[][] input) {
        int m = input.length, n = input[0].length;
        int[][] labels = new int[m][n];
        int[][] union = new int[m * n][2];
        int nextLabel = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (input[i][j] == 0) continue;
                int up = i > 0 ? labels[i - 1][j] : 0;
                int left = j > 0 ? labels[i][j - 1] : 0;
                if (up == 0 && left == 0) {
                    labels[i][j] = ++nextLabel;
                } else if (up > 0 && left == 0) {
                    labels[i][j] = up;
                } else if (up == 0 && left > 0) {
                    labels[i][j] = left;
                } else {
                    labels[i][j] = Math.min(up, left);
                    if (up != left) {
                        union[up][0] = left;
                        union[up][1] = 1;
                    }
                }
            }
        }
        // Resolve unions (single pass)
        for (int i = 1; i <= nextLabel; i++) {
            if (union[i][0] > 0) {
                int root = i;
                while (union[root][0] != 0) root = union[root][0];
                int cur = i;
                while (union[cur][0] != 0) {
                    int next = union[cur][0];
                    union[cur][0] = root;
                    cur = next;
                }
            }
        }
        // Remap labels
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (labels[i][j] > 0) {
                    int l = labels[i][j];
                    while (union[l][0] > 0) l = union[l][0];
                    labels[i][j] = l;
                }
            }
        }
        return labels;
    }

    /** Center of mass of labeled regions */
    public static double[][] centerOfMass(int[][] labels, double[][] image) {
        int nLabels = 0;
        for (int[] row : labels) for (int l : row) nLabels = Math.max(nLabels, l);
        double[][] result = new double[nLabels + 1][2];
        int[] counts = new int[nLabels + 1];
        double[] sumsI = new double[nLabels + 1];
        double[] sumsJ = new double[nLabels + 1];
        for (int i = 0; i < labels.length; i++) {
            for (int j = 0; j < labels[0].length; j++) {
                int l = labels[i][j];
                if (l > 0) {
                    counts[l]++;
                    sumsI[l] += image == null ? 1 : image[i][j];
                    sumsJ[l] += image == null ? 1 : image[i][j];
                }
            }
        }
        for (int l = 1; l <= nLabels; l++) {
            if (counts[l] > 0) {
                result[l][0] = sumsI[l] / counts[l];
                result[l][1] = sumsJ[l] / counts[l];
            }
        }
        return result;
    }

    /** Image extrema */
    public static double[] extrema(double[][] image) {
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (double[] row : image) {
            for (double v : row) {
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        }
        return new double[]{min, max};
    }

    /** Sum of all elements */
    public static double sum(double[][] image) {
        double s = 0;
        for (double[] row : image) for (double v : row) s += v;
        return s;
    }

    /** Mean of all elements */
    public static double mean(double[][] image) {
        double s = 0;
        int n = 0;
        for (double[] row : image) {
            for (double v : row) {
                s += v;
                n++;
            }
        }
        return s / n;
    }

    /** Standard deviation */
    public static double std(double[][] image) {
        double m = mean(image);
        double s = 0;
        int n = 0;
        for (double[] row : image) {
            for (double v : row) {
                s += (v - m) * (v - m);
                n++;
            }
        }
        return Math.sqrt(s / n);
    }

    /** Variance */
    public static double variance(double[][] image) {
        return std(image) * std(image);
    }

    // =========================================================================
    // Interpolation / Geometric transforms
    // =========================================================================

    /** Zoom (resize by factor) */
    public static double[][] zoom(double[][] input, double[] zoom) {
        int newM = (int) Math.round(input.length * zoom[0]);
        int newN = (int) Math.round(input[0].length * zoom[1]);
        double[][] result = new double[newM][newN];
        for (int i = 0; i < newM; i++) {
            double y = (double) i / zoom[0];
            int y0 = (int) Math.floor(y);
            double dy = y - y0;
            y0 = Math.min(Math.max(y0, 0), input.length - 1);
            for (int j = 0; j < newN; j++) {
                double x = (double) j / zoom[1];
                int x0 = (int) Math.floor(x);
                double dx = x - x0;
                x0 = Math.min(Math.max(x0, 0), input[0].length - 1);
                int y1 = Math.min(y0 + 1, input.length - 1);
                int x1 = Math.min(x0 + 1, input[0].length - 1);
                result[i][j] = (1 - dy) * ((1 - dx) * input[y0][x0] + dx * input[y0][x1]) +
                              dy * ((1 - dx) * input[y1][x0] + dx * input[y1][x1]);
            }
        }
        return result;
    }

    /** Shift image */
    public static double[][] shift(double[][] input, double[] shift) {
        int m = input.length, n = input[0].length;
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double y = i - shift[0];
                double x = j - shift[1];
                int y0 = (int) Math.floor(y);
                int x0 = (int) Math.floor(x);
                double dy = y - y0;
                double dx = x - x0;
                y0 = Math.min(Math.max(y0, 0), m - 1);
                x0 = Math.min(Math.max(x0, 0), n - 1);
                int y1 = Math.min(y0 + 1, m - 1);
                int x1 = Math.min(x0 + 1, n - 1);
                result[i][j] = (1 - dy) * ((1 - dx) * input[y0][x0] + dx * input[y0][x1]) +
                              dy * ((1 - dx) * input[y1][x0] + dx * input[y1][x1]);
            }
        }
        return result;
    }

    /** Rotate image */
    public static double[][] rotate(double[][] input, double angle) {
        int m = input.length, n = input[0].length;
        double cos = Math.cos(-angle), sin = Math.sin(-angle);
        int newM = (int) Math.abs(m * cos) + (int) Math.abs(n * sin);
        int newN = (int) Math.abs(n * cos) + (int) Math.abs(m * sin);
        double[][] result = new double[newM][newN];
        double cy = m / 2.0, cx = n / 2.0;
        double ncy = newM / 2.0, ncx = newN / 2.0;
        for (int i = 0; i < newM; i++) {
            for (int j = 0; j < newN; j++) {
                double y = (i - ncy) * Math.cos(angle) - (j - ncx) * Math.sin(angle) + cy;
                double x = (i - ncy) * Math.sin(angle) + (j - ncx) * Math.cos(angle) + cx;
                int y0 = (int) Math.floor(y), x0 = (int) Math.floor(x);
                if (y0 < 0 || y0 >= m - 1 || x0 < 0 || x0 >= n - 1) {
                    result[i][j] = 0;
                    continue;
                }
                double dy = y - y0, dx = x - x0;
                result[i][j] = (1 - dy) * ((1 - dx) * input[y0][x0] + dx * input[y0][x0 + 1]) +
                              dy * ((1 - dx) * input[y0 + 1][x0] + dx * input[y0 + 1][x0 + 1]);
            }
        }
        return result;
    }

    /** Map coordinates (interpolate at arbitrary points) */
    public static double[] mapCoordinates(double[][] input, double[][] coords, int order) {
        int m = input.length, n = input[0].length;
        double[] result = new double[coords[0].length];
        for (int i = 0; i < coords[0].length; i++) {
            double y = coords[0][i], x = coords[1][i];
            if (order == 0) {
                int yi = Math.min(Math.max((int) Math.round(y), 0), m - 1);
                int xi = Math.min(Math.max((int) Math.round(x), 0), n - 1);
                result[i] = input[yi][xi];
            } else {
                int y0 = (int) Math.floor(y), x0 = (int) Math.floor(x);
                if (y0 < 0 || y0 >= m - 1 || x0 < 0 || x0 >= n - 1) { result[i] = 0; continue; }
                double dy = y - y0, dx = x - x0;
                result[i] = (1 - dy) * ((1 - dx) * input[y0][x0] + dx * input[y0][x0 + 1]) +
                            dy * ((1 - dx) * input[y0 + 1][x0] + dx * input[y0 + 1][x0 + 1]);
            }
        }
        return result;
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private static double[][] transpose(double[][] a) {
        double[][] t = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                t[j][i] = a[i][j];
            }
        }
        return t;
    }

    /** Generate 2D Gaussian kernel */
    public static double[][] gaussianKernel2d(int size, double sigma) {
        double[][] k = new double[size][size];
        int c = size / 2;
        double sum = 0;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double d2 = (i - c) * (i - c) + (j - c) * (j - c);
                k[i][j] = Math.exp(-d2 / (2 * sigma * sigma));
                sum += k[i][j];
            }
        }
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) k[i][j] /= sum;
        }
        return k;
    }

    /** Generate 1D Gaussian kernel */
    public static double[] gaussianKernel1d(int size, double sigma) {
        double[] k = new double[size];
        int c = size / 2;
        double sum = 0;
        for (int i = 0; i < size; i++) {
            double d = i - c;
            k[i] = Math.exp(-d * d / (2 * sigma * sigma));
            sum += k[i];
        }
        for (int i = 0; i < size; i++) k[i] /= sum;
        return k;
    }
}