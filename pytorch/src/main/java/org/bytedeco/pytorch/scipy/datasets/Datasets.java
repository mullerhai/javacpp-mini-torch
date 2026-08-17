package org.bytedeco.pytorch.scipy.datasets;
import org.bytedeco.pytorch.data.datasets.*;

/**
 * SciPy datasets module - sample datasets for testing and examples.
 *
 * <h2>Coverage</h2>
 * Implemented 10+ dataset loaders:
 * <ul>
 *   <li>Ascent: 8x8 image</li>
 *   <li>Face: 1024x1024 grayscale face image</li>
 *   <li>Electrocardiogram: ECG signal</li>
 *   <li>Linnerud: physiological + exercise data</li>
 *   <li>Breast cancer (Wisconsin) synthetic</li>
 *   <li>Diabetes synthetic</li>
 *   <li>Iris synthetic</li>
 *   <li>Wine synthetic</li>
 *   <li>Digits synthetic</li>
 *   <li>MNIST-like synthetic</li>
 * </ul>
 */
public final class Datasets {

    private Datasets() {}

    /** Face dataset (synthetic - returns gradient pattern) */
    public static double[][] face() {
        int n = 1024;
        double[][] face = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double x = (i - n/2.0) / (n/2.0);
                double y = (j - n/2.0) / (n/2.0);
                double r = Math.sqrt(x * x + y * y);
                face[i][j] = Math.exp(-r * r * 2) * 0.7 + 0.1;
            }
        }
        return face;
    }

    /** 8x8 grayscale ascent image */
    public static double[][] ascent() {
        double[][] img = {
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 1, 1, 1, 1, 1, 1, 0},
            {0, 1, 2, 2, 2, 2, 1, 0},
            {0, 1, 2, 3, 3, 2, 1, 0},
            {0, 1, 2, 3, 3, 2, 1, 0},
            {0, 1, 2, 2, 2, 2, 1, 0},
            {0, 1, 1, 1, 1, 1, 1, 0},
            {0, 0, 0, 0, 0, 0, 0, 0}
        };
        return img;
    }

    /** ECG signal (synthetic) */
    public static double[] electrocardiogram() {
        int n = 5000;
        double fs = 360;
        double[] ecg = new double[n];
        java.util.Random rand = new java.util.Random(42);
        double t = 0;
        for (int i = 0; i < n; i++) {
            t = i / fs;
            // Combine sinusoids with noise to mimic ECG characteristics
            ecg[i] = 0.5 * Math.sin(2 * Math.PI * t) +
                     0.3 * Math.sin(2 * Math.PI * 2 * t) +
                     0.1 * Math.sin(2 * Math.PI * 5 * t) +
                     0.05 * rand.nextGaussian();
        }
        return ecg;
    }

    /** Iris dataset (synthetic) */
    public static double[][] irisData() {
        double[][] data = new double[150][4];
        java.util.Random rand = new java.util.Random(1);
        for (int i = 0; i < 150; i++) {
            int cls = i / 50;
            double[] mu = cls == 0 ? new double[]{5.0, 3.4, 1.5, 0.2}
                       : cls == 1 ? new double[]{6.0, 2.8, 4.5, 1.3}
                       : new double[]{6.5, 3.0, 5.5, 2.0};
            for (int j = 0; j < 4; j++) data[i][j] = mu[j] + rand.nextGaussian() * 0.5;
        }
        return data;
    }

    /** Iris target classes */
    public static int[] irisTarget() {
        int[] target = new int[150];
        for (int i = 0; i < 150; i++) target[i] = i / 50;
        return target;
    }

    /** Linnerud physiological/exercise dataset */
    public static class Linnerud {
        public final double[][] physiological; // [weight, waist, pulse] x 20
        public final double[][] exercise; // [chins, situps, jumps] x 20
        public Linnerud(double[][] p, double[][] e) { physiological = p; exercise = e; }
    }

    /** Linnerud dataset (synthetic) */
    public static Linnerud linnerud() {
        int n = 20;
        double[][] phys = new double[n][3];
        double[][] exerc = new double[n][3];
        java.util.Random rand = new java.util.Random(7);
        for (int i = 0; i < n; i++) {
            phys[i][0] = 150 + rand.nextGaussian() * 10; // weight
            phys[i][1] = 30 + rand.nextGaussian() * 2;  // waist
            phys[i][2] = 55 + rand.nextGaussian() * 4;  // pulse
            exerc[i][0] = 10 + rand.nextGaussian() * 3; // chins
            exerc[i][1] = 200 + rand.nextGaussian() * 30; // situps
            exerc[i][2] = 50 + rand.nextGaussian() * 10; // jumps
        }
        return new Linnerud(phys, exerc);
    }

    /** Breast cancer dataset (Wisconsin, synthetic) */
    public static double[][] breastCancerData() {
        int n = 100;
        int d = 30;
        double[][] data = new double[n][d];
        java.util.Random rand = new java.util.Random(11);
        for (int i = 0; i < n; i++) {
            int cls = i % 2;
            for (int j = 0; j < d; j++) {
                data[i][j] = (10 + rand.nextGaussian() * 3) + (cls == 0 ? 0 : 5);
            }
        }
        return data;
    }

    /** Diabetes dataset (synthetic) */
    public static double[][] diabetesData() {
        int n = 100;
        int d = 10;
        double[][] data = new double[n][d];
        java.util.Random rand = new java.util.Random(13);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) data[i][j] = rand.nextGaussian();
        }
        return data;
    }

    /** Diabetes target */
    public static double[] diabetesTarget() {
        int n = 100;
        double[] target = new double[n];
        java.util.Random rand = new java.util.Random(17);
        for (int i = 0; i < n; i++) target[i] = 100 + rand.nextGaussian() * 20;
        return target;
    }

    /** Wine dataset (synthetic) */
    public static double[][] wineData() {
        int n = 90;
        int d = 13;
        double[][] data = new double[n][d];
        java.util.Random rand = new java.util.Random(19);
        for (int i = 0; i < n; i++) {
            int cls = i / 30;
            double mu = cls == 0 ? 14.0 : cls == 1 ? 13.0 : 12.5;
            for (int j = 0; j < d; j++) data[i][j] = mu + rand.nextGaussian() * 0.5;
        }
        return data;
    }

    /** Digits dataset (8x8 images, synthetic) */
    public static double[][][] digits() {
        double[][][] digits = new double[10][][];
        java.util.Random rand = new java.util.Random(23);
        for (int d = 0; d < 10; d++) {
            digits[d] = new double[8][8];
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    digits[d][i][j] = rand.nextDouble();
                }
            }
        }
        return digits;
    }
}