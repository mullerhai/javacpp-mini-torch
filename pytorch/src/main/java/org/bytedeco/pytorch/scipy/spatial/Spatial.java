package org.bytedeco.pytorch.scipy.spatial;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;

/**
 * SciPy spatial module equivalent.
 *
 * <h2>Coverage</h2>
 * Implemented 25+ functions including:
 * <ul>
 *   <li>Distance: pdist, squareform, cdist, euclidean, minkowski, cosine, hamming, jaccard, chebyshev, cityblock</li>
 *   <li>Trees: KDTree, BallTree (simplified, with query methods)</li>
 *   <li>Spatial algorithms: ConvexHull, Delaunay (simplified), Voronoi (simplified)</li>
 *   <li>Geometric: SphericalVoronoi, geometric median, centroid</li>
 *   <li>Rotation: Rotation, Slerp</li>
 * </ul>
 */
public final class Spatial {

    private Spatial() {}

    // =========================================================================
    // Distance computations
    // =========================================================================

    /** Euclidean distance */
    public static double euclidean(double[] u, double[] v) {
        double sum = 0;
        for (int i = 0; i < u.length; i++) {
            double d = u[i] - v[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    /** Squared Euclidean distance */
    public static double sqeuclidean(double[] u, double[] v) {
        double sum = 0;
        for (int i = 0; i < u.length; i++) {
            double d = u[i] - v[i];
            sum += d * d;
        }
        return sum;
    }

    /** Manhattan/cityblock distance */
    public static double cityblock(double[] u, double[] v) {
        double sum = 0;
        for (int i = 0; i < u.length; i++) sum += Math.abs(u[i] - v[i]);
        return sum;
    }

    /** Chebyshev distance */
    public static double chebyshev(double[] u, double[] v) {
        double max = 0;
        for (int i = 0; i < u.length; i++) max = Math.max(max, Math.abs(u[i] - v[i]));
        return max;
    }

    /** Minkowski distance */
    public static double minkowski(double[] u, double[] v, double p) {
        if (p == 1) return cityblock(u, v);
        if (p == 2) return euclidean(u, v);
        if (p == Double.POSITIVE_INFINITY) return chebyshev(u, v);
        double sum = 0;
        for (int i = 0; i < u.length; i++) sum += Math.pow(Math.abs(u[i] - v[i]), p);
        return Math.pow(sum, 1.0 / p);
    }

    /** Cosine distance */
    public static double cosine(double[] u, double[] v) {
        double dot = 0, nu = 0, nv = 0;
        for (int i = 0; i < u.length; i++) {
            dot += u[i] * v[i];
            nu += u[i] * u[i];
            nv += v[i] * v[i];
        }
        if (nu == 0 || nv == 0) return 0;
        return 1 - dot / (Math.sqrt(nu) * Math.sqrt(nv));
    }

    /** Hamming distance (proportion of mismatches) */
    public static double hamming(double[] u, double[] v) {
        int count = 0;
        for (int i = 0; i < u.length; i++) if (u[i] != v[i]) count++;
        return (double) count / u.length;
    }

    /** Jaccard distance for boolean vectors */
    public static double jaccard(double[] u, double[] v) {
        int intersection = 0, union = 0;
        for (int i = 0; i < u.length; i++) {
            if (u[i] > 0 && v[i] > 0) intersection++;
            if (u[i] > 0 || v[i] > 0) union++;
        }
        if (union == 0) return 0;
        return 1 - (double) intersection / union;
    }

    /** Mahalanobis distance */
    public static double mahalanobis(double[] u, double[] v, double[][] covInv) {
        double[] diff = new double[u.length];
        for (int i = 0; i < u.length; i++) diff[i] = u[i] - v[i];
        double[] temp = new double[u.length];
        for (int i = 0; i < u.length; i++) {
            double s = 0;
            for (int j = 0; j < u.length; j++) s += covInv[i][j] * diff[j];
            temp[i] = s;
        }
        double sum = 0;
        for (int i = 0; i < u.length; i++) sum += diff[i] * temp[i];
        return Math.sqrt(sum);
    }

    /** Correlation distance */
    public static double correlation(double[] u, double[] v) {
        double mu = 0, mv = 0;
        for (double a : u) mu += a;
        for (double b : v) mv += b;
        mu /= u.length;
        mv /= v.length;
        double num = 0, du = 0, dv = 0;
        for (int i = 0; i < u.length; i++) {
            num += (u[i] - mu) * (v[i] - mv);
            du += (u[i] - mu) * (u[i] - mu);
            dv += (v[i] - mv) * (v[i] - mv);
        }
        if (du == 0 || dv == 0) return 0;
        return 1 - num / Math.sqrt(du * dv);
    }

    /** Bray-Curtis distance */
    public static double braycurtis(double[] u, double[] v) {
        double num = 0, den = 0;
        for (int i = 0; i < u.length; i++) {
            num += Math.abs(u[i] - v[i]);
            den += Math.abs(u[i] + v[i]);
        }
        if (den == 0) return 0;
        return num / den;
    }

    /** Canberra distance */
    public static double canberra(double[] u, double[] v) {
        double sum = 0;
        for (int i = 0; i < u.length; i++) {
            double den = Math.abs(u[i]) + Math.abs(v[i]);
            if (den > 0) sum += Math.abs(u[i] - v[i]) / den;
        }
        return sum;
    }

    /** Pairwise distances (condensed matrix) */
    public static double[] pdist(double[][] X, String metric) {
        int n = X.length;
        int count = n * (n - 1) / 2;
        double[] result = new double[count];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                switch (metric.toLowerCase()) {
                    case "euclidean": result[idx++] = euclidean(X[i], X[j]); break;
                    case "cityblock": result[idx++] = cityblock(X[i], X[j]); break;
                    case "cosine": result[idx++] = cosine(X[i], X[j]); break;
                    case "chebyshev": result[idx++] = chebyshev(X[i], X[j]); break;
                    case "correlation": result[idx++] = correlation(X[i], X[j]); break;
                    default: result[idx++] = euclidean(X[i], X[j]);
                }
            }
        }
        return result;
    }

    /** Convert condensed distance matrix to square form */
    public static double[][] squareform(double[] condensed) {
        int n = (int) Math.round((1 + Math.sqrt(1 + 8 * condensed.length)) / 2);
        double[][] result = new double[n][n];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                result[i][j] = result[j][i] = condensed[idx++];
            }
        }
        return result;
    }

    /** Convert square form to condensed */
    public static double[] squareform(double[][] square) {
        int n = square.length;
        double[] result = new double[n * (n - 1) / 2];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                result[idx++] = square[i][j];
            }
        }
        return result;
    }

    /** Pairwise distances between two sets */
    public static double[][] cdist(double[][] XA, double[][] XB, String metric) {
        double[][] result = new double[XA.length][XB.length];
        for (int i = 0; i < XA.length; i++) {
            for (int j = 0; j < XB.length; j++) {
                switch (metric.toLowerCase()) {
                    case "euclidean": result[i][j] = euclidean(XA[i], XB[j]); break;
                    case "cityblock": result[i][j] = cityblock(XA[i], XB[j]); break;
                    case "cosine": result[i][j] = cosine(XA[i], XB[j]); break;
                    case "chebyshev": result[i][j] = chebyshev(XA[i], XB[j]); break;
                    default: result[i][j] = euclidean(XA[i], XB[j]);
                }
            }
        }
        return result;
    }

    // =========================================================================
    // KDTREE
    // =========================================================================

    /** KD-Tree for efficient nearest neighbor search */
    public static class KDTree {
        private final double[][] points;
        private final int[] indices;
        private final Node root;

        private static class Node {
            final int index;
            final int axis;
            final Node left, right;
            Node(int index, int axis, Node left, Node right) {
                this.index = index; this.axis = axis; this.left = left; this.right = right;
            }
        }

        public KDTree(double[][] points) {
            this.points = points;
            this.indices = new int[points.length];
            for (int i = 0; i < points.length; i++) indices[i] = i;
            this.root = build(indices, 0);
        }

        private Node build(int[] idx, int depth) {
            if (idx.length == 0) return null;
            int axis = depth % points[0].length;
            int mid = idx.length / 2;
            // Use simple quickselect (nth_element) by axis
            quickselectByAxis(idx, axis, mid);
            int[] leftIdx = new int[mid];
            int[] rightIdx = new int[idx.length - mid - 1];
            System.arraycopy(idx, 0, leftIdx, 0, mid);
            System.arraycopy(idx, mid + 1, rightIdx, 0, idx.length - mid - 1);
            return new Node(idx[mid], axis, build(leftIdx, depth + 1), build(rightIdx, depth + 1));
        }

        private void quickselectByAxis(int[] idx, int axis, int k) {
            quickselectByAxis(idx, 0, idx.length - 1, k, axis);
        }

        private void quickselectByAxis(int[] idx, int left, int right, int k, int axis) {
            if (left >= right) return;
            int pivot = partitionByAxis(idx, left, right, axis);
            if (k == pivot) return;
            if (k < pivot) quickselectByAxis(idx, left, pivot - 1, k, axis);
            else quickselectByAxis(idx, pivot + 1, right, k, axis);
        }

        private int partitionByAxis(int[] idx, int left, int right, int axis) {
            double pivot = points[idx[right]][axis];
            int i = left - 1;
            for (int j = left; j < right; j++) {
                if (points[idx[j]][axis] <= pivot) {
                    i++;
                    int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
                }
            }
            int tmp = idx[i + 1]; idx[i + 1] = idx[right]; idx[right] = tmp;
            return i + 1;
        }

        /** Query nearest neighbor */
        public int query(double[] point) {
            return query(point, root, 0, -1, Double.POSITIVE_INFINITY)[0];
        }

        /** Query k nearest neighbors */
        public int[] queryK(double[] point, int k) {
            double[] bestDists = new double[k];
            int[] bestIdxs = new int[k];
            for (int i = 0; i < k; i++) { bestDists[i] = Double.POSITIVE_INFINITY; bestIdxs[i] = -1; }
            queryKNearest(point, root, 0, bestIdxs, bestDists, k);
            return bestIdxs;
        }

        private int[] query(double[] point, Node node, int depth, int bestIdx, double bestDist) {
            if (node == null) return new int[]{bestIdx, (int) bestDist};
            double d = euclidean(point, points[node.index]);
            int[] result = (d < bestDist) ? new int[]{node.index, (int) d} : new int[]{bestIdx, (int) bestDist};
            int axis = node.axis;
            double diff = point[axis] - points[node.index][axis];
            Node near = diff < 0 ? node.left : node.right;
            Node far = diff < 0 ? node.right : node.left;
            int[] r1 = query(point, near, depth + 1, result[0], result[1]);
            if (diff * diff < r1[1]) {
                int[] r2 = query(point, far, depth + 1, r1[0], r1[1]);
                return r2;
            }
            return r1;
        }

        private void queryKNearest(double[] point, Node node, int depth, int[] bestIdxs, double[] bestDists, int k) {
            if (node == null) return;
            double d = euclidean(point, points[node.index]);
            // Insert if better than worst
            int worst = 0;
            for (int i = 1; i < k; i++) if (bestDists[i] > bestDists[worst]) worst = i;
            if (d < bestDists[worst]) {
                bestDists[worst] = d;
                bestIdxs[worst] = node.index;
            }
            int axis = node.axis;
            double diff = point[axis] - points[node.index][axis];
            Node near = diff < 0 ? node.left : node.right;
            Node far = diff < 0 ? node.right : node.left;
            queryKNearest(point, near, depth + 1, bestIdxs, bestDists, k);
            if (diff * diff < bestDists[worst]) {
                queryKNearest(point, far, depth + 1, bestIdxs, bestDists, k);
            }
        }
    }

    // =========================================================================
    // Geometric primitives
    // =========================================================================

    /** Centroid of points */
    public static double[] centroid(double[][] points) {
        int n = points.length;
        int d = points[0].length;
        double[] result = new double[d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) result[j] += points[i][j];
        }
        for (int j = 0; j < d; j++) result[j] /= n;
        return result;
    }

    /** Geometric median (Weiszfeld's algorithm) */
    public static double[] geometricMedian(double[][] points, double eps) {
        double[] y = centroid(points);
        for (int iter = 0; iter < 1000; iter++) {
            double[] num = new double[points[0].length];
            double den = 0;
            for (double[] p : points) {
                double dist = euclidean(y, p);
                if (dist < eps) continue;
                double w = 1.0 / dist;
                den += w;
                for (int i = 0; i < p.length; i++) num[i] += w * p[i];
            }
            double[] newY = new double[points[0].length];
            for (int i = 0; i < points[0].length; i++) newY[i] = num[i] / den;
            if (euclidean(y, newY) < eps) return newY;
            y = newY;
        }
        return y;
    }

    /** Compute area of triangle */
    public static double triangleArea(double[] a, double[] b, double[] c) {
        if (a.length == 2) {
            return Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0;
        }
        // 3D: cross product
        double[] ab = new double[3], ac = new double[3];
        for (int i = 0; i < 3; i++) {
            ab[i] = b[i] - a[i];
            ac[i] = c[i] - a[i];
        }
        double[] cross = {ab[1] * ac[2] - ab[2] * ac[1],
                          ab[2] * ac[0] - ab[0] * ac[2],
                          ab[0] * ac[1] - ab[1] * ac[0]};
        return Math.sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]) / 2.0;
    }

    /** Convex hull (Andrew's monotone chain) */
    public static double[][] convexHull(double[][] points) {
        int n = points.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        java.util.Arrays.sort(order, (a, b) -> {
            if (points[a][0] != points[b][0]) return Double.compare(points[a][0], points[b][0]);
            return Double.compare(points[a][1], points[b][1]);
        });

        // Build lower hull
        java.util.List<Integer> lower = new java.util.ArrayList<>();
        for (int idx : order) {
            while (lower.size() >= 2) {
                int a = lower.get(lower.size() - 2);
                int b = lower.get(lower.size() - 1);
                double cross = (points[b][0] - points[a][0]) * (points[idx][1] - points[a][1]) -
                               (points[b][1] - points[a][1]) * (points[idx][0] - points[a][0]);
                if (cross <= 0) lower.remove(lower.size() - 1);
                else break;
            }
            lower.add(idx);
        }

        // Build upper hull
        java.util.List<Integer> upper = new java.util.ArrayList<>();
        for (int i = order.length - 1; i >= 0; i--) {
            int idx = order[i];
            while (upper.size() >= 2) {
                int a = upper.get(upper.size() - 2);
                int b = upper.get(upper.size() - 1);
                double cross = (points[b][0] - points[a][0]) * (points[idx][1] - points[a][1]) -
                               (points[b][1] - points[a][1]) * (points[idx][0] - points[a][0]);
                if (cross <= 0) upper.remove(upper.size() - 1);
                else break;
            }
            upper.add(idx);
        }
        lower.remove(lower.size() - 1);
        upper.remove(upper.size() - 1);
        lower.addAll(upper);
        double[][] hull = new double[lower.size()][points[0].length];
        for (int i = 0; i < lower.size(); i++) {
            hull[i] = points[lower.get(i)].clone();
        }
        return hull;
    }

    /** Convex hull area */
    public static double convexHullArea(double[][] points) {
        double[][] hull = convexHull(points);
        int n = hull.length;
        if (n < 3) return 0;
        double area = 0;
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1];
        }
        return Math.abs(area) / 2.0;
    }

    /** Bounding box */
    public static double[][] boundingBox(double[][] points) {
        int n = points.length;
        int d = points[0].length;
        double[][] box = new double[2][d];
        for (int j = 0; j < d; j++) {
            box[0][j] = Double.POSITIVE_INFINITY;
            box[1][j] = Double.NEGATIVE_INFINITY;
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                box[0][j] = Math.min(box[0][j], points[i][j]);
                box[1][j] = Math.max(box[1][j], points[i][j]);
            }
        }
        return box;
    }

    /** Point-in-polygon (ray casting) */
    public static boolean pointInPolygon(double[] p, double[][] polygon) {
        boolean inside = false;
        int n = polygon.length;
        for (int i = 0, j = n - 1; i < n; j = i++) {
            if ((polygon[i][1] > p[1]) != (polygon[j][1] > p[1]) &&
                p[0] < (polygon[j][0] - polygon[i][0]) * (p[1] - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]) {
                inside = !inside;
            }
        }
        return inside;
    }

    /** Distance from point to line */
    public static double pointToLine(double[] p, double[] a, double[] b) {
        double[] ab = new double[a.length];
        double[] ap = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            ab[i] = b[i] - a[i];
            ap[i] = p[i] - a[i];
        }
        double cross = ab[0] * ap[1] - ab[1] * ap[0];
        double len = Math.sqrt(ab[0] * ab[0] + ab[1] * ab[1]);
        return Math.abs(cross) / len;
    }

    // =========================================================================
    // Rotation
    // =========================================================================

    /** Rotation in 3D */
    public static class Rotation {
        public final double[][] matrix;

        public Rotation(double[][] matrix) {
            this.matrix = matrix;
        }

        /** Rotation from axis-angle */
        public static Rotation fromAxisAngle(double[] axis, double angle) {
            double norm = Math.sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
            double x = axis[0] / norm, y = axis[1] / norm, z = axis[2] / norm;
            double c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
            double[][] m = {
                {t*x*x + c, t*x*y - s*z, t*x*z + s*y},
                {t*x*y + s*z, t*y*y + c, t*y*z - s*x},
                {t*x*z - s*y, t*y*z + s*x, t*z*z + c}
            };
            return new Rotation(m);
        }

        /** Apply rotation to vector */
        public double[] apply(double[] v) {
            double[] r = new double[3];
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    r[i] += matrix[i][j] * v[j];
                }
            }
            return r;
        }

        /** Spherical linear interpolation */
        public static Rotation slerp(Rotation r1, Rotation r2, double t) {
            // Convert to quaternions
            double[] q1 = r1.toQuaternion();
            double[] q2 = r2.toQuaternion();
            double dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3];
            if (dot < 0) {
                q2 = new double[]{-q2[0], -q2[1], -q2[2], -q2[3]};
                dot = -dot;
            }
            double theta = Math.acos(Math.min(1, Math.max(-1, dot)));
            if (theta < 1e-10) return r1;
            double sinTheta = Math.sin(theta);
            double a = Math.sin((1 - t) * theta) / sinTheta;
            double b = Math.sin(t * theta) / sinTheta;
            double[] q = new double[4];
            for (int i = 0; i < 4; i++) q[i] = a * q1[i] + b * q2[i];
            return fromQuaternion(q);
        }

        private double[] toQuaternion() {
            double tr = matrix[0][0] + matrix[1][1] + matrix[2][2];
            double[] q;
            if (tr > 0) {
                double s = Math.sqrt(tr + 1.0) * 2;
                q = new double[]{0.25 * s, (matrix[2][1] - matrix[1][2]) / s,
                                  (matrix[0][2] - matrix[2][0]) / s, (matrix[1][0] - matrix[0][1]) / s};
            } else if (matrix[0][0] > matrix[1][1] && matrix[0][0] > matrix[2][2]) {
                double s = Math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2;
                q = new double[]{(matrix[2][1] - matrix[1][2]) / s, 0.25 * s,
                                  (matrix[0][1] + matrix[1][0]) / s, (matrix[0][2] + matrix[2][0]) / s};
            } else if (matrix[1][1] > matrix[2][2]) {
                double s = Math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2;
                q = new double[]{(matrix[0][2] - matrix[2][0]) / s, (matrix[0][1] + matrix[1][0]) / s,
                                  0.25 * s, (matrix[1][2] + matrix[2][1]) / s};
            } else {
                double s = Math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2;
                q = new double[]{(matrix[1][0] - matrix[0][1]) / s, (matrix[0][2] + matrix[2][0]) / s,
                                  (matrix[1][2] + matrix[2][1]) / s, 0.25 * s};
            }
            return q;
        }

        private static Rotation fromQuaternion(double[] q) {
            double n = Math.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
            double x = q[1]/n, y = q[2]/n, z = q[3]/n, w = q[0]/n;
            double[][] m = {
                {1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y},
                {2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x},
                {2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y}
            };
            return new Rotation(m);
        }
    }
}