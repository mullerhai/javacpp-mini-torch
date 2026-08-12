/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.distributed;
import org.bytedeco.pytorch.nn.*;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.empty;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Shared low-level helpers used by every distributed trainer in this package.
 *
 * <p>These helpers wrap the small, repetitive code paths that show up in
 * {@link DDPTrainer}, {@link FSDPTrainer}, {@link NativeDDPTrainer},
 * {@link NativeFSDPTrainer} and the new ZeRO / Expert / Sequence parallel
 * trainers: flattening parameter lists, padding to a world-aligned size,
 * reading / writing a flat float tensor to disk, building a contiguous
 * flat parameter buffer, gathering gradients and writing them back to a
 * module.
 *
 * <p>They are deliberately side-effect free, GC-friendly (every temporary
 * tensor is closed) and work in both real-backend and {@code local} mode
 * (single-process smoke).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class TrainerOps {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private TrainerOps() {}

    // ── Parameter / gradient collection ────────────────────────────────────

    /** Copy the live, non-null parameters of a module into a fresh list. */
    public static List<Tensor> collectParameters(org.bytedeco.pytorch.nn.Module module) {
        Objects.requireNonNull(module, "module");
        List<Tensor> out = new ArrayList<>();
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p != null && !p.isNull()) {
                out.add(p);
            }
        }
        return out;
    }

    /** Sum of {@link Tensor#numel()} over a parameter list (handles nulls). */
    public static long totalNumel(Collection<Tensor> tensors) {
        long total = 0;
        for (Tensor t : tensors) {
            if (t != null && !t.isNull()) {
                total += t.numel();
            }
        }
        return total;
    }

    /** Copy the live gradients of a parameter list. Missing grads become {@code null}. */
    public static List<Tensor> collectGradients(List<Tensor> params) {
        List<Tensor> grads = new ArrayList<>(params.size());
        for (Tensor p : params) {
            if (p == null || p.isNull()) {
                grads.add(null);
                continue;
            }
            try {
                Tensor g = p.grad();
                grads.add((g != null && !g.isNull() && g.defined()) ? g : null);
            } catch (Throwable ignored) {
                grads.add(null);
            }
        }
        return grads;
    }

    /** Sum of {@link Tensor#numel()} of the live gradients in a list. */
    public static long totalGradNumel(List<Tensor> grads) {
        long total = 0;
        for (Tensor g : grads) {
            if (g != null && !g.isNull() && g.defined()) {
                total += g.numel();
            }
        }
        return total;
    }

    // ── Flatten / unflatten ────────────────────────────────────────────────

    /**
     * Flatten a parameter list into a single contiguous 1D tensor of the
     * requested dtype. Returns an empty 1D tensor (zero elements) when
     * {@code params} is empty. Caller owns the returned tensor.
     */
    public static Tensor flatten(List<Tensor> params, org.bytedeco.pytorch.Device device, ScalarType dtype) {
        if (params.isEmpty()) {
            return zeros(0L).to(device, dtype);
        }
        TensorVector vec = new TensorVector();
        for (Tensor p : params) {
            if (p == null || p.isNull()) continue;
            vec.push_back(p.detach().flatten().to(device, dtype));
        }
        if (vec.size() == 0) {
            return zeros(0L).to(device, dtype);
        }
        return cat(vec).detach();
    }

    /**
     * Flatten a gradient list into a single contiguous 1D tensor of the
     * requested dtype. Parameters with no gradient are zero-padded so the
     * returned tensor has exactly {@code totalNumel} elements. Returns an
     * empty tensor when {@code totalNumel == 0}.
     */
    public static Tensor flattenGrads(List<Tensor> grads, List<Tensor> params,
                                     org.bytedeco.pytorch.Device device, ScalarType dtype) {
        long total = totalNumel(params);
        if (total == 0) {
            return zeros(0L).to(device, dtype);
        }
        TensorVector vec = new TensorVector();
        for (int i = 0; i < grads.size(); i++) {
            Tensor p = params.get(i);
            Tensor g = grads.get(i);
            if (p == null || p.isNull()) continue;
            long n = p.numel();
            if (g == null || g.isNull() || !g.defined()) {
                vec.push_back(zeros(n).to(device, dtype));
            } else {
                vec.push_back(g.detach().flatten().to(device, dtype));
            }
        }
        if (vec.size() == 0) {
            return zeros(total).to(device, dtype);
        }
        return cat(vec).detach();
    }

    /**
     * Write a flat 1D tensor back into the (live) parameters of a module
     * under a {@link org.bytedeco.pytorch.NoGradGuard}. Caller is responsible
     * for offset consistency (use {@link #writeFlatIntoParams} which does
     * this). The {@code expected} list is matched by index against the live
     * parameters of {@code module}; if they differ, the union is iterated in
     * module order and the flat tensor is read sequentially.
     */
    public static void writeFlatIntoParams(org.bytedeco.pytorch.nn.Module module, Tensor flat) {
        Objects.requireNonNull(module, "module");
        Objects.requireNonNull(flat, "flat");
        try (org.bytedeco.pytorch.NoGradGuard guard = new org.bytedeco.pytorch.NoGradGuard()) {
            long offset = 0;
            TensorVector params = module.parameters();
            for (long i = 0, n = params.size(); i < n; i++) {
                Tensor t = params.get(i);
                if (t == null || t.isNull()) continue;
                long num = t.numel();
                if (offset + num > flat.numel()) break;
                Tensor src = flat.narrow(0, offset, num).view(t.sizes());
                try {
                    t.copy_(src);
                } catch (Throwable ignored) {
                    // Param may be on a different device or layout; skip silently.
                }
                src.close();
                offset += num;
            }
        }
    }

    /**
     * Write a flat 1D tensor into a pre-known list of parameter tensors
     * (used by ZeRO when module.parameters() is not the canonical source).
     */
    public static void writeFlatIntoList(List<Tensor> targets, Tensor flat) {
        Objects.requireNonNull(targets, "targets");
        Objects.requireNonNull(flat, "flat");
        try (org.bytedeco.pytorch.NoGradGuard guard = new org.bytedeco.pytorch.NoGradGuard()) {
            long offset = 0;
            for (Tensor t : targets) {
                if (t == null || t.isNull()) continue;
                long num = t.numel();
                if (offset + num > flat.numel()) break;
                Tensor src = flat.narrow(0, offset, num).view(t.sizes());
                try {
                    t.copy_(src);
                } catch (Throwable ignored) {
                }
                src.close();
                offset += num;
            }
        }
    }

    // ── Padding ────────────────────────────────────────────────────────────

    /**
     * Pad (with zeros) a 1D tensor up to {@code targetSize} elements and
     * return a new tensor. If the input already has the target size, the
     * input is returned without copying (caller still owns it). If it is
     * larger, a {@code narrow} view is returned. The returned tensor is a
     * brand new contiguous tensor (caller owns it).
     */
    public static Tensor pad1D(Tensor src, long targetSize,
                               org.bytedeco.pytorch.Device device, ScalarType dtype) {
        if (src == null || src.isNull()) {
            return zeros(targetSize).to(device, dtype);
        }
        if (src.numel() == targetSize) {
            return src.contiguous().to(device, dtype);
        }
        if (src.numel() > targetSize) {
            return src.narrow(0, 0, targetSize).contiguous().to(device, dtype);
        }
        Tensor pad = zeros(targetSize - src.numel()).to(device, dtype);
        TensorVector v = new TensorVector();
        v.push_back(src.to(device, dtype));
        v.push_back(pad);
        Tensor out = cat(v);
        pad.close();
        return out;
    }

    /** Trim a 1D tensor down to {@code targetSize} (returns a fresh view). */
    public static Tensor trim1D(Tensor src, long targetSize) {
        if (src == null || src.isNull()) {
            return zeros(targetSize);
        }
        if (src.numel() <= targetSize) {
            return src;
        }
        return src.narrow(0, 0, targetSize).contiguous();
    }

    // ── Buffers & uniform tensors ─────────────────────────────────────────

    /** Allocate an uninitialised 1D tensor on the requested device. */
    public static Tensor empty1D(long size, org.bytedeco.pytorch.Device device, ScalarType dtype) {
        if (size <= 0) {
            return zeros(0L).to(device, dtype);
        }
        return empty(size).to(device, dtype);
    }

    /** Allocate a zeros 1D tensor on the requested device. */
    public static Tensor zeros1D(long size, org.bytedeco.pytorch.Device device, ScalarType dtype) {
        if (size <= 0) {
            return zeros(0L).to(device, dtype);
        }
        return zeros(size).to(device, dtype);
    }

    // ── Gradient averaging / cloning ──────────────────────────────────────

    /**
     * Average a list of gradient tensors in place by dividing by
     * {@code worldSize}. Leaves the list unchanged (tensors are modified).
     */
    public static void divideInPlace(List<Tensor> grads, int worldSize) {
        if (worldSize <= 1) return;
        Scalar denom = new Scalar(worldSize);
        for (Tensor g : grads) {
            if (g != null && !g.isNull() && g.defined()) {
                try {
                    g.div_(denom);
                } catch (Throwable ignored) {
                }
            }
        }
    }

    /**
     * Copy {@code src} into {@code dst} element-wise. Both tensors must have
     * the same numel; this is a thin wrapper around {@code dst.copy_(src)}
     * that swallows device / layout mismatches so it can be used as a
     * best-effort checkpoint restore.
     */
    public static void safeCopy(Tensor dst, Tensor src) {
        if (dst == null || dst.isNull() || src == null || src.isNull()) return;
        try {
            dst.copy_(src);
        } catch (Throwable ignored) {
        }
    }

    // ── Bucketing ──────────────────────────────────────────────────────────

    /**
     * Assign parameter indices to buckets by greedy byte-size packing.
     *
     * <p>Returns a list of buckets where each bucket is the list of indices
     * (into {@code params}) that fit under {@code bucketCapBytes}. The
     * resulting list is suitable for c10d's {@code bucket_indices} argument.
     */
    public static List<int[]> buildBucketsBySize(List<Tensor> params, long bucketCapBytes) {
        List<int[]> buckets = new ArrayList<>();
        long firstBucketCap = bucketCapBytes;
        if (params.isEmpty()) return buckets;
        long cap = firstBucketCap;
        List<Integer> current = new ArrayList<>();
        long currentBytes = 0;
        for (int i = 0; i < params.size(); i++) {
            Tensor p = params.get(i);
            if (p == null || p.isNull()) continue;
            long bytes = p.numel() * Math.max(1, p.element_size());
            if (!current.isEmpty() && currentBytes + bytes > cap) {
                buckets.add(current.stream().mapToInt(Integer::intValue).toArray());
                current = new ArrayList<>();
                currentBytes = 0;
                cap = bucketCapBytes;
            }
            current.add(i);
            currentBytes += bytes;
        }
        if (!current.isEmpty()) {
            buckets.add(current.stream().mapToInt(Integer::intValue).toArray());
        }
        return buckets;
    }

    /** Convenience: total bytes of a list of tensors ({@code numel * element_size}). */
    public static long totalBytes(Collection<Tensor> tensors) {
        long total = 0;
        for (Tensor t : tensors) {
            if (t != null && !t.isNull()) {
                total += t.numel() * Math.max(1, t.element_size());
            }
        }
        return total;
    }

    // ── Small helpers ─────────────────────────────────────────────────────

    public static byte defaultDtype() {
        return ScalarType.Float.value;
    }

    public static ScalarType defaultScalarType() {
        return ScalarType.Float;
    }
}
