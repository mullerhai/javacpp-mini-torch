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
package org.bytedeco.pytorch.llm.transformers.utils;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.InferenceMode;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import static org.bytedeco.pytorch.global.torch.ScalarType;

/**
 * Tensor / shape / dtype helpers that wrap javacpp tensors with NumPy / PyTorch
 * ergonomics. Mirrors the helpers in {@code transformers.utils.Tensor} and
 * {@code transformers.testing_utils._tf_gpu_memory_limit}.
 *
 * <p>All methods are stateless and side-effect free unless explicitly noted.
 * When a method accepts a {@code long[]} shape, it must be non-null and non-empty.
 */
public final class T {

    private T() {}

    // ---------- creation ----------------------------------------------------

    public static Tensor zeros(long[] shape) {
        return torch.zeros(shape);
    }

    public static Tensor zeros(long[] shape, ScalarType dtype) {
        return torch.zeros(shape, new TensorOptions(dtype));
    }

    public static Tensor ones(long[] shape) {
        return torch.ones(shape);
    }

    public static Tensor ones(long[] shape, ScalarType dtype) {
        return torch.ones(shape, new TensorOptions(dtype));
    }

    public static Tensor full(long[] shape, Scalar value) {
        return torch.full(shape, value);
    }

    public static Tensor arange(long end) {
        return torch.arange(new Scalar(end));
    }

    public static Tensor arange(long start, long end) {
        return torch.arange(new Scalar(start), new Scalar(end));
    }

    public static Tensor arange(long start, long end, long step) {
        return torch.arange(new Scalar(start), new Scalar(end), new Scalar(step));
    }

    public static Tensor eye(long n) {
        return torch.eye(n);
    }

    public static Tensor eye(long n, long m) {
        return torch.eye(n, m);
    }

    /** NumPy-style empty (uninitialised). */
    public static Tensor empty(long[] shape) {
        return torch.empty(shape);
    }

    public static Tensor empty(long[] shape, ScalarType dtype) {
        // torch.empty(long[], TensorOptions, MemoryFormatOptional) requires the
        // MemoryFormatOptional even if we don't want to override it.
        return torch.empty(shape, new TensorOptions(dtype),
                new org.bytedeco.pytorch.MemoryFormatOptional());
    }

    public static Tensor rand(long[] shape) {
        return torch.rand(shape);
    }

    public static Tensor randn(long[] shape) {
        return torch.randn(shape);
    }

    public static Tensor tensor(float[] data) {
        return torch.tensor(data);
    }

    public static Tensor tensor(int[] data) {
        return torch.tensor(data);
    }

    public static Tensor tensor(long[] data) {
        return torch.tensor(data);
    }

    public static Tensor tensor(float[] data, long[] shape) {
        return torch.tensor(data).reshape(shape);
    }

    /** Construct a tensor backed by the given Java NIO buffers.
     *  NOTE: requires org.bytedeco.numpy on the classpath; if it isn't,
     *  this method throws. Use {@link #tensor(float[])} or a similar overload
     *  to avoid the numpy dependency. */
    public static Tensor tensor(FloatBuffer buf) {
        throw new UnsupportedOperationException(
                "tensor(FloatBuffer) requires org.bytedeco.numpy on the classpath; use tensor(float[]) instead");
    }

    public static Tensor tensor(LongBuffer buf) {
        throw new UnsupportedOperationException(
                "tensor(LongBuffer) requires org.bytedeco.numpy on the classpath; use tensor(long[]) instead");
    }

    public static Tensor tensor(IntBuffer buf) {
        throw new UnsupportedOperationException(
                "tensor(IntBuffer) requires org.bytedeco.numpy on the classpath; use tensor(int[]) instead");
    }

    // ---------- indexing / shape -------------------------------------------

    public static long numel(Tensor t) {
        return t.numel();
    }

    public static long dim(Tensor t) {
        return t.dim();
    }

    public static long size(Tensor t, long dim) {
        return t.size(dim);
    }

    public static long[] shape(Tensor t) {
        long nd = t.dim();
        long[] s = new long[(int) nd];
        for (int i = 0; i < nd; i++) s[i] = t.size(i);
        return s;
    }

    public static Tensor reshape(Tensor t, long... shape) {
        return t.reshape(shape);
    }

    public static Tensor view(Tensor t, long... shape) {
        return t.view(shape);
    }

    public static Tensor transpose(Tensor t, long dim0, long dim1) {
        return t.transpose(dim0, dim1);
    }

    public static Tensor permute(Tensor t, long... dims) {
        return t.permute(dims);
    }

    public static Tensor squeeze(Tensor t) {
        return t.squeeze();
    }

    public static Tensor squeeze(Tensor t, long dim) {
        return t.squeeze(dim);
    }

    public static Tensor unsqueeze(Tensor t, long dim) {
        return t.unsqueeze(dim);
    }

    public static Tensor expand(Tensor t, long... shape) {
        return t.expand(shape);
    }

    public static Tensor contiguous(Tensor t) {
        return t.contiguous();
    }

    public static Tensor detach(Tensor t) {
        return t.detach();
    }

    public static Tensor clone(Tensor t) {
        return t.clone();
    }

    public static Tensor to(Tensor t, ScalarType dtype) {
        return t.to(dtype);
    }

    public static Tensor cpu(Tensor t) {
        return t.cpu();
    }

    public static Tensor cuda(Tensor t) {
        return t.cuda();
    }

    public static Tensor to(Tensor t, org.bytedeco.pytorch.Device device) {
        // Forward to Tensor.to(Device, TypeMeta) with null leaving the
        // original tensor dtype.
        return t.to(device, (org.bytedeco.pytorch.TypeMeta) null);
    }

    // ---------- dtype cast (long/float/byte) -------------------------------

    public static Tensor toLong(Tensor t) {
        return t.to(ScalarType.Long);
    }

    public static Tensor toFloat(Tensor t) {
        return t.to(ScalarType.Float);
    }

    public static Tensor toInt(Tensor t) {
        return t.to(ScalarType.Int);
    }

    public static Tensor toBool(Tensor t) {
        return t.to(ScalarType.Bool);
    }

    // ---------- reductions --------------------------------------------------

    public static Tensor mean(Tensor t) {
        return t.mean();
    }

    public static Tensor mean(Tensor t, long... dims) {
        return t.mean(dims);
    }

    public static Tensor sum(Tensor t) {
        return t.sum();
    }

    public static Tensor sum(Tensor t, long... dims) {
        return t.sum(dims);
    }

    public static Tensor max(Tensor t) {
        return t.max();
    }

    /** Returns the value-only (without indices) tensor from torch.max along a dim.
     *  Use {@link #maxWithIndices(Tensor, long)} to also get the indices. */
    public static Tensor max(Tensor t, long dim) {
        return torch.max(t, dim).get0();
    }

    public static Tensor min(Tensor t) {
        return t.min();
    }

    public static Tensor argmax(Tensor t) {
        return t.argmax();
    }

    public static Tensor argmax(Tensor t, long dim) {
        return t.argmax(new LongOptional(dim), false);
    }

    public static Tensor argmin(Tensor t, long dim) {
        return t.argmin(new LongOptional(dim), false);
    }

    // ---------- random ------------------------------------------------------

    public static long seed(long seed) {
        torch.manual_seed(seed);
        return seed;
    }

    public static long manualSeed(long seed) {
        torch.manual_seed(seed);
        return seed;
    }

    public static long currentSeed() {
        return torch.getDefaultCPUGenerator().current_seed();
    }

    public static boolean cudaIsAvailable() {
        return torch.cuda_is_available();
    }

    public static int cudaDeviceCount() {
        return (int) torch.cuda_device_count();
    }

    // ---------- join --------------------------------------------------------

    /** Stack a list of tensors along a new dim. */
    public static Tensor stack(Tensor... tensors) {
        TensorVector v = new TensorVector();
        for (Tensor t : tensors) v.push_back(t);
        return torch.stack(v, /*dim=*/0);
    }

    public static Tensor stack(long dim, Tensor... tensors) {
        TensorVector v = new TensorVector();
        for (Tensor t : tensors) v.push_back(t);
        return torch.stack(v, dim);
    }

    /** Concatenate a list of tensors along an existing dim. */
    public static Tensor cat(long dim, Tensor... tensors) {
        TensorVector v = new TensorVector();
        for (Tensor t : tensors) v.push_back(t);
        return torch.cat(v, dim);
    }

    public static Tensor cat(Tensor[] list, long dim) {
        TensorVector v = new TensorVector();
        for (Tensor t : list) v.push_back(t);
        return torch.cat(v, dim);
    }

    // ---------- fill --------------------------------------------------------

    public static Tensor fill_(Tensor t, Scalar value) {
        return t.fill_(value);
    }

    public static Tensor fill_(Tensor t, double value) {
        return t.fill_(new Scalar(value));
    }

    public static Tensor fill_(Tensor t, long value) {
        return t.fill_(new Scalar(value));
    }

    public static Tensor zero_(Tensor t) {
        return t.zero_();
    }

    // ---------- indexing helpers -------------------------------------------

    public static Tensor indexSelect(Tensor t, long dim, Tensor indices) {
        return t.index_select(dim, indices);
    }

    public static Tensor gather(Tensor t, long dim, Tensor index) {
        return t.gather(dim, index);
    }

    public static Tensor maskedSelect(Tensor t, Tensor mask) {
        return t.masked_select(mask);
    }

    public static Tensor maskedFill(Tensor t, Tensor mask, Scalar value) {
        return t.masked_fill(mask, value);
    }

    public static Tensor maskedFill_(Tensor t, Tensor mask, double value) {
        return t.masked_fill_(mask, new Scalar(value));
    }

    public static Tensor select(Tensor t, long dim, long index) {
        return t.select(dim, index);
    }

    public static Tensor slice(Tensor t, long dim, long start, long end, long step) {
        return t.slice(dim, new LongOptional(start), new LongOptional(end), step);
    }

    public static Tensor slice(Tensor t, long dim, long start, long end) {
        return t.slice(dim, new LongOptional(start), new LongOptional(end), /*step=*/1);
    }

    // ---------- scalar to tensor ---------------------------------------------

    /** Return a Scalar equal to the single value held by the tensor. */
    public static Scalar toScalar(Tensor t) {
        return new Scalar(t.item_double());
    }

    public static double toDouble(Tensor t) {
        return t.item_double();
    }

    public static float itemFloat(Tensor t) {
        return t.item_float();
    }

    public static long itemLong(Tensor t) {
        return t.item_long();
    }

    public static boolean itemBool(Tensor t) {
        return t.item_bool();
    }

    /** Copy the tensor contents into a freshly allocated int[].
     *  Requires the tensor to be CPU-resident. */
    public static int[] toIntArray(Tensor t) {
        IntPointer p = t.data_ptr_int();
        long n = t.numel();
        int[] out = new int[(int) n];
        p.get(out);
        return out;
    }

    /** Copy the tensor contents into a freshly allocated long[]. */
    public static long[] toLongArray(Tensor t) {
        LongPointer p = t.data_ptr_long();
        long n = t.numel();
        long[] out = new long[(int) n];
        p.get(out);
        return out;
    }

    /** Copy the tensor contents into a freshly allocated float[]. */
    public static float[] toFloatArray(Tensor t) {
        FloatPointer p = t.data_ptr_float();
        long n = t.numel();
        float[] out = new float[(int) n];
        p.get(out);
        return out;
    }

    // ---------- arithmetic --------------------------------------------------

    public static Tensor add(Tensor a, Tensor b) { return a.add(b); }
    public static Tensor sub(Tensor a, Tensor b) { return a.sub(b); }
    public static Tensor mul(Tensor a, Tensor b) { return a.mul(b); }
    public static Tensor div(Tensor a, Tensor b) { return a.div(b); }

    public static Tensor add(Tensor a, double v) { return a.add(new Scalar(v)); }
    public static Tensor mul(Tensor a, double v) { return a.mul(new Scalar(v)); }
    public static Tensor sub(Tensor a, double v) { return a.sub(new Scalar(v)); }
    public static Tensor div(Tensor a, double v) { return a.div(new Scalar(v)); }

    // ---------- matmul -----------------------------------------------------

    public static Tensor matmul(Tensor a, Tensor b) {
        return torch.matmul(a, b);
    }

    public static Tensor mm(Tensor a, Tensor b) {
        return torch.mm(a, b);
    }

    public static Tensor bmm(Tensor a, Tensor b) {
        return torch.bmm(a, b);
    }

    // ---------- softmax / log_softmax ---------------------------------------

    public static Tensor softmax(Tensor t, long dim) {
        return torch.softmax(t, dim);
    }

    public static Tensor logSoftmax(Tensor t, long dim) {
        return torch.log_softmax(t, dim);
    }

    // ---------- utilities ---------------------------------------------------

    /** Returns the top-k values along the given dim. Use {@link #topKWithIndices} for indices too. */
    public static Tensor topK(Tensor t, long k, long dim) {
        return torch.topk(t, k, dim, true, false).get0();
    }

    public static Tensor[] split(Tensor t, long splitSize, long dim) {
        org.bytedeco.pytorch.TensorVector v = torch.split(t, splitSize, dim);
        Tensor[] out = new Tensor[(int) v.size()];
        for (long i = 0; i < v.size(); i++) out[(int) i] = v.get(i);
        return out;
    }

    public static Tensor[] chunk(Tensor t, long chunks, long dim) {
        org.bytedeco.pytorch.TensorVector v = torch.chunk(t, chunks, dim);
        Tensor[] out = new Tensor[(int) v.size()];
        for (long i = 0; i < v.size(); i++) out[(int) i] = v.get(i);
        return out;
    }

    public static Tensor neg(Tensor t) { return t.neg(); }
    public static Tensor abs(Tensor t) { return t.abs(); }
    public static Tensor exp(Tensor t) { return t.exp(); }
    public static Tensor log(Tensor t) { return t.log(); }
    public static Tensor sqrt(Tensor t) { return t.sqrt(); }
    public static Tensor sin(Tensor t) { return t.sin(); }
    public static Tensor cos(Tensor t) { return t.cos(); }
    public static Tensor tanh(Tensor t) { return t.tanh(); }
    public static Tensor relu(Tensor t) { return torch.relu(t); }

    // ---------- comparison --------------------------------------------------

    public static Tensor eq(Tensor a, Tensor b) { return a.eq(b); }
    public static Tensor ne(Tensor a, Tensor b) { return a.ne(b); }
    public static Tensor lt(Tensor a, Tensor b) { return a.lt(b); }
    public static Tensor le(Tensor a, Tensor b) { return a.le(b); }
    public static Tensor gt(Tensor a, Tensor b) { return a.gt(b); }
    public static Tensor ge(Tensor a, Tensor b) { return a.ge(b); }

    public static Tensor eq(Tensor a, double v) { return a.eq(new Scalar(v)); }
    public static Tensor lt(Tensor a, double v) { return a.lt(new Scalar(v)); }
    public static Tensor gt(Tensor a, double v) { return a.gt(new Scalar(v)); }

    // ---------- no-grad helpers --------------------------------------------

    public static <R> R noGrad(java.util.concurrent.Callable<R> body) {
        try (NoGradGuard guard = new NoGradGuard()) {
            return body.call();
        } catch (RuntimeException re) { throw re; }
        catch (Exception e) { throw new RuntimeException(e); }
    }

    public static <R> R inferenceMode(java.util.concurrent.Callable<R> body) {
        try (InferenceMode guard = new InferenceMode()) {
            return body.call();
        } catch (RuntimeException re) { throw re; }
        catch (Exception e) { throw new RuntimeException(e); }
    }
}