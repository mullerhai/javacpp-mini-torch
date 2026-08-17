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
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.amp;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.global.torch.DeviceType;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Automatic Mixed Precision (autocast) context for PyTorch.
 *
 * <p>Automatically casts operations to appropriate precision based on operation type:
 * <ul>
 *   <li>Linear, Conv: FP16/BF16/FP8</li>
 *   <li>LayerNorm, Softmax: FP32</li>
 *   <li>Loss computation: matches forward precision</li>
 * </ul>
 *
 * <p>Reference: PyTorch autocast and NVIDIA AMP
 *
 * <pre>{@code
 * try (AutocastContext ctx = new AutocastContext(device, AmpPrecision.FP16, AmpPrecision.FP16)) {
 *     // All operations within this context use FP16
 *     Tensor out = model.forward(input);
 * }
 * }</pre>
 */
public class AutocastContext implements AutoCloseable {

    // Thread-local stack for nested autocast - must be declared before DISABLED
    private static final ThreadLocal<List<AutocastContext>> STACK = ThreadLocal.withInitial(ArrayList::new);

    /** Disabled context singleton. */
    public static final AutocastContext DISABLED = new AutocastContext(null, null, null, null, true);

    private final AmpManager manager;
    private final Device device;
    private final AmpPrecision forwardPrecision;
    private final AmpPrecision backwardPrecision;
    private final boolean disabled;
    private final List<Tensor> cachedTensors;

    public AutocastContext(AmpManager manager, Device device,
                         AmpPrecision forwardPrecision, AmpPrecision backwardPrecision) {
        this(manager, device, forwardPrecision, backwardPrecision, false);
    }

    private AutocastContext(AmpManager manager, Device device,
                          AmpPrecision forwardPrecision, AmpPrecision backwardPrecision,
                          boolean disabled) {
        this.manager = manager;
        this.device = device;
        this.forwardPrecision = forwardPrecision;
        this.backwardPrecision = backwardPrecision;
        this.disabled = disabled;
        this.cachedTensors = new ArrayList<>();

        // Push to stack
        if (!disabled) {
            STACK.get().add(this);
        }
    }

    /**
     * Create an autocast context for a specific device.
     */
    public static AutocastContext create(Device device, AmpPrecision precision) {
        return new AutocastContext(null, device, precision, precision, false);
    }

    /**
     * Create an autocast context for CUDA with FP16.
     */
    public static AutocastContext cudaFp16() {
        return new AutocastContext(null, new Device(DeviceType.CUDA, (byte) 0),
                AmpPrecision.FP16, AmpPrecision.FP16, false);
    }

    /**
     * Create an autocast context for CUDA with BF16.
     */
    public static AutocastContext cudaBf16() {
        return new AutocastContext(null, new Device(DeviceType.CUDA, (byte) 0),
                AmpPrecision.BF16, AmpPrecision.BF16, false);
    }

    /**
     * Create an autocast context for CUDA with FP8.
     */
    public static AutocastContext cudaFp8() {
        return new AutocastContext(null, new Device(DeviceType.CUDA, (byte) 0),
                AmpPrecision.FP8_E4M3, AmpPrecision.FP8_E5M2, false);
    }

    /**
     * Check if autocast is enabled for the current thread.
     */
    public static boolean isEnabled() {
        return !STACK.get().isEmpty();
    }

    /**
     * Get the current autocast context.
     */
    public static AutocastContext getCurrent() {
        List<AutocastContext> stack = STACK.get();
        return stack.isEmpty() ? DISABLED : stack.get(stack.size() - 1);
    }

    /**
     * Cast tensor to forward precision.
     */
    public Tensor cast(Tensor input) {
        if (disabled || input == null || !input.defined()) {
            return input;
        }
        ScalarType target = toScalarType(forwardPrecision);
        if (input.scalar_type() == target) {
            return input;
        }
        return input.to(target);
    }

    /**
     * Cast tensor to backward precision.
     */
    public Tensor castBackward(Tensor input) {
        if (disabled || input == null || !input.defined()) {
            return input;
        }
        ScalarType target = toScalarType(backwardPrecision);
        if (input.scalar_type() == target) {
            return input;
        }
        return input.to(target);
    }

    /**
     * Cast tensor to FP32.
     */
    public Tensor castFp32(Tensor input) {
        if (disabled || input == null || !input.defined()) {
            return input;
        }
        return input.to(ScalarType.Float);
    }

    /**
     * Cast tensor to FP16.
     */
    public Tensor castFp16(Tensor input) {
        if (disabled || input == null || !input.defined()) {
            return input;
        }
        return input.to(ScalarType.Half);
    }

    /**
     * Cast tensor to BF16.
     */
    public Tensor castBf16(Tensor input) {
        if (disabled || input == null || !input.defined()) {
            return input;
        }
        return input.to(ScalarType.BFloat16);
    }

    /**
     * Get the target scalar type for a precision.
     */
    public static ScalarType toScalarType(AmpPrecision precision) {
        if (precision == null) return ScalarType.Float;
        switch (precision) {
            case FP16: return ScalarType.Half;
            case BF16: return ScalarType.BFloat16;
            case FP32: return ScalarType.Float;
            case FP64: return ScalarType.Double;
            case FP8_E4M3: return ScalarType.Float8_e4m3fn;
            case FP8_E5M2: return ScalarType.Float8_e5m2;
            case INT8: return ScalarType.QInt8;
            case INT4: return ScalarType.Char;
            default: return ScalarType.Float;
        }
    }

    /**
     * Check if operation should use FP32.
     */
    public static boolean isFp32Operation(String opName) {
        if (opName == null) return false;
        String lower = opName.toLowerCase();
        return lower.contains("layer_norm") ||
               lower.contains("layernorm") ||
               lower.contains("softmax") ||
               lower.contains("loss") ||
               lower.contains("cross_entropy") ||
               lower.contains("nll_loss") ||
               lower.contains("batch_norm") ||
               lower.contains("layernorm") ||
               lower.contains("rms_norm") ||
               lower.contains("group_norm") ||
               lower.contains("instance_norm");
    }

    /**
     * Check if operation should use lower precision.
     */
    public static boolean isReducedPrecisionOperation(String opName) {
        if (opName == null) return false;
        String lower = opName.toLowerCase();
        return lower.contains("linear") ||
               lower.contains("conv") ||
               lower.contains("matmul") ||
               lower.contains("bmm") ||
               lower.contains("addmm") ||
               lower.contains("einsum");
    }

    /**
     * Get device.
     */
    public Device device() {
        return device;
    }

    /**
     * Get forward precision.
     */
    public AmpPrecision forwardPrecision() {
        return forwardPrecision;
    }

    /**
     * Get backward precision.
     */
    public AmpPrecision backwardPrecision() {
        return backwardPrecision;
    }

    /**
     * Check if autocast is enabled.
     */
//    public boolean isEnabled() {
//        return !disabled;
//    }

    @Override
    public void close() {
        // Pop from stack
        if (!disabled) {
            List<AutocastContext> stack = STACK.get();
            stack.remove(this);
        }

        // Close cached tensors
        for (Tensor t : cachedTensors) {
            try { t.close(); } catch (Exception ignored) {}
        }
        cachedTensors.clear();
    }

    /**
     * Cache a tensor for later cleanup.
     */
    public void cache(Tensor tensor) {
        if (tensor != null && tensor.defined()) {
            cachedTensors.add(tensor);
        }
    }
}
