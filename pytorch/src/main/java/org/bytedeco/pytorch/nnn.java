/*
 * JavaPP-mini: PyTorch-style nn module factory API
 *
 * This class provides a Python-like API for creating and registering nn modules.
 * Inspired by Python PyTorch's `torch.nn.Linear(...)` syntax.
 *
 * Example usage:
 *
 *   // Traditional way:
 *   public class MyModule extends Module {
 *       private LinearImpl fc1;
 *       private ReLUImpl relu;
 *
 *       public MyModule() {
 *           super("MyModule");
 *           fc1 = register_module("fc1", new LinearImpl(512, 256));
 *           relu = register_module("relu", new ReLUImpl());
 *       }
 *   }
 *
 *   // New way with nn:
 *   public class MyModule extends Module {
 *       public MyModule() {
 *           super("MyModule");
 *           LinearImpl fc1 = nn.linear("fc1", 512, 256);
 *           ReLUImpl relu = nn.relu("relu");
 *       }
 *   }
 *
 *   // Using withParent() for nested modules:
 *   public class NestedModule extends Module {
 *       public NestedModule() {
 *           super("NestedModule");
 *           nn.withParent(this, () -> {
 *               nn.linear("fc1", 512, 256);
 *               nn.relu("relu");
 *           });
 *       }
 *   }
 */
package org.bytedeco.pytorch;
import org.bytedeco.pytorch.data.*;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.enumtype.*;

import org.bytedeco.javacpp.BooleanPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.DoubleOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.enumtype.Conv2dPadding;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.Module;

/**
 * PyTorch-style nn module factory with fluent API.
 *
 * <p>Provides a Python-like way to create and register nn modules:
 * <ul>
 *   <li>Named registration: {@code nn.linear("name", in_features, out_features)}</li>
 *   <li>Unnamed creation: {@code nn.linear(in_features, out_features)}</li>
 *   <li>Sequential chaining: {@code nn.linear(...).relu(...).linear(...)}</li>
 * </ul>
 *
 * <p>Uses ThreadLocal to track the current parent module context, so you don't
 * need to explicitly pass the parent module reference.
 *
 * <p>Supported modules: Linear, Conv1d/2d/3d, ReLU, Dropout, BatchNorm, etc.
 */
public class nnn {

    private nnn() {} // Static-only utility class

    /** ThreadLocal context for parent module tracking. */
    private static final ThreadLocal<Module> PARENT_CONTEXT = new ThreadLocal<>();

    /**
     * Get the current parent module context.
     * @return the parent Module, or null if not set
     */
    public static Module getParent() {
        return PARENT_CONTEXT.get();
    }

    /**
     * Set the current parent module context.
     * @param parent the parent Module
     * @return the previous parent (for restoration)
     */
    static Module setParent(Module parent) {
        Module prev = PARENT_CONTEXT.get();
        PARENT_CONTEXT.set(parent);
        return prev;
    }

    /**
     * Clear the parent context.
     */
    static void clearParent() {
        PARENT_CONTEXT.remove();
    }

    /**
     * Execute a runnable within a given parent module context.
     * Automatically manages ThreadLocal state and exception safety.
     *
     * @param parent the parent Module to set as context
     * @param run    the code to execute
     * @return the parent module (for chaining)
     */
    public static Module withParent(Module parent, Runnable run) {
        Module prev = setParent(parent);
        try {
            run.run();
        } finally {
            setParent(prev);
        }
        return parent;
    }

    // ========================================================================
    // REGISTRATION HELPER
    // ========================================================================

    private static <M extends Module> M register(String name, M module) {
        Module parent = getParent();
        if (parent != null) {
            parent.register_module(name, module);
        }
        return module;
    }

    private static <M extends Module> M create(M module) {
        return module;
    }

    // ========================================================================
    // LINEAR
    // ========================================================================

    // Named registration methods
    public static LinearImpl linear(String name, long inFeatures, long outFeatures) {
        return register(name, new LinearImpl(inFeatures, outFeatures));
    }

    public static LinearImpl linear(String name, long inFeatures, long outFeatures, boolean bias) {
        return register(name, new LinearImpl(new LinearOptions(inFeatures, outFeatures).bias(bias)));
    }

    public static LinearImpl linear(String name, LinearOptions options) {
        return register(name, new LinearImpl(options));
    }

    // Unnamed creation methods (for Sequential)
    public static LinearImpl linear(long inFeatures, long outFeatures) {
        return create(new LinearImpl(inFeatures, outFeatures));
    }

    public static LinearImpl linear(long inFeatures, long outFeatures, boolean bias) {
        return create(new LinearImpl(new LinearOptions(inFeatures, outFeatures).bias(bias)));
    }

    public static LinearImpl linear(LinearOptions options) {
        return create(new LinearImpl(options));
    }

    // ========================================================================
    // BILINEAR
    // ========================================================================

    public static BilinearImpl bilinear(String name, long in1Features, long in2Features, long outFeatures) {
        return register(name, new BilinearImpl(in1Features, in2Features, outFeatures));
    }

    public static BilinearImpl bilinear(String name, long in1Features, long in2Features, long outFeatures, boolean bias) {
        return register(name, new BilinearImpl(new BilinearOptions(in1Features, in2Features, outFeatures).bias(bias)));
    }

    public static BilinearImpl bilinear(String name, BilinearOptions options) {
        return register(name, new BilinearImpl(options));
    }

    public static BilinearImpl bilinear(long in1Features, long in2Features, long outFeatures) {
        return create(new BilinearImpl(in1Features, in2Features, outFeatures));
    }

    // ========================================================================
    // CONVOLUTION 1D
    // ========================================================================

    public static Conv1dImpl conv1d(String name, long inChannels, long outChannels, long kernelSize) {
        return register(name, new Conv1dImpl(inChannels, outChannels, new LongPointer(kernelSize)));
    }

    public static Conv1dImpl conv1d(String name, long inChannels, long outChannels, long kernelSize, long stride) {
        return register(name, new Conv1dImpl(new Conv1dOptions(inChannels, outChannels, new LongPointer(kernelSize)).stride(new LongPointer(stride))));
    }

    public static Conv1dImpl conv1d(String name, Conv1dOptions options) {
        return register(name, new Conv1dImpl(options));
    }

    public static Conv1dImpl conv1d(long inChannels, long outChannels, long kernelSize) {
        return create(new Conv1dImpl(inChannels, outChannels, new LongPointer(kernelSize)));
    }

    // ========================================================================
    // CONVOLUTION 2D
    // ========================================================================

    public static Conv2dImpl conv2d(String name, long inChannels, long outChannels, long kernelSize) {
        return register(name, new Conv2dImpl(inChannels, outChannels, new LongPointer(kernelSize)));
    }

    public static Conv2dImpl conv2d(String name, long inChannels, long outChannels, long kernelSize, long stride) {
        return register(name, new Conv2dImpl(new Conv2dOptions(inChannels, outChannels, new LongPointer(kernelSize)).stride(new LongPointer(stride))));
    }

    public static Conv2dImpl conv2d(String name, long inChannels, long outChannels, long kernelSize, long stride, long padding) {
        return register(name, new Conv2dImpl(new Conv2dOptions(inChannels, outChannels, new LongPointer(kernelSize)).stride(new LongPointer(stride)).padding(new Conv2dPadding(new LongPointer(padding)))));
    }

    public static Conv2dImpl conv2d(String name, Conv2dOptions options) {
        return register(name, new Conv2dImpl(options));
    }

    public static Conv2dImpl conv2d(long inChannels, long outChannels, long kernelSize) {
        return create(new Conv2dImpl(inChannels, outChannels, new LongPointer(kernelSize)));
    }

    // ========================================================================
    // CONVOLUTION 3D
    // ========================================================================

    public static Conv3dImpl conv3d(String name, long inChannels, long outChannels, long kernelSize) {
        return register(name, new Conv3dImpl(inChannels, outChannels, new LongPointer(kernelSize)));
    }

    public static Conv3dImpl conv3d(String name, long inChannels, long outChannels, long kernelSize, long stride) {
        return register(name, new Conv3dImpl(new Conv3dOptions(inChannels, outChannels, new LongPointer(kernelSize)).stride(new LongPointer(stride))));
    }

    public static Conv3dImpl conv3d(String name, Conv3dOptions options) {
        return register(name, new Conv3dImpl(options));
    }

    public static Conv3dImpl conv3d(long inChannels, long outChannels, long kernelSize) {
        return create(new Conv3dImpl(inChannels, outChannels, new LongPointer(kernelSize)));
    }

    // ========================================================================
    // CONV TRANSPOSE 1D
    // ========================================================================

    public static ConvTranspose1dImpl conv_transpose1d(String name, long inChannels, long outChannels, long kernelSize) {
        return register(name, new ConvTranspose1dImpl(inChannels, outChannels, new LongPointer(kernelSize)));
    }

    public static ConvTranspose1dImpl conv_transpose1d(String name, ConvTranspose1dOptions options) {
        return register(name, new ConvTranspose1dImpl(options));
    }

    public static ConvTranspose1dImpl conv_transpose1d(long inChannels, long outChannels, long kernelSize) {
        return create(new ConvTranspose1dImpl(inChannels, outChannels, new LongPointer(kernelSize)));
    }

    // ========================================================================
    // CONV TRANSPOSE 2D
    // ========================================================================

    public static ConvTranspose2dImpl conv_transpose2d(String name, long inChannels, long outChannels, long kernelSize) {
        return register(name, new ConvTranspose2dImpl(inChannels, outChannels, new LongPointer(kernelSize)));
    }

    public static ConvTranspose2dImpl conv_transpose2d(String name, long inChannels, long outChannels, long kernelSize, long stride) {
        return register(name, new ConvTranspose2dImpl(new ConvTranspose2dOptions(inChannels, outChannels, new LongPointer(kernelSize)).stride(new LongPointer(stride))));
    }

    public static ConvTranspose2dImpl conv_transpose2d(String name, ConvTranspose2dOptions options) {
        return register(name, new ConvTranspose2dImpl(options));
    }

    public static ConvTranspose2dImpl conv_transpose2d(long inChannels, long outChannels, long kernelSize) {
        return create(new ConvTranspose2dImpl(inChannels, outChannels, new LongPointer(kernelSize)));
    }

    public static ConvTranspose2dImpl conv_transpose2d(ConvTranspose2dOptions options) {
        return create(new ConvTranspose2dImpl(options));
    }

    // ========================================================================
    // CONV TRANSPOSE 3D
    // ========================================================================

    public static ConvTranspose3dImpl conv_transpose3d(String name, long inChannels, long outChannels, long kernelSize) {
        return register(name, new ConvTranspose3dImpl(inChannels, outChannels, new LongPointer(kernelSize)));
    }

    public static ConvTranspose3dImpl conv_transpose3d(String name, ConvTranspose3dOptions options) {
        return register(name, new ConvTranspose3dImpl(options));
    }

    public static ConvTranspose3dImpl conv_transpose3d(long inChannels, long outChannels, long kernelSize) {
        return create(new ConvTranspose3dImpl(inChannels, outChannels, new LongPointer(kernelSize)));
    }

    public static ConvTranspose3dImpl conv_transpose3d(ConvTranspose3dOptions options) {
        return create(new ConvTranspose3dImpl(options));
    }

    // ========================================================================
    // POOLING 1D
    // ========================================================================

    public static AvgPool1dImpl avg_pool1d(String name, long kernelSize) {
        return register(name, new AvgPool1dImpl(new LongPointer(kernelSize)));
    }

    public static AvgPool1dImpl avg_pool1d(String name, long kernelSize, long stride) {
        return register(name, new AvgPool1dImpl(new AvgPool1dOptions(new LongPointer(kernelSize)).stride(new LongPointer(stride))));
    }

    public static AvgPool1dImpl avg_pool1d(long kernelSize) {
        return create(new AvgPool1dImpl(new LongPointer(kernelSize)));
    }

    public static AvgPool1dImpl avg_pool1d(String name, AvgPool1dOptions options) {
        return register(name, new AvgPool1dImpl(options));
    }
    public static AvgPool1dImpl avg_pool1d(AvgPool1dOptions options) {
        return create(new AvgPool1dImpl(options));
    }

    public static MaxPool1dImpl max_pool1d(String name, long kernelSize) {
        return register(name, new MaxPool1dImpl(new LongPointer(kernelSize)));
    }

    public static MaxPool1dImpl max_pool1d(String name, long kernelSize, long stride) {
        return register(name, new MaxPool1dImpl(new MaxPool1dOptions(new LongPointer(kernelSize)).stride(new LongPointer(stride))));
    }

    public static MaxPool1dImpl max_pool1d(long kernelSize) {
        return create(new MaxPool1dImpl(new LongPointer(kernelSize)));
    }

    public static MaxPool1dImpl max_pool1d(String name, MaxPool1dOptions options) {
        return register(name, new MaxPool1dImpl(options));
    }
    public static MaxPool1dImpl max_pool1d(MaxPool1dOptions options) {
        return create(new MaxPool1dImpl(options));
    }

    public static MaxUnpool1dImpl max_unpool1d(String name, long kernelSize) {
        return register(name, new MaxUnpool1dImpl(new LongPointer(kernelSize)));
    }

    public static MaxUnpool1dImpl max_unpool1d(long kernelSize) {
        return create(new MaxUnpool1dImpl(new LongPointer(kernelSize)));
    }

    public static MaxUnpool1dImpl max_unpool1d(String name, MaxUnpool1dOptions options) {
        return register(name, new MaxUnpool1dImpl(options));
    }
    public static MaxUnpool1dImpl max_unpool1d(MaxUnpool1dOptions options) {
        return create(new MaxUnpool1dImpl(options));
    }

    public static AdaptiveAvgPool1dImpl adaptive_avg_pool1d(String name, long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return register(name, new AdaptiveAvgPool1dImpl(vec));
    }

    public static AdaptiveAvgPool1dImpl adaptive_avg_pool1d(long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return create(new AdaptiveAvgPool1dImpl(vec));
    }

    public static AdaptiveAvgPool1dImpl adaptive_avg_pool1d(String name, AdaptiveAvgPool1dOptions options) {
        return register(name, new AdaptiveAvgPool1dImpl(options));
    }
    public static AdaptiveAvgPool1dImpl adaptive_avg_pool1d(AdaptiveAvgPool1dOptions options) {
        return create(new AdaptiveAvgPool1dImpl(options));
    }

    public static AdaptiveMaxPool1dImpl adaptive_max_pool1d(String name, long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return register(name, new AdaptiveMaxPool1dImpl(vec));
    }

    public static AdaptiveMaxPool1dImpl adaptive_max_pool1d(long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return create(new AdaptiveMaxPool1dImpl(vec));
    }

    public static AdaptiveMaxPool1dImpl adaptive_max_pool1d(String name, AdaptiveMaxPool1dOptions options) {
        return register(name, new AdaptiveMaxPool1dImpl(options));
    }
    public static AdaptiveMaxPool1dImpl adaptive_max_pool1d(AdaptiveMaxPool1dOptions options) {
        return create(new AdaptiveMaxPool1dImpl(options));
    }

    public static LPPool1dImpl lp_pool1d(String name, long kernelSize, double normType) {
        return register(name, new LPPool1dImpl(normType,new LongPointer(kernelSize)));
    }

    public static LPPool1dImpl lp_pool1d(long kernelSize, double normType) {
        return create(new LPPool1dImpl(normType,new LongPointer(kernelSize)));
    }

    public static LPPool1dImpl lp_pool1d(String name, LPPool1dOptions options) {
        return register(name, new LPPool1dImpl(options));
    }
    public static LPPool1dImpl lp_pool1d(LPPool1dOptions options) {
        return create(new LPPool1dImpl(options));
    }

    // ========================================================================
    // POOLING 2D
    // ========================================================================

    public static AvgPool2dImpl avg_pool2d(String name, long kernelSize) {
        return register(name, new AvgPool2dImpl(new LongPointer(kernelSize)));
    }

    public static AvgPool2dImpl avg_pool2d(String name, long kernelSize, long stride) {
        return register(name, new AvgPool2dImpl(new AvgPool2dOptions(new LongPointer(kernelSize)).stride(new LongPointer(stride))));
    }

    public static AvgPool2dImpl avg_pool2d(long kernelSize) {
        return create(new AvgPool2dImpl(new LongPointer(kernelSize)));
    }

    public static AvgPool2dImpl avg_pool2d(String name, AvgPool2dOptions options) {
        return register(name, new AvgPool2dImpl(options));
    }
    public static AvgPool2dImpl avg_pool2d(AvgPool2dOptions options) {
        return create(new AvgPool2dImpl(options));
    }

    public static MaxPool2dImpl max_pool2d(String name, long kernelSize) {
        return register(name, new MaxPool2dImpl(new LongPointer(kernelSize)));
    }

    public static MaxPool2dImpl max_pool2d(String name, long kernelSize, long stride) {
        return register(name, new MaxPool2dImpl(new MaxPool2dOptions(new LongPointer(kernelSize)).stride(new LongPointer(stride))));
    }

    public static MaxPool2dImpl max_pool2d(long kernelSize) {
        return create(new MaxPool2dImpl(new LongPointer(kernelSize)));
    }

    public static MaxPool2dImpl max_pool2d(String name, MaxPool2dOptions options) {
        return register(name, new MaxPool2dImpl(options));
    }
    public static MaxPool2dImpl max_pool2d(MaxPool2dOptions options) {
        return create(new MaxPool2dImpl(options));
    }

    public static MaxUnpool2dImpl max_unpool2d(String name, long kernelSize) {
        return register(name, new MaxUnpool2dImpl(new LongPointer(kernelSize)));
    }

    public static MaxUnpool2dImpl max_unpool2d(long kernelSize) {
        return create(new MaxUnpool2dImpl(new LongPointer(kernelSize)));
    }

    public static MaxUnpool2dImpl max_unpool2d(String name, MaxUnpool2dOptions options) {
        return register(name, new MaxUnpool2dImpl(options));
    }
    public static MaxUnpool2dImpl max_unpool2d(MaxUnpool2dOptions options) {
        return create(new MaxUnpool2dImpl(options));
    }

    public static AdaptiveAvgPool2dImpl adaptive_avg_pool2d(String name, long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return register(name, new AdaptiveAvgPool2dImpl(vec));
    }

    public static AdaptiveAvgPool2dImpl adaptive_avg_pool2d(long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return create(new AdaptiveAvgPool2dImpl(vec));
    }

    public static AdaptiveAvgPool2dImpl adaptive_avg_pool2d(String name, AdaptiveAvgPool2dOptions options) {
        return register(name, new AdaptiveAvgPool2dImpl(options));
    }
    public static AdaptiveAvgPool2dImpl adaptive_avg_pool2d(AdaptiveAvgPool2dOptions options) {
        return create(new AdaptiveAvgPool2dImpl(options));
    }

    public static AdaptiveMaxPool2dImpl adaptive_max_pool2d(String name, long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return register(name, new AdaptiveMaxPool2dImpl(vec));
    }

    public static AdaptiveMaxPool2dImpl adaptive_max_pool2d(long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return create(new AdaptiveMaxPool2dImpl(vec));
    }

    public static AdaptiveMaxPool2dImpl adaptive_max_pool2d(String name, AdaptiveMaxPool2dOptions options) {
        return register(name, new AdaptiveMaxPool2dImpl(options));
    }
    public static AdaptiveMaxPool2dImpl adaptive_max_pool2d(AdaptiveMaxPool2dOptions options) {
        return create(new AdaptiveMaxPool2dImpl(options));
    }

    public static LPPool2dImpl lp_pool2d(String name, long kernelSize, double normType) {
        return register(name, new LPPool2dImpl(normType, new LongPointer(kernelSize)));
    }

    public static LPPool2dImpl lp_pool2d(long kernelSize, double normType) {
        return create(new LPPool2dImpl(normType, new LongPointer(kernelSize)));
    }

    public static LPPool2dImpl lp_pool2d(String name, LPPool2dOptions options) {
        return register(name, new LPPool2dImpl(options));
    }
    public static LPPool2dImpl lp_pool2d(LPPool2dOptions options) {
        return create(new LPPool2dImpl(options));
    }

    public static FractionalMaxPool2dImpl fractional_max_pool2d(String name, long kernelSize) {
        return register(name, new FractionalMaxPool2dImpl(new LongPointer(kernelSize)));
    }

    public static FractionalMaxPool2dImpl fractional_max_pool2d(long kernelSize) {
        return create(new FractionalMaxPool2dImpl(new LongPointer(kernelSize)));
    }

    public static FractionalMaxPool2dImpl fractional_max_pool2d(String name, FractionalMaxPool2dOptions options) {
        return register(name, new FractionalMaxPool2dImpl(options));
    }
    public static FractionalMaxPool2dImpl fractional_max_pool2d(FractionalMaxPool2dOptions options) {
        return create(new FractionalMaxPool2dImpl(options));
    }

    // ========================================================================
    // POOLING 3D
    // ========================================================================

    public static AvgPool3dImpl avg_pool3d(String name, long kernelSize) {
        return register(name, new AvgPool3dImpl(new LongPointer(kernelSize)));
    }

    public static AvgPool3dImpl avg_pool3d(String name, long kernelSize, long stride) {
        return register(name, new AvgPool3dImpl(new AvgPool3dOptions(new LongPointer(kernelSize)).stride(new LongPointer(stride))));
    }

    public static AvgPool3dImpl avg_pool3d(long kernelSize) {
        return create(new AvgPool3dImpl(new LongPointer(kernelSize)));
    }

    public static AvgPool3dImpl avg_pool3d(String name, AvgPool3dOptions options) {
        return register(name, new AvgPool3dImpl(options));
    }
    public static AvgPool3dImpl avg_pool3d(AvgPool3dOptions options) {
        return create(new AvgPool3dImpl(options));
    }

    public static MaxPool3dImpl max_pool3d(String name, long kernelSize) {
        return register(name, new MaxPool3dImpl(new LongPointer(kernelSize)));
    }

    public static MaxPool3dImpl max_pool3d(String name, long kernelSize, long stride) {
        return register(name, new MaxPool3dImpl(new MaxPool3dOptions(new LongPointer(kernelSize)).stride(new LongPointer(stride))));
    }

    public static MaxPool3dImpl max_pool3d(long kernelSize) {
        return create(new MaxPool3dImpl(new LongPointer(kernelSize)));
    }

    public static MaxPool3dImpl max_pool3d(String name, MaxPool3dOptions options) {
        return register(name, new MaxPool3dImpl(options));
    }
    public static MaxPool3dImpl max_pool3d(MaxPool3dOptions options) {
        return create(new MaxPool3dImpl(options));
    }

    public static MaxUnpool3dImpl max_unpool3d(String name, long kernelSize) {
        return register(name, new MaxUnpool3dImpl(new LongPointer(kernelSize)));
    }

    public static MaxUnpool3dImpl max_unpool3d(long kernelSize) {
        return create(new MaxUnpool3dImpl(new LongPointer(kernelSize)));
    }

    public static MaxUnpool3dImpl max_unpool3d(String name, MaxUnpool3dOptions options) {
        return register(name, new MaxUnpool3dImpl(options));
    }
    public static MaxUnpool3dImpl max_unpool3d(MaxUnpool3dOptions options) {
        return create(new MaxUnpool3dImpl(options));
    }

    public static AdaptiveAvgPool3dImpl adaptive_avg_pool3d(String name, long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return register(name, new AdaptiveAvgPool3dImpl(vec));
    }

    public static AdaptiveAvgPool3dImpl adaptive_avg_pool3d(long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return create(new AdaptiveAvgPool3dImpl(vec));
    }

    public static AdaptiveAvgPool3dImpl adaptive_avg_pool3d(String name, AdaptiveAvgPool3dOptions options) {
        return register(name, new AdaptiveAvgPool3dImpl(options));
    }
    public static AdaptiveAvgPool3dImpl adaptive_avg_pool3d(AdaptiveAvgPool3dOptions options) {
        return create(new AdaptiveAvgPool3dImpl(options));
    }

    public static AdaptiveMaxPool3dImpl adaptive_max_pool3d(String name, long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return register(name, new AdaptiveMaxPool3dImpl(vec));
    }

    public static AdaptiveMaxPool3dImpl adaptive_max_pool3d(long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return create(new AdaptiveMaxPool3dImpl(vec));
    }

    public static AdaptiveMaxPool3dImpl adaptive_max_pool3d(String name, AdaptiveMaxPool3dOptions options) {
        return register(name, new AdaptiveMaxPool3dImpl(options));
    }
    public static AdaptiveMaxPool3dImpl adaptive_max_pool3d(AdaptiveMaxPool3dOptions options) {
        return create(new AdaptiveMaxPool3dImpl(options));
    }

    public static LPPool3dImpl lp_pool3d(String name, long kernelSize, double normType) {
        return register(name, new LPPool3dImpl(normType, new LongPointer(kernelSize)));
    }

    public static LPPool3dImpl lp_pool3d(long kernelSize, double normType) {
        return create(new LPPool3dImpl(normType, new LongPointer(kernelSize)));
    }

    public static LPPool3dImpl lp_pool3d(String name, LPPool3dOptions options) {
        return register(name, new LPPool3dImpl(options));
    }
    public static LPPool3dImpl lp_pool3d(LPPool3dOptions options) {
        return create(new LPPool3dImpl(options));
    }

    public static FractionalMaxPool3dImpl fractional_max_pool3d(String name, long kernelSize) {
        return register(name, new FractionalMaxPool3dImpl(new LongPointer(kernelSize)));
    }

    public static FractionalMaxPool3dImpl fractional_max_pool3d(long kernelSize) {
        return create(new FractionalMaxPool3dImpl(new LongPointer(kernelSize)));
    }

    public static FractionalMaxPool3dImpl fractional_max_pool3d(String name, FractionalMaxPool3dOptions options) {
        return register(name, new FractionalMaxPool3dImpl(options));
    }
    public static FractionalMaxPool3dImpl fractional_max_pool3d(FractionalMaxPool3dOptions options) {
        return create(new FractionalMaxPool3dImpl(options));
    }

    // ========================================================================
    // PADDING
    // ========================================================================

    // ReflectionPad
    public static ReflectionPad1dImpl reflection_pad1d(String name, long padding) {
        return register(name, new ReflectionPad1dImpl(new LongPointer(padding)));
    }

    public static ReflectionPad1dImpl reflection_pad1d(long padding) {
        return create(new ReflectionPad1dImpl(new LongPointer(padding)));
    }

    public static ReflectionPad1dImpl reflection_pad1d(String name, ReflectionPad1dOptions options) {
        return register(name, new ReflectionPad1dImpl(options));
    }
    public static ReflectionPad1dImpl reflection_pad1d(ReflectionPad1dOptions options) {
        return create(new ReflectionPad1dImpl(options));
    }

    public static ReflectionPad2dImpl reflection_pad2d(String name, long padding) {
        return register(name, new ReflectionPad2dImpl(new LongPointer(padding)));
    }

    public static ReflectionPad2dImpl reflection_pad2d(String name, long... padding) {
        LongVector vec = new LongVector(padding);
        return register(name, new ReflectionPad2dImpl(vec));
    }

    public static ReflectionPad2dImpl reflection_pad2d(long padding) {
        return create(new ReflectionPad2dImpl(new LongPointer(padding)));
    }

    public static ReflectionPad2dImpl reflection_pad2d(String name, ReflectionPad2dOptions options) {
        return register(name, new ReflectionPad2dImpl(options));
    }
    public static ReflectionPad2dImpl reflection_pad2d(ReflectionPad2dOptions options) {
        return create(new ReflectionPad2dImpl(options));
    }

    public static ReflectionPad3dImpl reflection_pad3d(String name, long... padding) {
        LongVector vec = new LongVector(padding);
        return register(name, new ReflectionPad3dImpl(vec));
    }

    public static ReflectionPad3dImpl reflection_pad3d(long... padding) {
        LongVector vec = new LongVector(padding);
        return create(new ReflectionPad3dImpl(vec));
    }

    public static ReflectionPad3dImpl reflection_pad3d(String name, ReflectionPad3dOptions options) {
        return register(name, new ReflectionPad3dImpl(options));
    }
    public static ReflectionPad3dImpl reflection_pad3d(ReflectionPad3dOptions options) {
        return create(new ReflectionPad3dImpl(options));
    }

    // ReplicationPad
    public static ReplicationPad1dImpl replication_pad1d(String name, long padding) {
        return register(name, new ReplicationPad1dImpl(new LongPointer(padding)));
    }

    public static ReplicationPad1dImpl replication_pad1d(long padding) {
        return create(new ReplicationPad1dImpl(new LongPointer(padding)));
    }

    public static ReplicationPad1dImpl replication_pad1d(String name, ReplicationPad1dOptions options) {
        return register(name, new ReplicationPad1dImpl(options));
    }
    public static ReplicationPad1dImpl replication_pad1d(ReplicationPad1dOptions options) {
        return create(new ReplicationPad1dImpl(options));
    }

    public static ReplicationPad2dImpl replication_pad2d(String name, long padding) {
        return register(name, new ReplicationPad2dImpl(new LongPointer(padding)));
    }

    public static ReplicationPad2dImpl replication_pad2d(String name, long... padding) {
        LongVector vec = new LongVector(padding);
        return register(name, new ReplicationPad2dImpl(vec));
    }

    public static ReplicationPad2dImpl replication_pad2d(long padding) {
        return create(new ReplicationPad2dImpl(new LongPointer(padding)));
    }

    public static ReplicationPad2dImpl replication_pad2d(String name, ReplicationPad2dOptions options) {
        return register(name, new ReplicationPad2dImpl(options));
    }
    public static ReplicationPad2dImpl replication_pad2d(ReplicationPad2dOptions options) {
        return create(new ReplicationPad2dImpl(options));
    }

    public static ReplicationPad3dImpl replication_pad3d(String name, long... padding) {
        LongVector vec = new LongVector(padding);
        return register(name, new ReplicationPad3dImpl(vec));
    }

    public static ReplicationPad3dImpl replication_pad3d(long... padding) {
        LongVector vec = new LongVector(padding);
        return create(new ReplicationPad3dImpl(vec));
    }

    public static ReplicationPad3dImpl replication_pad3d(String name, ReplicationPad3dOptions options) {
        return register(name, new ReplicationPad3dImpl(options));
    }
    public static ReplicationPad3dImpl replication_pad3d(ReplicationPad3dOptions options) {
        return create(new ReplicationPad3dImpl(options));
    }

    // ConstantPad
    public static ConstantPad1dImpl constant_pad1d(String name, long padding, float value) {
        return register(name, new ConstantPad1dImpl(new LongPointer(padding), value));
    }

    public static ConstantPad1dImpl constant_pad1d(long padding, float value) {
        return create(new ConstantPad1dImpl(new LongPointer(padding), value));
    }

    public static ConstantPad1dImpl constant_pad1d(String name, ConstantPad1dOptions options) {
        return register(name, new ConstantPad1dImpl(options));
    }
    public static ConstantPad1dImpl constant_pad1d(ConstantPad1dOptions options) {
        return create(new ConstantPad1dImpl(options));
    }

    public static ConstantPad2dImpl constant_pad2d(String name, long padding, float value) {
        return register(name, new ConstantPad2dImpl(new LongPointer(padding), value));
    }

    public static ConstantPad2dImpl constant_pad2d(String name, long... padding) {
        LongVector vec = new LongVector(padding);
        return register(name, new ConstantPad2dImpl(new LongPointer(vec), 0)); // value defaults to 0
    }

    public static ConstantPad2dImpl constant_pad2d(long padding, float value) {
        return create(new ConstantPad2dImpl(new LongPointer(padding), value));
    }

    public static ConstantPad2dImpl constant_pad2d(String name, ConstantPad2dOptions options) {
        return register(name, new ConstantPad2dImpl(options));
    }
    public static ConstantPad2dImpl constant_pad2d(ConstantPad2dOptions options) {
        return create(new ConstantPad2dImpl(options));
    }

//    public static ConstantPad3dImpl constant_pad3d(String name, long... padding) {
//        LongVector vec = new LongVector(padding);
//        return register(name, new ConstantPad3dImpl(new LongPointer(vec), 0));
//    }

    public static ConstantPad3dImpl constant_pad3d(String name, long... padding) {
        LongVector vec = new LongVector(padding);
        return register(name, new ConstantPad3dImpl(new LongPointer(vec), 0));
    }

    public static ConstantPad3dImpl constant_pad3d(long padding, float value) {
        return create(new ConstantPad3dImpl(new LongPointer(padding), value));
    }

    public static ConstantPad3dImpl constant_pad3d(String name, ConstantPad3dOptions options) {
        return register(name, new ConstantPad3dImpl(options));
    }
    public static ConstantPad3dImpl constant_pad3d(ConstantPad3dOptions options) {
        return create(new ConstantPad3dImpl(options));
    }

    // ZeroPad
    public static ZeroPad1dImpl zero_pad1d(String name, long padding) {
        return register(name, new ZeroPad1dImpl(new LongPointer(padding)));
    }

    public static ZeroPad1dImpl zero_pad1d(long padding) {
        return create(new ZeroPad1dImpl(new LongPointer(padding)));
    }

    public static ZeroPad1dImpl zero_pad1d(String name, ZeroPad1dOptions options) {
        return register(name, new ZeroPad1dImpl(options));
    }
    public static ZeroPad1dImpl zero_pad1d(ZeroPad1dOptions options) {
        return create(new ZeroPad1dImpl(options));
    }

    public static ZeroPad2dImpl zero_pad2d(String name, long padding) {
        return register(name, new ZeroPad2dImpl(new LongPointer(padding)));
    }

    public static ZeroPad2dImpl zero_pad2d(long padding) {
        return create(new ZeroPad2dImpl(new LongPointer(padding)));
    }

    public static ZeroPad2dImpl zero_pad2d(String name, ZeroPad2dOptions options) {
        return register(name, new ZeroPad2dImpl(options));
    }
    public static ZeroPad2dImpl zero_pad2d(ZeroPad2dOptions options) {
        return create(new ZeroPad2dImpl(options));
    }

    public static ZeroPad3dImpl zero_pad3d(String name, long padding) {
        return register(name, new ZeroPad3dImpl(new LongPointer(padding)));
    }

    public static ZeroPad3dImpl zero_pad3d(long padding) {
        return create(new ZeroPad3dImpl(new LongPointer(padding)));
    }

    public static ZeroPad3dImpl zero_pad3d(String name, ZeroPad3dOptions options) {
        return register(name, new ZeroPad3dImpl(options));
    }
    public static ZeroPad3dImpl zero_pad3d(ZeroPad3dOptions options) {
        return create(new ZeroPad3dImpl(options));
    }

    // ========================================================================
    // NORMALIZATION
    // ========================================================================

    // BatchNorm
    public static BatchNorm1dImpl batch_norm1d(String name, long numFeatures) {
        return register(name, new BatchNorm1dImpl(new LongPointer(numFeatures)));
    }

    public static BatchNorm1dImpl batch_norm1d(String name, long numFeatures, float momentum, float eps) {
        return register(name, new BatchNorm1dImpl(new BatchNormOptions(numFeatures).momentum(new DoubleOptional(momentum)).eps(eps)));
    }

    public static BatchNorm1dImpl batch_norm1d(long numFeatures) {
        return create(new BatchNorm1dImpl(new LongPointer(numFeatures)));
    }

    public static BatchNorm1dImpl batch_norm1d(String name, BatchNormOptions options) {
        return register(name, new BatchNorm1dImpl(options));
    }
    public static BatchNorm1dImpl batch_norm1d(BatchNormOptions options) {
        return create(new BatchNorm1dImpl(options));
    }

    public static BatchNorm2dImpl batch_norm2d(String name, long numFeatures) {
        return register(name, new BatchNorm2dImpl(new LongPointer(numFeatures)));
    }

    public static BatchNorm2dImpl batch_norm2d(String name, long numFeatures, float momentum, float eps) {
        return register(name, new BatchNorm2dImpl(new BatchNormOptions(numFeatures).momentum(new DoubleOptional(momentum)).eps(eps)));
    }

    public static BatchNorm2dImpl batch_norm2d(long numFeatures) {
        return create(new BatchNorm2dImpl(new LongPointer(numFeatures)));
    }

    public static BatchNorm2dImpl batch_norm2d(String name, BatchNormOptions options) {
        return register(name, new BatchNorm2dImpl(options));
    }
    public static BatchNorm2dImpl batch_norm2d(BatchNormOptions options) {
        return create(new BatchNorm2dImpl(options));
    }

    public static BatchNorm3dImpl batch_norm3d(String name, long numFeatures) {
        return register(name, new BatchNorm3dImpl(new LongPointer(numFeatures)));
    }

    public static BatchNorm3dImpl batch_norm3d(String name, long numFeatures, float momentum, float eps) {
        return register(name, new BatchNorm3dImpl(new BatchNormOptions(numFeatures).momentum(new DoubleOptional(momentum)).eps(eps)));
    }

    public static BatchNorm3dImpl batch_norm3d(long numFeatures) {
        return create(new BatchNorm3dImpl(new LongPointer(numFeatures)));
    }

    public static BatchNorm3dImpl batch_norm3d(String name, BatchNormOptions options) {
        return register(name, new BatchNorm3dImpl(options));
    }
    public static BatchNorm3dImpl batch_norm3d(BatchNormOptions options) {
        return create(new BatchNorm3dImpl(options));
    }

    // InstanceNorm
    public static InstanceNorm1dImpl instance_norm1d(String name, long numFeatures) {
        return register(name, new InstanceNorm1dImpl(new LongPointer(numFeatures)));
    }

    public static InstanceNorm1dImpl instance_norm1d(long numFeatures) {
        return create(new InstanceNorm1dImpl(new LongPointer(numFeatures)));
    }

    public static InstanceNorm1dImpl instance_norm1d(String name, InstanceNormOptions options) {
        return register(name, new InstanceNorm1dImpl(options));
    }
    public static InstanceNorm1dImpl instance_norm1d(InstanceNormOptions options) {
        return create(new InstanceNorm1dImpl(options));
    }

    public static InstanceNorm2dImpl instance_norm2d(String name, long numFeatures) {
        return register(name, new InstanceNorm2dImpl(new LongPointer(numFeatures)));
    }

    public static InstanceNorm2dImpl instance_norm2d(long numFeatures) {
        return create(new InstanceNorm2dImpl(new LongPointer(numFeatures)));
    }

    public static InstanceNorm2dImpl instance_norm2d(String name, InstanceNormOptions options) {
        return register(name, new InstanceNorm2dImpl(options));
    }
    public static InstanceNorm2dImpl instance_norm2d(InstanceNormOptions options) {
        return create(new InstanceNorm2dImpl(options));
    }

    public static InstanceNorm3dImpl instance_norm3d(String name, long numFeatures) {
        return register(name, new InstanceNorm3dImpl(new LongPointer(numFeatures)));
    }

    public static InstanceNorm3dImpl instance_norm3d(long numFeatures) {
        return create(new InstanceNorm3dImpl(new LongPointer(numFeatures)));
    }

    public static InstanceNorm3dImpl instance_norm3d(String name, InstanceNormOptions options) {
        return register(name, new InstanceNorm3dImpl(options));
    }
    public static InstanceNorm3dImpl instance_norm3d(InstanceNormOptions options) {
        return create(new InstanceNorm3dImpl(options));
    }

    // LayerNorm
    public static LayerNormImpl layer_norm(String name, long... normalizedShape) {
        LongVector vec = new LongVector(normalizedShape);
        return register(name, new LayerNormImpl(vec));
    }

    public static LayerNormImpl layer_norm(String name, LayerNormOptions options) {
        return register(name, new LayerNormImpl(options));
    }

    public static LayerNormImpl layer_norm(long... normalizedShape) {
        LongVector vec = new LongVector(normalizedShape);
        return create(new LayerNormImpl(vec));
    }

    public static LayerNormImpl layer_norm(LayerNormOptions options) {
        return create(new LayerNormImpl(options));
    }

    // GroupNorm
    public static GroupNormImpl group_norm(String name, long numGroups, long numChannels) {
        return register(name, new GroupNormImpl(numGroups, numChannels));
    }

    public static GroupNormImpl group_norm(String name, long numGroups, long numChannels, float eps) {
        return register(name, new GroupNormImpl(new GroupNormOptions(numGroups, numChannels).eps(eps)));
    }

    public static GroupNormImpl group_norm(long numGroups, long numChannels) {
        return create(new GroupNormImpl(numGroups, numChannels));
    }

    public static GroupNormImpl group_norm(String name, GroupNormOptions options) {
        return register(name, new GroupNormImpl(options));
    }
    public static GroupNormImpl group_norm(GroupNormOptions options) {
        return create(new GroupNormImpl(options));
    }

    // LocalResponseNorm
    public static LocalResponseNormImpl local_response_norm(String name, long size) {
        return register(name, new LocalResponseNormImpl(size));
    }

    public static LocalResponseNormImpl local_response_norm(String name, long size, float alpha, float beta, float k) {
        return register(name, new LocalResponseNormImpl(new LocalResponseNormOptions(size).alpha(alpha).beta(beta).k(k)));
    }

    public static LocalResponseNormImpl local_response_norm(long size) {
        return create(new LocalResponseNormImpl(size));
    }

    public static LocalResponseNormImpl local_response_norm(String name, LocalResponseNormOptions options) {
        return register(name, new LocalResponseNormImpl(options));
    }
    public static LocalResponseNormImpl local_response_norm(LocalResponseNormOptions options) {
        return create(new LocalResponseNormImpl(options));
    }

    // CrossMapLRN2d
    public static CrossMapLRN2dImpl cross_map_lrn2d(String name, long size) {
        return register(name, new CrossMapLRN2dImpl(size));
    }

    public static CrossMapLRN2dImpl cross_map_lrn2d(long size) {
        return create(new CrossMapLRN2dImpl(size));
    }

    public static CrossMapLRN2dImpl cross_map_lrn2d(String name, CrossMapLRN2dOptions options) {
        return register(name, new CrossMapLRN2dImpl(options));
    }
    public static CrossMapLRN2dImpl cross_map_lrn2d(CrossMapLRN2dOptions options) {
        return create(new CrossMapLRN2dImpl(options));
    }

    // ========================================================================
    // ACTIVATIONS
    // ========================================================================

    public static ReLUImpl relu(String name) {
        return register(name, new ReLUImpl());
    }

    public static ReLUImpl relu(String name, boolean inplace) {
        return register(name, new ReLUImpl(new ReLUOptions().inplace(inplace)));
    }

    public static ReLUImpl relu() {
        return create(new ReLUImpl());
    }

    public static ReLUImpl relu(String name, ReLUOptions options) {
        return register(name, new ReLUImpl(options));
    }
    public static ReLUImpl relu(ReLUOptions options) {
        return create(new ReLUImpl(options));
    }

    public static ReLU6Impl relu6(String name) {
        return register(name, new ReLU6Impl());
    }

    public static ReLU6Impl relu6(String name, boolean inplace) {
        return register(name, new ReLU6Impl(new ReLU6Options().inplace(inplace)));
    }

    public static ReLU6Impl relu6() {
        return create(new ReLU6Impl());
    }

    public static ReLU6Impl relu6(String name, ReLU6Options options) {
        return register(name, new ReLU6Impl(options));
    }
    public static ReLU6Impl relu6(ReLU6Options options) {
        return create(new ReLU6Impl(options));
    }

    public static RReLUImpl rrelu(String name) {
        return register(name, new RReLUImpl());
    }

    public static RReLUImpl rrelu(String name, float lower, float upper) {
        return register(name, new RReLUImpl(new RReLUOptions().lower(lower).upper(upper)));
    }

    public static RReLUImpl rrelu() {
        return create(new RReLUImpl());
    }

    public static RReLUImpl rrelu(String name, RReLUOptions options) {
        return register(name, new RReLUImpl(options));
    }
    public static RReLUImpl rrelu(RReLUOptions options) {
        return create(new RReLUImpl(options));
    }

    public static LeakyReLUImpl leaky_relu(String name) {
        return register(name, new LeakyReLUImpl());
    }

    public static LeakyReLUImpl leaky_relu(String name, float negativeSlope) {
        return register(name, new LeakyReLUImpl(new LeakyReLUOptions().negative_slope(negativeSlope)));
    }

    public static LeakyReLUImpl leaky_relu(String name, float negativeSlope, boolean inplace) {
        return register(name, new LeakyReLUImpl(new LeakyReLUOptions().negative_slope(negativeSlope).inplace(inplace)));
    }

    public static LeakyReLUImpl leaky_relu() {
        return create(new LeakyReLUImpl());
    }

    public static LeakyReLUImpl leaky_relu(String name, LeakyReLUOptions options) {
        return register(name, new LeakyReLUImpl(options));
    }
    public static LeakyReLUImpl leaky_relu(LeakyReLUOptions options) {
        return create(new LeakyReLUImpl(options));
    }

    public static PReLUImpl prelu(String name) {
        return register(name, new PReLUImpl());
    }

    public static PReLUImpl prelu(String name, long numParameters, float init) {
        return register(name, new PReLUImpl(new PReLUOptions(numParameters).init(init)));
    }

    public static PReLUImpl prelu() {
        return create(new PReLUImpl());
    }

    public static PReLUImpl prelu(String name, PReLUOptions options) {
        return register(name, new PReLUImpl(options));
    }
    public static PReLUImpl prelu(PReLUOptions options) {
        return create(new PReLUImpl(options));
    }

    public static ELUImpl elu(String name) {
        return register(name, new ELUImpl());
    }

    public static ELUImpl elu(String name, float alpha) {
        return register(name, new ELUImpl(new ELUOptions().alpha(alpha)));
    }

    public static ELUImpl elu(String name, float alpha, boolean inplace) {
        return register(name, new ELUImpl(new ELUOptions().alpha(alpha).inplace(inplace)));
    }

    public static ELUImpl elu() {
        return create(new ELUImpl());
    }

    public static ELUImpl elu(String name, ELUOptions options) {
        return register(name, new ELUImpl(options));
    }
    public static ELUImpl elu(ELUOptions options) {
        return create(new ELUImpl(options));
    }

    public static CELUImpl celu(String name) {
        return register(name, new CELUImpl());
    }

    public static CELUImpl celu(String name, float alpha) {
        return register(name, new CELUImpl(new CELUOptions().alpha(alpha)));
    }

    public static CELUImpl celu() {
        return create(new CELUImpl());
    }

    public static CELUImpl celu(String name, CELUOptions options) {
        return register(name, new CELUImpl(options));
    }
    public static CELUImpl celu(CELUOptions options) {
        return create(new CELUImpl(options));
    }

    public static SELUImpl selu(String name) {
        return register(name, new SELUImpl());
    }

    public static SELUImpl selu(String name, boolean inplace) {
        return register(name, new SELUImpl(new SELUOptions().inplace(inplace)));
    }

    public static SELUImpl selu() {
        return create(new SELUImpl());
    }

    public static SELUImpl selu(String name, SELUOptions options) {
        return register(name, new SELUImpl(options));
    }
    public static SELUImpl selu(SELUOptions options) {
        return create(new SELUImpl(options));
    }

    public static GLUImpl glu(String name) {
        return register(name, new GLUImpl());
    }

    public static GLUImpl glu(String name, long dim) {
        return register(name, new GLUImpl(new GLUOptions(dim)));
    }

    public static GLUImpl glu() {
        return create(new GLUImpl());
    }

    public static GLUImpl glu(String name, GLUOptions options) {
        return register(name, new GLUImpl(options));
    }
    public static GLUImpl glu(GLUOptions options) {
        return create(new GLUImpl(options));
    }

    public static GELUImpl gelu(String name) {
        return register(name, new GELUImpl());
    }

    public static GELUImpl gelu(String name, String approximate) {
        return register(name, new GELUImpl(new GELUOptions().approximate(approximate)));
    }

    public static GELUImpl gelu() {
        return create(new GELUImpl());
    }

    public static GELUImpl gelu(String name, GELUOptions options) {
        return register(name, new GELUImpl(options));
    }
    public static GELUImpl gelu(GELUOptions options) {
        return create(new GELUImpl(options));
    }

    public static SiLUImpl silu(String name) {
        return register(name, new SiLUImpl());
    }

    public static SiLUImpl silu(String name, boolean inplace) {
        return register(name, new SiLUImpl(new BooleanPointer(inplace)));
    }

    public static SiLUImpl silu() {
        return create(new SiLUImpl());
    }

    public static MishImpl mish(String name) {
        return register(name, new MishImpl());
    }

    public static MishImpl mish(String name, boolean inplace) {
        return register(name, new MishImpl(new BooleanPointer(inplace)));
    }

    public static MishImpl mish() {
        return create(new MishImpl());
    }

    public static TanhImpl tanh(String name) {
        return register(name, new TanhImpl());
    }

    public static TanhImpl tanh() {
        return create(new TanhImpl());
    }

    public static HardtanhImpl hardtanh(String name) {
        return register(name, new HardtanhImpl());
    }

    public static HardtanhImpl hardtanh(String name, float minVal, float maxVal) {
        return register(name, new HardtanhImpl(new HardtanhOptions().min_val(minVal).max_val(maxVal)));
    }

    public static HardtanhImpl hardtanh() {
        return create(new HardtanhImpl());
    }

    public static HardtanhImpl hardtanh(String name, HardtanhOptions options) {
        return register(name, new HardtanhImpl(options));
    }
    public static HardtanhImpl hardtanh(HardtanhOptions options) {
        return create(new HardtanhImpl(options));
    }

    public static SigmoidImpl sigmoid(String name) {
        return register(name, new SigmoidImpl());
    }

    public static SigmoidImpl sigmoid() {
        return create(new SigmoidImpl());
    }

    public static SoftmaxImpl softmax(String name, long dim) {
        return register(name, new SoftmaxImpl(dim));
    }

//    public static SoftmaxImpl softmax(String name, long dim, long? dtype) {
//        // Note: dtype handling would require additional Options setup
//        return register(name, new SoftmaxImpl(dim));
//    }

    public static SoftmaxImpl softmax(long dim) {
        return create(new SoftmaxImpl(dim));
    }

    public static SoftmaxImpl softmax(String name, SoftmaxOptions options) {
        return register(name, new SoftmaxImpl(options));
    }
    public static SoftmaxImpl softmax(SoftmaxOptions options) {
        return create(new SoftmaxImpl(options));
    }

    public static SoftminImpl softmin(String name, long dim) {
        return register(name, new SoftminImpl(dim));
    }

    public static SoftminImpl softmin(long dim) {
        return create(new SoftminImpl(dim));
    }

    public static SoftminImpl softmin(String name, SoftminOptions options) {
        return register(name, new SoftminImpl(options));
    }
    public static SoftminImpl softmin(SoftminOptions options) {
        return create(new SoftminImpl(options));
    }

    public static LogSoftmaxImpl log_softmax(String name, long dim) {
        return register(name, new LogSoftmaxImpl(dim));
    }

    public static LogSoftmaxImpl log_softmax(long dim) {
        return create(new LogSoftmaxImpl(dim));
    }

    public static LogSoftmaxImpl log_softmax(String name, LogSoftmaxOptions options) {
        return register(name, new LogSoftmaxImpl(options));
    }
    public static LogSoftmaxImpl log_softmax(LogSoftmaxOptions options) {
        return create(new LogSoftmaxImpl(options));
    }

    public static Softmax2dImpl softmax2d(String name) {
        return register(name, new Softmax2dImpl());
    }

    public static Softmax2dImpl softmax2d() {
        return create(new Softmax2dImpl());
    }

    public static Softmax2dImpl softmax2d(String name, SoftmaxOptions options) {
        return register(name, new Softmax2dImpl(options));
    }
    public static Softmax2dImpl softmax2d(SoftmaxOptions options) {
        return create(new Softmax2dImpl(options));
    }

    public static SoftplusImpl softplus(String name) {
        return register(name, new SoftplusImpl());
    }

    public static SoftplusImpl softplus(String name, float beta, float threshold) {
        return register(name, new SoftplusImpl(new SoftplusOptions().beta(beta).threshold(threshold)));
    }

    public static SoftplusImpl softplus() {
        return create(new SoftplusImpl());
    }

    public static SoftplusImpl softplus(String name, SoftplusOptions options) {
        return register(name, new SoftplusImpl(options));
    }
    public static SoftplusImpl softplus(SoftplusOptions options) {
        return create(new SoftplusImpl(options));
    }

    public static SoftshrinkImpl softshrink(String name) {
        return register(name, new SoftshrinkImpl());
    }

    public static SoftshrinkImpl softshrink(String name, float lambda) {
        return register(name, new SoftshrinkImpl(new SoftshrinkOptions().lambda(lambda)));
    }

    public static SoftshrinkImpl softshrink() {
        return create(new SoftshrinkImpl());
    }

    public static SoftshrinkImpl softshrink(String name, SoftshrinkOptions options) {
        return register(name, new SoftshrinkImpl(options));
    }
    public static SoftshrinkImpl softshrink(SoftshrinkOptions options) {
        return create(new SoftshrinkImpl(options));
    }

    public static SoftsignImpl softsign(String name) {
        return register(name, new SoftsignImpl());
    }

    public static SoftsignImpl softsign() {
        return create(new SoftsignImpl());
    }

    public static TanhshrinkImpl tanhshrink(String name) {
        return register(name, new TanhshrinkImpl());
    }

    public static TanhshrinkImpl tanhshrink() {
        return create(new TanhshrinkImpl());
    }

    public static ThresholdImpl threshold(String name, float threshold, float value) {
        return register(name, new ThresholdImpl(threshold, value));
    }

    public static ThresholdImpl threshold(String name, ThresholdOptions options) {
        return register(name, new ThresholdImpl(options));
    }

    public static ThresholdImpl threshold(float threshold, float value) {
        return create(new ThresholdImpl(threshold, value));
    }

    public static ThresholdImpl threshold(ThresholdOptions options) {
        return create(new ThresholdImpl(options));
    }

    public static HardshrinkImpl hardshrink(String name) {
        return register(name, new HardshrinkImpl());
    }

    public static HardshrinkImpl hardshrink(String name, float lambda) {
        return register(name, new HardshrinkImpl(new HardshrinkOptions().lambda(lambda)));
    }

    public static HardshrinkImpl hardshrink() {
        return create(new HardshrinkImpl());
    }

    public static HardshrinkImpl hardshrink(String name, HardshrinkOptions options) {
        return register(name, new HardshrinkImpl(options));
    }
    public static HardshrinkImpl hardshrink(HardshrinkOptions options) {
        return create(new HardshrinkImpl(options));
    }

    public static LogSigmoidImpl log_sigmoid(String name) {
        return register(name, new LogSigmoidImpl());
    }

    public static LogSigmoidImpl log_sigmoid() {
        return create(new LogSigmoidImpl());
    }

    // ========================================================================
    // DROPOUT
    // ========================================================================

    public static DropoutImpl dropout(String name, float p) {
        return register(name, new DropoutImpl(p));
    }

    public static DropoutImpl dropout(String name, float p, boolean inplace) {
        return register(name, new DropoutImpl(new DropoutOptions(p).inplace(inplace)));
    }

    public static DropoutImpl dropout(float p) {
        return create(new DropoutImpl(p));
    }

    public static DropoutImpl dropout() {
        return create(new DropoutImpl());
    }

    public static DropoutImpl dropout(String name, DropoutOptions options) {
        return register(name, new DropoutImpl(options));
    }
    public static DropoutImpl dropout(DropoutOptions options) {
        return create(new DropoutImpl(options));
    }

    public static Dropout2dImpl dropout2d(String name, float p) {
        return register(name, new Dropout2dImpl(p));
    }

    public static Dropout2dImpl dropout2d(float p) {
        return create(new Dropout2dImpl(p));
    }

    public static Dropout2dImpl dropout2d() {
        return create(new Dropout2dImpl());
    }

    public static Dropout2dImpl dropout2d(String name, DropoutOptions options) {
        return register(name, new Dropout2dImpl(options));
    }
    public static Dropout2dImpl dropout2d(DropoutOptions options) {
        return create(new Dropout2dImpl(options));
    }

    public static Dropout3dImpl dropout3d(String name, float p) {
        return register(name, new Dropout3dImpl(p));
    }

    public static Dropout3dImpl dropout3d(float p) {
        return create(new Dropout3dImpl(p));
    }

    public static Dropout3dImpl dropout3d() {
        return create(new Dropout3dImpl());
    }

    public static Dropout3dImpl dropout3d(String name, DropoutOptions options) {
        return register(name, new Dropout3dImpl(options));
    }
    public static Dropout3dImpl dropout3d(DropoutOptions options) {
        return create(new Dropout3dImpl(options));
    }

    public static AlphaDropoutImpl alpha_dropout(String name, float p) {
        return register(name, new AlphaDropoutImpl(p));
    }

    public static AlphaDropoutImpl alpha_dropout(float p) {
        return create(new AlphaDropoutImpl(p));
    }

    public static AlphaDropoutImpl alpha_dropout() {
        return create(new AlphaDropoutImpl());
    }

    public static FeatureAlphaDropoutImpl feature_alpha_dropout(String name, float p) {
        return register(name, new FeatureAlphaDropoutImpl(p));
    }

    public static FeatureAlphaDropoutImpl feature_alpha_dropout(float p) {
        return create(new FeatureAlphaDropoutImpl(p));
    }

    public static FeatureAlphaDropoutImpl feature_alpha_dropout() {
        return create(new FeatureAlphaDropoutImpl());
    }

    // ========================================================================
    // SPARSE
    // ========================================================================

    public static EmbeddingImpl embedding(String name, long numEmbeddings, long embeddingDim) {
        return register(name, new EmbeddingImpl(numEmbeddings, embeddingDim));
    }

    public static EmbeddingImpl embedding(String name, EmbeddingOptions options) {
        return register(name, new EmbeddingImpl(options));
    }

    public static EmbeddingImpl embedding(long numEmbeddings, long embeddingDim) {
        return create(new EmbeddingImpl(numEmbeddings, embeddingDim));
    }

    public static EmbeddingImpl embedding(EmbeddingOptions options) {
        return create(new EmbeddingImpl(options));
    }

    public static EmbeddingBagImpl embedding_bag(String name, long numEmbeddings, long embeddingDim) {
        return register(name, new EmbeddingBagImpl(numEmbeddings, embeddingDim));
    }

    public static EmbeddingBagImpl embedding_bag(String name, EmbeddingBagOptions options) {
        return register(name, new EmbeddingBagImpl(options));
    }

    public static EmbeddingBagImpl embedding_bag(long numEmbeddings, long embeddingDim) {
        return create(new EmbeddingBagImpl(numEmbeddings, embeddingDim));
    }

    public static EmbeddingBagImpl embedding_bag(EmbeddingBagOptions options) {
        return create(new EmbeddingBagImpl(options));
    }

    // ========================================================================
    // LOSS FUNCTIONS
    // ========================================================================

    public static L1LossImpl l1_loss(String name) {
        return register(name, new L1LossImpl());
    }

    public static L1LossImpl l1_loss(String name, L1LossOptions options) {
        return register(name, new L1LossImpl(options));
    }

    public static L1LossImpl l1_loss() {
        return create(new L1LossImpl());
    }

    public static L1LossImpl l1_loss(L1LossOptions options) {
        return create(new L1LossImpl(options));
    }

    public static MSELossImpl mse_loss(String name) {
        return register(name, new MSELossImpl());
    }

    public static MSELossImpl mse_loss(String name, MSELossOptions options) {
        return register(name, new MSELossImpl(options));
    }

    public static MSELossImpl mse_loss() {
        return create(new MSELossImpl());
    }

    public static MSELossImpl mse_loss(MSELossOptions options) {
        return create(new MSELossImpl(options));
    }

    public static CrossEntropyLossImpl cross_entropy_loss(String name) {
        return register(name, new CrossEntropyLossImpl());
    }

    public static CrossEntropyLossImpl cross_entropy_loss(String name, CrossEntropyLossOptions options) {
        return register(name, new CrossEntropyLossImpl(options));
    }

    public static CrossEntropyLossImpl cross_entropy_loss() {
        return create(new CrossEntropyLossImpl());
    }

    public static CrossEntropyLossImpl cross_entropy_loss(CrossEntropyLossOptions options) {
        return create(new CrossEntropyLossImpl(options));
    }

    public static BCELossImpl bce_loss(String name) {
        return register(name, new BCELossImpl());
    }

    public static BCELossImpl bce_loss(String name, BCELossOptions options) {
        return register(name, new BCELossImpl(options));
    }

    public static BCELossImpl bce_loss() {
        return create(new BCELossImpl());
    }

    public static BCELossImpl bce_loss(BCELossOptions options) {
        return create(new BCELossImpl(options));
    }

    public static BCEWithLogitsLossImpl bce_with_logits_loss(String name) {
        return register(name, new BCEWithLogitsLossImpl());
    }

    public static BCEWithLogitsLossImpl bce_with_logits_loss(String name, BCEWithLogitsLossOptions options) {
        return register(name, new BCEWithLogitsLossImpl(options));
    }

    public static BCEWithLogitsLossImpl bce_with_logits_loss() {
        return create(new BCEWithLogitsLossImpl());
    }

    public static BCEWithLogitsLossImpl bce_with_logits_loss(BCEWithLogitsLossOptions options) {
        return create(new BCEWithLogitsLossImpl(options));
    }

    public static NLLLossImpl nll_loss(String name) {
        return register(name, new NLLLossImpl());
    }

    public static NLLLossImpl nll_loss(String name, NLLLossOptions options) {
        return register(name, new NLLLossImpl(options));
    }

    public static NLLLossImpl nll_loss() {
        return create(new NLLLossImpl());
    }

    public static NLLLossImpl nll_loss(NLLLossOptions options) {
        return create(new NLLLossImpl(options));
    }

    public static KLDivLossImpl kl_div_loss(String name) {
        return register(name, new KLDivLossImpl());
    }

    public static KLDivLossImpl kl_div_loss(String name, KLDivLossOptions options) {
        return register(name, new KLDivLossImpl(options));
    }

    public static KLDivLossImpl kl_div_loss() {
        return create(new KLDivLossImpl());
    }

    public static KLDivLossImpl kl_div_loss(KLDivLossOptions options) {
        return create(new KLDivLossImpl(options));
    }

    public static SmoothL1LossImpl smooth_l1_loss(String name) {
        return register(name, new SmoothL1LossImpl());
    }

    public static SmoothL1LossImpl smooth_l1_loss(String name, SmoothL1LossOptions options) {
        return register(name, new SmoothL1LossImpl(options));
    }

    public static SmoothL1LossImpl smooth_l1_loss() {
        return create(new SmoothL1LossImpl());
    }

    public static SmoothL1LossImpl smooth_l1_loss(SmoothL1LossOptions options) {
        return create(new SmoothL1LossImpl(options));
    }

    public static HuberLossImpl huber_loss(String name) {
        return register(name, new HuberLossImpl());
    }

    public static HuberLossImpl huber_loss(String name, HuberLossOptions options) {
        return register(name, new HuberLossImpl(options));
    }

    public static HuberLossImpl huber_loss() {
        return create(new HuberLossImpl());
    }

    public static HuberLossImpl huber_loss(HuberLossOptions options) {
        return create(new HuberLossImpl(options));
    }

    public static HingeEmbeddingLossImpl hinge_embedding_loss(String name) {
        return register(name, new HingeEmbeddingLossImpl());
    }

    public static HingeEmbeddingLossImpl hinge_embedding_loss(String name, HingeEmbeddingLossOptions options) {
        return register(name, new HingeEmbeddingLossImpl(options));
    }

    public static HingeEmbeddingLossImpl hinge_embedding_loss() {
        return create(new HingeEmbeddingLossImpl());
    }

    public static HingeEmbeddingLossImpl hinge_embedding_loss(HingeEmbeddingLossOptions options) {
        return create(new HingeEmbeddingLossImpl(options));
    }

    public static MultiMarginLossImpl multi_margin_loss(String name) {
        return register(name, new MultiMarginLossImpl());
    }

    public static MultiMarginLossImpl multi_margin_loss(String name, MultiMarginLossOptions options) {
        return register(name, new MultiMarginLossImpl(options));
    }

    public static MultiMarginLossImpl multi_margin_loss() {
        return create(new MultiMarginLossImpl());
    }

    public static MultiMarginLossImpl multi_margin_loss(MultiMarginLossOptions options) {
        return create(new MultiMarginLossImpl(options));
    }

    public static CosineEmbeddingLossImpl cosine_embedding_loss(String name) {
        return register(name, new CosineEmbeddingLossImpl());
    }

    public static CosineEmbeddingLossImpl cosine_embedding_loss(String name, CosineEmbeddingLossOptions options) {
        return register(name, new CosineEmbeddingLossImpl(options));
    }

    public static CosineEmbeddingLossImpl cosine_embedding_loss() {
        return create(new CosineEmbeddingLossImpl());
    }

    public static CosineEmbeddingLossImpl cosine_embedding_loss(CosineEmbeddingLossOptions options) {
        return create(new CosineEmbeddingLossImpl(options));
    }

    public static TripletMarginLossImpl triplet_margin_loss(String name) {
        return register(name, new TripletMarginLossImpl());
    }

    public static TripletMarginLossImpl triplet_margin_loss(String name, TripletMarginLossOptions options) {
        return register(name, new TripletMarginLossImpl(options));
    }

    public static TripletMarginLossImpl triplet_margin_loss() {
        return create(new TripletMarginLossImpl());
    }

    public static TripletMarginLossImpl triplet_margin_loss(TripletMarginLossOptions options) {
        return create(new TripletMarginLossImpl(options));
    }

    public static TripletMarginWithDistanceLossImpl triplet_margin_with_distance_loss(String name) {
        return register(name, new TripletMarginWithDistanceLossImpl());
    }

    public static TripletMarginWithDistanceLossImpl triplet_margin_with_distance_loss() {
        return create(new TripletMarginWithDistanceLossImpl());
    }

    public static TripletMarginWithDistanceLossImpl triplet_margin_with_distance_loss(String name, TripletMarginWithDistanceLossOptions options) {
        return register(name, new TripletMarginWithDistanceLossImpl(options));
    }
    public static TripletMarginWithDistanceLossImpl triplet_margin_with_distance_loss(TripletMarginWithDistanceLossOptions options) {
        return create(new TripletMarginWithDistanceLossImpl(options));
    }

    public static CTCLossImpl ctc_loss(String name) {
        return register(name, new CTCLossImpl());
    }

    public static CTCLossImpl ctc_loss(String name, CTCLossOptions options) {
        return register(name, new CTCLossImpl(options));
    }

    public static CTCLossImpl ctc_loss() {
        return create(new CTCLossImpl());
    }

    public static CTCLossImpl ctc_loss(CTCLossOptions options) {
        return create(new CTCLossImpl(options));
    }

    public static PoissonNLLLossImpl poisson_nll_loss(String name) {
        return register(name, new PoissonNLLLossImpl());
    }

    public static PoissonNLLLossImpl poisson_nll_loss(String name, PoissonNLLLossOptions options) {
        return register(name, new PoissonNLLLossImpl(options));
    }

    public static PoissonNLLLossImpl poisson_nll_loss() {
        return create(new PoissonNLLLossImpl());
    }

    public static PoissonNLLLossImpl poisson_nll_loss(PoissonNLLLossOptions options) {
        return create(new PoissonNLLLossImpl(options));
    }

    public static MarginRankingLossImpl margin_ranking_loss(String name) {
        return register(name, new MarginRankingLossImpl());
    }

    public static MarginRankingLossImpl margin_ranking_loss(String name, MarginRankingLossOptions options) {
        return register(name, new MarginRankingLossImpl(options));
    }

    public static MarginRankingLossImpl margin_ranking_loss() {
        return create(new MarginRankingLossImpl());
    }

    public static MarginRankingLossImpl margin_ranking_loss(MarginRankingLossOptions options) {
        return create(new MarginRankingLossImpl(options));
    }

    public static MultiLabelMarginLossImpl multi_label_margin_loss(String name) {
        return register(name, new MultiLabelMarginLossImpl());
    }

    public static MultiLabelMarginLossImpl multi_label_margin_loss(String name, MultiLabelMarginLossOptions options) {
        return register(name, new MultiLabelMarginLossImpl(options));
    }

    public static MultiLabelMarginLossImpl multi_label_margin_loss() {
        return create(new MultiLabelMarginLossImpl());
    }

    public static MultiLabelMarginLossImpl multi_label_margin_loss(MultiLabelMarginLossOptions options) {
        return create(new MultiLabelMarginLossImpl(options));
    }

    public static SoftMarginLossImpl soft_margin_loss(String name) {
        return register(name, new SoftMarginLossImpl());
    }

    public static SoftMarginLossImpl soft_margin_loss(String name, SoftMarginLossOptions options) {
        return register(name, new SoftMarginLossImpl(options));
    }

    public static SoftMarginLossImpl soft_margin_loss() {
        return create(new SoftMarginLossImpl());
    }

    public static SoftMarginLossImpl soft_margin_loss(SoftMarginLossOptions options) {
        return create(new SoftMarginLossImpl(options));
    }

    public static MultiLabelSoftMarginLossImpl multi_label_soft_margin_loss(String name) {
        return register(name, new MultiLabelSoftMarginLossImpl());
    }

    public static MultiLabelSoftMarginLossImpl multi_label_soft_margin_loss(String name, MultiLabelSoftMarginLossOptions options) {
        return register(name, new MultiLabelSoftMarginLossImpl(options));
    }

    public static MultiLabelSoftMarginLossImpl multi_label_soft_margin_loss() {
        return create(new MultiLabelSoftMarginLossImpl());
    }

    public static MultiLabelSoftMarginLossImpl multi_label_soft_margin_loss(MultiLabelSoftMarginLossOptions options) {
        return create(new MultiLabelSoftMarginLossImpl(options));
    }

    // ========================================================================
    // VISION
    // ========================================================================

    public static FlattenImpl flatten(String name, long startDim, long endDim) {
        return register(name, new FlattenImpl(new FlattenOptions().start_dim(startDim).end_dim(endDim)));
    }

    public static FlattenImpl flatten(String name) {
        return register(name, new FlattenImpl());
    }

    public static FlattenImpl flatten(long startDim, long endDim) {
        return create(new FlattenImpl(new FlattenOptions().start_dim(startDim).end_dim(endDim)));
    }

    public static FlattenImpl flatten() {
        return create(new FlattenImpl());
    }

    public static FlattenImpl flatten(String name, FlattenOptions options) {
        return register(name, new FlattenImpl(options));
    }
    public static FlattenImpl flatten(FlattenOptions options) {
        return create(new FlattenImpl(options));
    }

    public static UnflattenImpl unflatten(String name, long dim, long... shape) {
        LongVector vec = new LongVector(shape);
        return register(name, new UnflattenImpl(dim, vec));
    }

    public static UnflattenImpl unflatten(long dim, long... shape) {
        LongVector vec = new LongVector(shape);
        return create(new UnflattenImpl(dim, vec));
    }

    public static UnflattenImpl unflatten(String name, UnflattenOptions options) {
        return register(name, new UnflattenImpl(options));
    }
    public static UnflattenImpl unflatten(UnflattenOptions options) {
        return create(new UnflattenImpl(options));
    }

    public static IdentityImpl identity(String name) {
        return register(name, new IdentityImpl());
    }

    public static IdentityImpl identity() {
        return create(new IdentityImpl());
    }

    public static PixelShuffleImpl pixel_shuffle(String name, long upscaleFactor) {
        return register(name, new PixelShuffleImpl(new PixelShuffleOptions(upscaleFactor).upscale_factor(upscaleFactor)));
    }

    public static PixelShuffleImpl pixel_shuffle(long upscaleFactor) {
        return create(new PixelShuffleImpl(new PixelShuffleOptions(upscaleFactor).upscale_factor(upscaleFactor)));
    }

    public static PixelShuffleImpl pixel_shuffle(String name, PixelShuffleOptions options) {
        return register(name, new PixelShuffleImpl(options));
    }
    public static PixelShuffleImpl pixel_shuffle(PixelShuffleOptions options) {
        return create(new PixelShuffleImpl(options));
    }

    public static PixelUnshuffleImpl pixel_unshuffle(String name, long downscaleFactor) {
        return register(name, new PixelUnshuffleImpl(new PixelUnshuffleOptions(downscaleFactor).downscale_factor(downscaleFactor)));
    }

    public static PixelUnshuffleImpl pixel_unshuffle(long downscaleFactor) {
        return create(new PixelUnshuffleImpl(new PixelUnshuffleOptions(downscaleFactor).downscale_factor(downscaleFactor)));
    }

    public static PixelUnshuffleImpl pixel_unshuffle(String name, PixelUnshuffleOptions options) {
        return register(name, new PixelUnshuffleImpl(options));
    }
    public static PixelUnshuffleImpl pixel_unshuffle(PixelUnshuffleOptions options) {
        return create(new PixelUnshuffleImpl(options));
    }

    public static UpsampleImpl upsample(String name, long... size) {
        LongVector vec = new LongVector(size);
        return register(name, new UpsampleImpl(vec));
    }

    public static UpsampleImpl upsample(String name, UpsampleOptions options) {
        return register(name, new UpsampleImpl(options));
    }

    public static UpsampleImpl upsample(long... size) {
        LongVector vec = new LongVector(size);
        return create(new UpsampleImpl(vec));
    }

    public static UpsampleImpl upsample(UpsampleOptions options) {
        return create(new UpsampleImpl(options));
    }

    public static FoldImpl fold(String name, long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return register(name, new FoldImpl(vec));
    }

    public static FoldImpl fold(long... outputSize) {
        LongVector vec = new LongVector(outputSize);
        return create(new FoldImpl(vec));
    }

    public static FoldImpl fold(String name, FoldOptions options) {
        return register(name, new FoldImpl(options));
    }
    public static FoldImpl fold(FoldOptions options) {
        return create(new FoldImpl(options));
    }

    public static UnfoldImpl unfold(String name, long... kernelSize) {
        LongVector vec = new LongVector(kernelSize);
        return register(name, new UnfoldImpl(vec));
    }

    public static UnfoldImpl unfold(long... kernelSize) {
        LongVector vec = new LongVector(kernelSize);
        return create(new UnfoldImpl(vec));
    }

    public static UnfoldImpl unfold(String name, UnfoldOptions options) {
        return register(name, new UnfoldImpl(options));
    }
    public static UnfoldImpl unfold(UnfoldOptions options) {
        return create(new UnfoldImpl(options));
    }

    // ========================================================================
    // DISTANCE
    // ========================================================================

    public static CosineSimilarityImpl cosine_similarity(String name) {
        return register(name, new CosineSimilarityImpl());
    }

    public static CosineSimilarityImpl cosine_similarity(String name, CosineSimilarityOptions options) {
        return register(name, new CosineSimilarityImpl(options));
    }

    public static CosineSimilarityImpl cosine_similarity() {
        return create(new CosineSimilarityImpl());
    }

    public static CosineSimilarityImpl cosine_similarity(CosineSimilarityOptions options) {
        return create(new CosineSimilarityImpl(options));
    }

    public static PairwiseDistanceImpl pairwise_distance(String name) {
        return register(name, new PairwiseDistanceImpl());
    }

    public static PairwiseDistanceImpl pairwise_distance(String name, PairwiseDistanceOptions options) {
        return register(name, new PairwiseDistanceImpl(options));
    }

    public static PairwiseDistanceImpl pairwise_distance() {
        return create(new PairwiseDistanceImpl());
    }

    public static PairwiseDistanceImpl pairwise_distance(PairwiseDistanceOptions options) {
        return create(new PairwiseDistanceImpl(options));
    }

    // ========================================================================
    // CONTAINERS
    // ========================================================================

    /**
     * Create and register a Sequential module.
     *
     * @param name module name
     * @return the created SequentialImpl
     */
    public static SequentialImpl sequential(String name) {
        SequentialImpl seq = new SequentialImpl();
        Module parent = getParent();
        if (parent != null) {
            parent.register_module(name, seq);
        }
        return seq;
    }

    /**
     * Create a Sequential module (for use without registration).
     *
     * @return the created SequentialImpl
     */
    public static SequentialImpl sequential() {
        return create(new SequentialImpl());
    }

    /**
     * Create and register a ModuleList.
     *
     * @param name module name
     * @return the created ModuleListImpl
     */
    public static ModuleListImpl module_list(String name) {
        ModuleListImpl list = new ModuleListImpl();
        Module parent = getParent();
        if (parent != null) {
            parent.register_module(name, list);
        }
        return list;
    }

    /**
     * Create a ModuleList.
     *
     * @return the created ModuleListImpl
     */
    public static ModuleListImpl module_list() {
        return create(new ModuleListImpl());
    }

    /**
     * Create and register a ModuleDict.
     *
     * @param name module name
     * @return the created ModuleDictImpl
     */
    public static ModuleDictImpl module_dict(String name) {
        ModuleDictImpl dict = new ModuleDictImpl();
        Module parent = getParent();
        if (parent != null) {
            parent.register_module(name, dict);
        }
        return dict;
    }

    /**
     * Create a ModuleDict.
     *
     * @return the created ModuleDictImpl
     */
    public static ModuleDictImpl module_dict() {
        return create(new ModuleDictImpl());
    }

    /**
     * Create and register a ParameterList.
     *
     * @param name module name
     * @return the created ParameterListImpl
     */
    public static ParameterListImpl parameter_list(String name) {
        ParameterListImpl list = new ParameterListImpl();
        Module parent = getParent();
        if (parent != null) {
            parent.register_module(name, list);
        }
        return list;
    }

    /**
     * Create a ParameterList.
     *
     * @return the created ParameterListImpl
     */
    public static ParameterListImpl parameter_list() {
        return create(new ParameterListImpl());
    }

    /**
     * Create and register a ParameterDict.
     *
     * @param name module name
     * @return the created ParameterDictImpl
     */
    public static ParameterDictImpl parameter_dict(String name) {
        ParameterDictImpl dict = new ParameterDictImpl();
        Module parent = getParent();
        if (parent != null) {
            parent.register_module(name, dict);
        }
        return dict;
    }

    /**
     * Create a ParameterDict.
     *
     * @return the created ParameterDictImpl
     */
    public static ParameterDictImpl parameter_dict() {
        return create(new ParameterDictImpl());
    }

    // ========================================================================
    // RECURRENT
    // ========================================================================

    public static RNNImpl rnn(String name, long inputSize, long hiddenSize, long numLayers) {
        return register(name, new RNNImpl(new RNNOptions(inputSize, hiddenSize).hidden_size(hiddenSize).num_layers(numLayers)));
    }

    public static RNNImpl rnn(String name, RNNOptions options) {
        return register(name, new RNNImpl(options));
    }

    public static RNNImpl rnn(long inputSize, long hiddenSize, long numLayers) {
        return create(new RNNImpl(new RNNOptions(inputSize, hiddenSize).input_size(inputSize).hidden_size(hiddenSize).num_layers(numLayers)));
    }

    public static RNNImpl rnn(RNNOptions options) {
        return create(new RNNImpl(options));
    }

    public static LSTMImpl lstm(String name, long inputSize, long hiddenSize, long numLayers) {
        return register(name, new LSTMImpl(new LSTMOptions(inputSize, hiddenSize).input_size(inputSize).hidden_size(hiddenSize).num_layers(numLayers)));
    }

    public static LSTMImpl lstm(String name, LSTMOptions options) {
        return register(name, new LSTMImpl(options));
    }

    public static LSTMImpl lstm(long inputSize, long hiddenSize, long numLayers) {
        return create(new LSTMImpl(new LSTMOptions(inputSize, hiddenSize).input_size(inputSize).hidden_size(hiddenSize).num_layers(numLayers)));
    }

    public static LSTMImpl lstm(LSTMOptions options) {
        return create(new LSTMImpl(options));
    }

    public static GRUImpl gru(String name, long inputSize, long hiddenSize, long numLayers) {
        return register(name, new GRUImpl(new GRUOptions(inputSize, hiddenSize).input_size(inputSize).hidden_size(hiddenSize).num_layers(numLayers)));
    }

    public static GRUImpl gru(String name, GRUOptions options) {
        return register(name, new GRUImpl(options));
    }

    public static GRUImpl gru(long inputSize, long hiddenSize, long numLayers) {
        return create(new GRUImpl(new GRUOptions(inputSize, hiddenSize).input_size(inputSize).hidden_size(hiddenSize).num_layers(numLayers)));
    }

    public static GRUImpl gru(GRUOptions options) {
        return create(new GRUImpl(options));
    }

    public static RNNCellImpl rnn_cell(String name, long inputSize, long hiddenSize) {
        return register(name, new RNNCellImpl(new RNNCellOptions(inputSize, hiddenSize).input_size(inputSize).hidden_size(hiddenSize)));
    }

    public static RNNCellImpl rnn_cell(String name, RNNCellOptions options) {
        return register(name, new RNNCellImpl(options));
    }

    public static RNNCellImpl rnn_cell(long inputSize, long hiddenSize) {
        return create(new RNNCellImpl(new RNNCellOptions(inputSize, hiddenSize).input_size(inputSize).hidden_size(hiddenSize)));
    }

    public static RNNCellImpl rnn_cell(RNNCellOptions options) {
        return create(new RNNCellImpl(options));
    }

    public static LSTMCellImpl lstm_cell(String name, long inputSize, long hiddenSize) {
        return register(name, new LSTMCellImpl(new LSTMCellOptions(inputSize, hiddenSize).input_size(inputSize).hidden_size(hiddenSize)));
    }

    public static LSTMCellImpl lstm_cell(String name, LSTMCellOptions options) {
        return register(name, new LSTMCellImpl(options));
    }

    public static LSTMCellImpl lstm_cell(long inputSize, long hiddenSize) {
        return create(new LSTMCellImpl(new LSTMCellOptions(inputSize, hiddenSize).input_size(inputSize).hidden_size(hiddenSize)));
    }

    public static LSTMCellImpl lstm_cell(LSTMCellOptions options) {
        return create(new LSTMCellImpl(options));
    }

    public static GRUCellImpl gru_cell(String name, long inputSize, long hiddenSize) {
        return register(name, new GRUCellImpl(new GRUCellOptions(inputSize, hiddenSize).input_size(inputSize).hidden_size(hiddenSize)));
    }

    public static GRUCellImpl gru_cell(String name, GRUCellOptions options) {
        return register(name, new GRUCellImpl(options));
    }

    public static GRUCellImpl gru_cell(long inputSize, long hiddenSize) {
        return create(new GRUCellImpl(new GRUCellOptions(inputSize, hiddenSize).input_size(inputSize).hidden_size(hiddenSize)));
    }

    public static GRUCellImpl gru_cell(GRUCellOptions options) {
        return create(new GRUCellImpl(options));
    }

    // ========================================================================
    // TRANSFORMER
    // ========================================================================

    public static TransformerImpl transformer(String name) {
        return register(name, new TransformerImpl(new Pointer()));
    }

    public static TransformerImpl transformer(String name, TransformerOptions options) {
        return register(name, new TransformerImpl(options));
    }

    public static TransformerImpl transformer() {
        return create(new TransformerImpl(new Pointer()));
    }

    public static TransformerImpl transformer(TransformerOptions options) {
        return create(new TransformerImpl(options));
    }

    public static TransformerEncoderImpl transformer_encoder(String name) {
        return register(name, new TransformerEncoderImpl(new Pointer()));
    }

    public static TransformerEncoderImpl transformer_encoder(String name, TransformerEncoderOptions options) {
        return register(name, new TransformerEncoderImpl(options));
    }

    public static TransformerEncoderImpl transformer_encoder() {
        return create(new TransformerEncoderImpl(new Pointer()));
    }

    public static TransformerEncoderImpl transformer_encoder(TransformerEncoderOptions options) {
        return create(new TransformerEncoderImpl(options));
    }

    public static TransformerDecoderImpl transformer_decoder(String name) {
        return register(name, new TransformerDecoderImpl(new Pointer()));
    }

    public static TransformerDecoderImpl transformer_decoder(String name, TransformerDecoderOptions options) {
        return register(name, new TransformerDecoderImpl(options));
    }

    public static TransformerDecoderImpl transformer_decoder() {
        return create(new TransformerDecoderImpl(new Pointer()));
    }

    public static TransformerDecoderImpl transformer_decoder(TransformerDecoderOptions options) {
        return create(new TransformerDecoderImpl(options));
    }

    public static TransformerEncoderLayerImpl transformer_encoder_layer(String name) {
        return register(name, new TransformerEncoderLayerImpl(new Pointer()));
    }

    public static TransformerEncoderLayerImpl transformer_encoder_layer(String name, TransformerEncoderLayerOptions options) {
        return register(name, new TransformerEncoderLayerImpl(options));
    }

    public static TransformerEncoderLayerImpl transformer_encoder_layer() {
        return create(new TransformerEncoderLayerImpl(new Pointer()));
    }

    public static TransformerEncoderLayerImpl transformer_encoder_layer(TransformerEncoderLayerOptions options) {
        return create(new TransformerEncoderLayerImpl(options));
    }

    public static TransformerDecoderLayerImpl transformer_decoder_layer(String name) {
        return register(name, new TransformerDecoderLayerImpl(new Pointer()));
    }

    public static TransformerDecoderLayerImpl transformer_decoder_layer(String name, TransformerDecoderLayerOptions options) {
        return register(name, new TransformerDecoderLayerImpl(options));
    }

    public static TransformerDecoderLayerImpl transformer_decoder_layer() {
        return create(new TransformerDecoderLayerImpl(new Pointer()));
    }

    public static TransformerDecoderLayerImpl transformer_decoder_layer(TransformerDecoderLayerOptions options) {
        return create(new TransformerDecoderLayerImpl(options));
    }

    public static MultiheadAttentionImpl multihead_attention(String name) {
        return register(name, new MultiheadAttentionImpl(new Pointer()));
    }

    public static MultiheadAttentionImpl multihead_attention(String name, MultiheadAttentionOptions options) {
        return register(name, new MultiheadAttentionImpl(options));
    }

    public static MultiheadAttentionImpl multihead_attention() {
        return create(new MultiheadAttentionImpl(new Pointer()));
    }

    public static MultiheadAttentionImpl multihead_attention(MultiheadAttentionOptions options) {
        return create(new MultiheadAttentionImpl(options));
    }

    // ========================================================================
    // ADAPTIVE LOG SOFTMAX WITH LOSS
    // ========================================================================

    public static AdaptiveLogSoftmaxWithLossImpl adaptive_log_softmax_with_loss(String name, long nClasses, long nIn, long[] cutoffs) {
        return register(name, new AdaptiveLogSoftmaxWithLossImpl(new AdaptiveLogSoftmaxWithLossOptions(nClasses, nIn, new LongVector(cutoffs)).n_classes(nClasses).in_features(nIn).cutoffs(new LongVector(cutoffs))));
    }

    public static AdaptiveLogSoftmaxWithLossImpl adaptive_log_softmax_with_loss(String name, AdaptiveLogSoftmaxWithLossOptions options) {
        return register(name, new AdaptiveLogSoftmaxWithLossImpl(options));
    }

    public static AdaptiveLogSoftmaxWithLossImpl adaptive_log_softmax_with_loss(long nClasses, long nIn, long[] cutoffs) {
        return create(new AdaptiveLogSoftmaxWithLossImpl(new AdaptiveLogSoftmaxWithLossOptions(nClasses, nIn, new LongVector(cutoffs)).n_classes(nClasses).in_features(nIn).cutoffs(new LongVector(cutoffs))));
    }

    public static AdaptiveLogSoftmaxWithLossImpl adaptive_log_softmax_with_loss(AdaptiveLogSoftmaxWithLossOptions options) {
        return create(new AdaptiveLogSoftmaxWithLossImpl(options));
    }
}
