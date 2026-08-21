package org.bytedeco.pytorch.nn;

import org.bytedeco.javacpp.BooleanPointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.DoubleOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorList;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.enumtype.Conv1dPadding;
import org.bytedeco.pytorch.enumtype.Conv2dPadding;
import org.bytedeco.pytorch.enumtype.Conv3dPadding;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.options.AlphaDropoutFuncOptions;
import org.bytedeco.pytorch.nn.options.BatchNormFuncOptions;
import org.bytedeco.pytorch.nn.options.Conv1dFuncOptions;
import org.bytedeco.pytorch.nn.options.Conv2dFuncOptions;
import org.bytedeco.pytorch.nn.options.Conv3dFuncOptions;
import org.bytedeco.pytorch.nn.options.ConvTranspose1dFuncOptions;
import org.bytedeco.pytorch.nn.options.ConvTranspose2dFuncOptions;
import org.bytedeco.pytorch.nn.options.ConvTranspose3dFuncOptions;
import org.bytedeco.pytorch.nn.options.DropoutFuncOptions;
import org.bytedeco.pytorch.nn.options.ELUOptions;
import org.bytedeco.pytorch.nn.options.EmbeddingBagFuncOptions;
import org.bytedeco.pytorch.nn.options.EmbeddingFuncOptions;
import org.bytedeco.pytorch.nn.options.FeatureAlphaDropoutFuncOptions;
import org.bytedeco.pytorch.nn.options.FoldOptions;
import org.bytedeco.pytorch.nn.options.GELUOptions;
import org.bytedeco.pytorch.nn.options.GLUOptions;
import org.bytedeco.pytorch.nn.options.GRUCellOptions;
import org.bytedeco.pytorch.nn.options.GRUOptions;
import org.bytedeco.pytorch.nn.options.GumbelSoftmaxFuncOptions;
import org.bytedeco.pytorch.nn.options.HardshrinkOptions;
import org.bytedeco.pytorch.nn.options.HardtanhOptions;
import org.bytedeco.pytorch.nn.options.InstanceNormFuncOptions;
import org.bytedeco.pytorch.nn.options.L1LossOptions;
import org.bytedeco.pytorch.nn.options.LSTMCellOptions;
import org.bytedeco.pytorch.nn.options.LSTMOptions;
import org.bytedeco.pytorch.nn.options.LeakyReLUOptions;
import org.bytedeco.pytorch.nn.options.LocalResponseNormOptions;
import org.bytedeco.pytorch.nn.options.LogSoftmaxFuncOptions;
import org.bytedeco.pytorch.nn.options.MSELossOptions;
import org.bytedeco.pytorch.nn.options.MaxUnpool1dFuncOptions;
import org.bytedeco.pytorch.nn.options.MaxUnpool2dFuncOptions;
import org.bytedeco.pytorch.nn.options.MaxUnpool3dFuncOptions;
import org.bytedeco.pytorch.nn.options.NLLLossOptions;
import org.bytedeco.pytorch.nn.options.NormalizeFuncOptions;
import org.bytedeco.pytorch.nn.options.PadFuncOptions;
import org.bytedeco.pytorch.nn.options.PoissonNLLLossOptions;
import org.bytedeco.pytorch.nn.options.ReLU6Options;
import org.bytedeco.pytorch.nn.options.ReLUOptions;
import org.bytedeco.pytorch.nn.options.RNNCellOptions;
import org.bytedeco.pytorch.nn.options.RNNOptions;
import org.bytedeco.pytorch.nn.options.RReLUOptions;
import org.bytedeco.pytorch.nn.options.SELUOptions;
import org.bytedeco.pytorch.nn.options.SmoothL1LossOptions;
import org.bytedeco.pytorch.nn.options.SoftMarginLossOptions;
import org.bytedeco.pytorch.nn.options.SoftmaxFuncOptions;
import org.bytedeco.pytorch.nn.options.SoftminFuncOptions;
import org.bytedeco.pytorch.nn.options.SoftplusOptions;
import org.bytedeco.pytorch.nn.options.SoftshrinkOptions;
import org.bytedeco.pytorch.nn.options.ThresholdOptions;
import org.bytedeco.pytorch.nn.options.TripletMarginLossOptions;
import org.bytedeco.pytorch.nn.options.UnfoldOptions;
import org.bytedeco.pytorch.nn.options.BCELossOptions;
import org.bytedeco.pytorch.nn.options.BCEWithLogitsLossOptions;
import org.bytedeco.pytorch.nn.options.CosineEmbeddingLossOptions;
import org.bytedeco.pytorch.nn.options.CosineSimilarityOptions;
import org.bytedeco.pytorch.nn.options.CrossEntropyLossOptions;
import org.bytedeco.pytorch.nn.options.CTCLossOptions;
import org.bytedeco.pytorch.nn.options.HingeEmbeddingLossOptions;
import org.bytedeco.pytorch.nn.options.HuberLossOptions;
import org.bytedeco.pytorch.nn.options.KLDivLossOptions;
import org.bytedeco.pytorch.nn.options.MarginRankingLossOptions;
import org.bytedeco.pytorch.nn.options.MultiLabelMarginLossOptions;
import org.bytedeco.pytorch.nn.options.MultiLabelSoftMarginLossOptions;
import org.bytedeco.pytorch.nn.options.MultiMarginLossOptions;
import org.bytedeco.pytorch.nn.options.PairwiseDistanceOptions;
import org.bytedeco.pytorch.nn.options.TripletMarginWithDistanceLossOptions;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * PyTorch-style functional API.
 *
 * <p>This class mirrors {@code torch.nn.functional} from Python PyTorch.
 * Functions are statically available and most delegate to {@link torch}
 * which exposes the native {@code at::} and {@code torch::nn::functional}
 * C++ APIs from libtorch.
 *
 * <p>Signatures match the Python API as closely as Java types permit. The
 * underlying native calls accept the {@link TensorOptions} and feature
 * flags described in the Python reference.
 */
public class Functional {

    private Functional() {}

    // ========================================================================
    // CONVOLUTION
    // ========================================================================

    /**
     * 1D convolution functional.
     *
     * <p>Python signature: {@code F.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)}
     */
    public static Tensor conv1d(Tensor input, Tensor weight, Tensor bias,
                                long[] stride, long[] padding, long[] dilation, long groups) {
        Conv1dFuncOptions opt = new Conv1dFuncOptions()
                .stride(new LongPointer(stride))
                .padding(new Conv1dPadding(new LongPointer(padding)))
                .dilation(new LongPointer(dilation))
                .groups(groups);
        if (bias != null) {
            opt.bias(bias);
        }
        return torch.conv1d(input, weight, opt);
    }

    public static Tensor conv1d(Tensor input, Tensor weight, long[] stride, long[] padding, long[] dilation, long groups) {
        return conv1d(input, weight, null, stride, padding, dilation, groups);
    }

    public static Tensor conv1d(Tensor input, Tensor weight, Tensor bias, long stride, long padding, long dilation, long groups) {
        return conv1d(input, weight, bias, new long[]{stride}, new long[]{padding}, new long[]{dilation}, groups);
    }

    public static Tensor conv1d(Tensor input, Tensor weight, long stride, long padding, long dilation, long groups) {
        return conv1d(input, weight, null, new long[]{stride}, new long[]{padding}, new long[]{dilation}, groups);
    }

    public static Tensor conv1d(Tensor input, Tensor weight) {
        return torch.conv1d(input, weight, new Conv1dFuncOptions());
    }

    public static Tensor conv1d(Tensor input, Tensor weight, Conv1dFuncOptions options) {
        return torch.conv1d(input, weight, options);
    }

    /** 2D convolution functional. */
    public static Tensor conv2d(Tensor input, Tensor weight, Tensor bias,
                                long[] stride, long[] padding, long[] dilation, long groups) {
        Conv2dFuncOptions opt = new Conv2dFuncOptions()
                .stride(new LongPointer(stride))
                .padding(new Conv2dPadding(new LongPointer(padding)))
                .dilation(new LongPointer(dilation))
                .groups(groups);
        if (bias != null) {
            opt.bias(bias);
        }
        return torch.conv2d(input, weight, opt);
    }

    public static Tensor conv2d(Tensor input, Tensor weight, long[] stride, long[] padding, long[] dilation, long groups) {
        return conv2d(input, weight, null, stride, padding, dilation, groups);
    }

    public static Tensor conv2d(Tensor input, Tensor weight, Tensor bias, long stride, long padding, long dilation, long groups) {
        return conv2d(input, weight, bias, new long[]{stride, stride}, new long[]{padding, padding}, new long[]{dilation, dilation}, groups);
    }

    public static Tensor conv2d(Tensor input, Tensor weight, long stride, long padding, long dilation, long groups) {
        return conv2d(input, weight, null, new long[]{stride, stride}, new long[]{padding, padding}, new long[]{dilation, dilation}, groups);
    }

    public static Tensor conv2d(Tensor input, Tensor weight) {
        return torch.conv2d(input, weight, new Conv2dFuncOptions());
    }

    public static Tensor conv2d(Tensor input, Tensor weight, Conv2dFuncOptions options) {
        return torch.conv2d(input, weight, options);
    }

    /** 3D convolution functional. */
    public static Tensor conv3d(Tensor input, Tensor weight, Tensor bias,
                                long[] stride, long[] padding, long[] dilation, long groups) {
        Conv3dFuncOptions opt = new Conv3dFuncOptions()
                .stride(new LongPointer(stride))
                .padding(new Conv3dPadding(new LongPointer(padding)))
                .dilation(new LongPointer(dilation))
                .groups(groups);
        if (bias != null) {
            opt.bias(bias);
        }
        return torch.conv3d(input, weight, opt);
    }

    public static Tensor conv3d(Tensor input, Tensor weight, long[] stride, long[] padding, long[] dilation, long groups) {
        return conv3d(input, weight, null, stride, padding, dilation, groups);
    }

    public static Tensor conv3d(Tensor input, Tensor weight, Tensor bias, long stride, long padding, long dilation, long groups) {
        return conv3d(input, weight, bias, new long[]{stride}, new long[]{padding}, new long[]{dilation}, groups);
    }

    public static Tensor conv3d(Tensor input, Tensor weight, long stride, long padding, long dilation, long groups) {
        return conv3d(input, weight, null, new long[]{stride}, new long[]{padding}, new long[]{dilation}, groups);
    }

    public static Tensor conv3d(Tensor input, Tensor weight) {
        return torch.conv3d(input, weight, new Conv3dFuncOptions());
    }

    public static Tensor conv3d(Tensor input, Tensor weight, Conv3dFuncOptions options) {
        return torch.conv3d(input, weight, options);
    }

    /** ConvTranspose1d. */
    public static Tensor conv_transpose1d(Tensor input, Tensor weight, Tensor bias,
                                          long[] stride, long[] padding, long[] output_padding,
                                          long groups, long[] dilation) {
        ConvTranspose1dFuncOptions opt = new ConvTranspose1dFuncOptions()
                .stride(new LongPointer(stride))
                .padding(new LongPointer(padding))
                .output_padding(new LongPointer(output_padding))
                .groups(groups)
                .dilation(new LongPointer(dilation));
        if (bias != null) {
            opt.bias(bias);
        }
        return torch.conv_transpose1d(input, weight, opt);
    }

    public static Tensor conv_transpose1d(Tensor input, Tensor weight, long stride, long padding, long output_padding, long groups, long dilation) {
        return conv_transpose1d(input, weight, null,
                new long[]{stride}, new long[]{padding}, new long[]{output_padding}, groups,
                new long[]{dilation});
    }

    public static Tensor conv_transpose1d(Tensor input, Tensor weight) {
        return torch.conv_transpose1d(input, weight, new ConvTranspose1dFuncOptions());
    }

    public static Tensor conv_transpose1d(Tensor input, Tensor weight, ConvTranspose1dFuncOptions options) {
        return torch.conv_transpose1d(input, weight, options);
    }

    /** ConvTranspose2d. */
    public static Tensor conv_transpose2d(Tensor input, Tensor weight, Tensor bias,
                                          long[] stride, long[] padding, long[] output_padding,
                                          long groups, long[] dilation) {
        ConvTranspose2dFuncOptions opt = new ConvTranspose2dFuncOptions()
                .stride(new LongPointer(stride))
                .padding(new LongPointer(padding))
                .output_padding(new LongPointer(output_padding))
                .groups(groups)
                .dilation(new LongPointer(dilation));
        if (bias != null) {
            opt.bias(bias);
        }
        return torch.conv_transpose2d(input, weight, opt);
    }

    public static Tensor conv_transpose2d(Tensor input, Tensor weight, long stride, long padding, long output_padding, long groups, long dilation) {
        return conv_transpose2d(input, weight, null,
                new long[]{stride, stride}, new long[]{padding, padding}, new long[]{output_padding, output_padding}, groups,
                new long[]{dilation, dilation});
    }

    public static Tensor conv_transpose2d(Tensor input, Tensor weight) {
        return torch.conv_transpose2d(input, weight, new ConvTranspose2dFuncOptions());
    }

    public static Tensor conv_transpose2d(Tensor input, Tensor weight, ConvTranspose2dFuncOptions options) {
        return torch.conv_transpose2d(input, weight, options);
    }

    /** ConvTranspose3d. */
    public static Tensor conv_transpose3d(Tensor input, Tensor weight, Tensor bias,
                                          long[] stride, long[] padding, long[] output_padding,
                                          long groups, long[] dilation) {
        ConvTranspose3dFuncOptions opt = new ConvTranspose3dFuncOptions()
                .stride(new LongPointer(stride))
                .padding(new LongPointer(padding))
                .output_padding(new LongPointer(output_padding))
                .groups(groups)
                .dilation(new LongPointer(dilation));
        if (bias != null) {
            opt.bias(bias);
        }
        return torch.conv_transpose3d(input, weight, opt);
    }

    public static Tensor conv_transpose3d(Tensor input, Tensor weight, long stride, long padding, long output_padding, long groups, long dilation) {
        return conv_transpose3d(input, weight, null,
                new long[]{stride}, new long[]{padding}, new long[]{output_padding}, groups,
                new long[]{dilation});
    }

    public static Tensor conv_transpose3d(Tensor input, Tensor weight) {
        return torch.conv_transpose3d(input, weight, new ConvTranspose3dFuncOptions());
    }

    public static Tensor conv_transpose3d(Tensor input, Tensor weight, ConvTranspose3dFuncOptions options) {
        return torch.conv_transpose3d(input, weight, options);
    }

    // ========================================================================
    // POOLING
    // ========================================================================

    /** Average pooling 1D. */
    public static Tensor avg_pool1d(Tensor input, long[] kernelSize, long[] stride, long[] padding,
                                    boolean ceilMode, boolean countIncludePad) {
        return torch.avg_pool1d(input, kernelSize, stride, padding, ceilMode, countIncludePad);
    }

    public static Tensor avg_pool1d(Tensor input, long kernelSize, long stride, long padding) {
        return avg_pool1d(input, new long[]{kernelSize}, new long[]{stride}, new long[]{padding}, false, true);
    }

    /** Average pooling 2D. */
    public static Tensor avg_pool2d(Tensor input, long[] kernelSize, long[] stride, long[] padding,
                                    boolean ceilMode, boolean countIncludePad, long divisorOverride) {
        return torch.avg_pool2d(input, kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride);
    }

    public static Tensor avg_pool2d(Tensor input, long[] kernelSize, long[] stride, long[] padding) {
        return avg_pool2d(input, kernelSize, stride, padding, false, true, 0);
    }

    public static Tensor avg_pool2d(Tensor input, long kernelSize, long stride, long padding) {
        return avg_pool2d(input, new long[]{kernelSize, kernelSize}, new long[]{stride, stride}, new long[]{padding, padding});
    }

    /** Average pooling 3D. */
    public static Tensor avg_pool3d(Tensor input, long[] kernelSize, long[] stride, long[] padding,
                                    boolean ceilMode, boolean countIncludePad, long divisorOverride) {
        return torch.avg_pool3d(input, kernelSize, stride, padding, ceilMode, countIncludePad, divisorOverride);
    }

    public static Tensor avg_pool3d(Tensor input, long[] kernelSize, long[] stride, long[] padding) {
        return avg_pool3d(input, kernelSize, stride, padding, false, true, 0);
    }

    public static Tensor avg_pool3d(Tensor input, long kernelSize, long stride, long padding) {
        return avg_pool3d(input, new long[]{kernelSize, kernelSize, kernelSize},
                new long[]{stride, stride, stride}, new long[]{padding, padding, padding});
    }

    /** Max pooling 1D. */
    public static Tensor max_pool1d(Tensor input, long[] kernelSize, long[] stride, long[] padding,
                                    long[] dilation, boolean ceilMode, boolean returnIndices) {
        return torch.max_pool1d(input, kernelSize, stride, padding, dilation, ceilMode, returnIndices);
    }

    public static Tensor max_pool1d(Tensor input, long kernelSize, long stride, long padding) {
        return max_pool1d(input, new long[]{kernelSize}, new long[]{stride}, new long[]{padding},
                new long[]{1}, false, false);
    }

    /** Max pooling 2D. */
    public static Tensor max_pool2d(Tensor input, long[] kernelSize, long[] stride, long[] padding,
                                    long[] dilation, boolean ceilMode, boolean returnIndices) {
        return torch.max_pool2d(input, kernelSize, stride, padding, dilation, ceilMode, returnIndices);
    }

    public static Tensor max_pool2d(Tensor input, long[] kernelSize, long[] stride, long[] padding) {
        return max_pool2d(input, kernelSize, stride, padding, new long[]{1, 1}, false, false);
    }

    public static Tensor max_pool2d(Tensor input, long kernelSize, long stride, long padding) {
        return max_pool2d(input, new long[]{kernelSize, kernelSize}, new long[]{stride, stride}, new long[]{padding, padding});
    }

    /** Max pooling 3D. */
    public static Tensor max_pool3d(Tensor input, long[] kernelSize, long[] stride, long[] padding,
                                    long[] dilation, boolean ceilMode, boolean returnIndices) {
        return torch.max_pool3d(input, kernelSize, stride, padding, dilation, ceilMode, returnIndices);
    }

    public static Tensor max_pool3d(Tensor input, long[] kernelSize, long[] stride, long[] padding) {
        return max_pool3d(input, kernelSize, stride, padding, new long[]{1, 1, 1}, false, false);
    }

    public static Tensor max_pool3d(Tensor input, long kernelSize, long stride, long padding) {
        return max_pool3d(input, new long[]{kernelSize, kernelSize, kernelSize},
                new long[]{stride, stride, stride}, new long[]{padding, padding, padding});
    }

    /** Adaptive average pooling 1D. */
    public static Tensor adaptive_avg_pool1d(Tensor input, long[] outputSize) {
        return torch.adaptive_avg_pool1d(input, outputSize);
    }

    public static Tensor adaptive_avg_pool1d(Tensor input, long outputSize) {
        return adaptive_avg_pool1d(input, new long[]{outputSize});
    }

    /** Adaptive average pooling 2D. */
    public static Tensor adaptive_avg_pool2d(Tensor input, long[] outputSize) {
        return torch.adaptive_avg_pool2d(input, outputSize);
    }

    public static Tensor adaptive_avg_pool2d(Tensor input, long outputH, long outputW) {
        return adaptive_avg_pool2d(input, new long[]{outputH, outputW});
    }

    /** Adaptive average pooling 3D. */
    public static Tensor adaptive_avg_pool3d(Tensor input, long[] outputSize) {
        return torch.adaptive_avg_pool3d(input, outputSize);
    }

    public static Tensor adaptive_avg_pool3d(Tensor input, long outputD, long outputH, long outputW) {
        return adaptive_avg_pool3d(input, new long[]{outputD, outputH, outputW});
    }

    /** Adaptive max pooling 1D. */
    public static Tensor adaptive_max_pool1d(Tensor input, long[] outputSize, boolean returnIndices) {
        return torch.adaptive_max_pool1d(input, outputSize, returnIndices);
    }

    public static Tensor adaptive_max_pool1d(Tensor input, long outputSize) {
        return adaptive_max_pool1d(input, new long[]{outputSize}, false);
    }

    /** Adaptive max pooling 2D. */
    public static Tensor adaptive_max_pool2d(Tensor input, long[] outputSize, boolean returnIndices) {
        return torch.adaptive_max_pool2d(input, outputSize, returnIndices);
    }

    public static Tensor adaptive_max_pool2d(Tensor input, long outputH, long outputW) {
        return adaptive_max_pool2d(input, new long[]{outputH, outputW}, false);
    }

    /** Adaptive max pooling 3D. */
    public static Tensor adaptive_max_pool3d(Tensor input, long[] outputSize, boolean returnIndices) {
        return torch.adaptive_max_pool3d(input, outputSize, returnIndices);
    }

    public static Tensor adaptive_max_pool3d(Tensor input, long outputD, long outputH, long outputW) {
        return adaptive_max_pool3d(input, new long[]{outputD, outputH, outputW}, false);
    }

    /** Fractional max pooling 2D. */
    public static Tensor fractional_max_pool2d(Tensor input, long[] kernelSize, long[] outputSize,
                                                Tensor returnIndices0) {
        return torch.fractional_max_pool2d(input, kernelSize, outputSize, returnIndices0);
    }

    public static Tensor fractional_max_pool2d(Tensor input, long kernelSize, long outputSize) {
        return fractional_max_pool2d(input, new long[]{kernelSize, kernelSize},
                new long[]{outputSize, outputSize}, null);
    }

    /** Fractional max pooling 3D. */
    public static Tensor fractional_max_pool3d(Tensor input, long[] kernelSize, long[] outputSize,
                                                Tensor returnIndices0) {
        return torch.fractional_max_pool3d(input, kernelSize, outputSize, returnIndices0);
    }

    /** Lp pooling 1D. */
    public static Tensor lp_pool1d(Tensor input, double normType, long kernelSize, long stride, boolean ceilMode) {
        return torch.lp_pool1d(input, normType, kernelSize, stride, ceilMode);
    }

    /** Lp pooling 2D. */
    public static Tensor lp_pool2d(Tensor input, double normType, long kernelSize, long stride, boolean ceilMode) {
        return torch.lp_pool2d(input, normType, kernelSize, stride, ceilMode);
    }

    /** Lp pooling 3D. */
    public static Tensor lp_pool3d(Tensor input, double normType, long kernelSize, long stride, boolean ceilMode) {
        return torch.lp_pool3d(input, normType, kernelSize, stride, ceilMode);
    }

    /** Max unpool 1D. */
    public static Tensor max_unpool1d(Tensor input, Tensor indices, long[] outputSize) {
        MaxUnpool1dFuncOptions opt = new MaxUnpool1dFuncOptions().output_size(new LongVector(outputSize));
        return torch.max_unpool1d(input, indices, opt);
    }

    public static Tensor max_unpool1d(Tensor input, Tensor indices, long outputSize) {
        return max_unpool1d(input, indices, new long[]{outputSize});
    }

    /** Max unpool 2D. */
    public static Tensor max_unpool2d(Tensor input, Tensor indices, long[] outputSize) {
        MaxUnpool2dFuncOptions opt = new MaxUnpool2dFuncOptions().output_size(new LongVector(outputSize));
        return torch.max_unpool2d(input, indices, opt);
    }

    public static Tensor max_unpool2d(Tensor input, Tensor indices, long outputH, long outputW) {
        return max_unpool2d(input, indices, new long[]{outputH, outputW});
    }

    /** Max unpool 3D. */
    public static Tensor max_unpool3d(Tensor input, Tensor indices, long[] outputSize) {
        MaxUnpool3dFuncOptions opt = new MaxUnpool3dFuncOptions().output_size(new LongVector(outputSize));
        return torch.max_unpool3d(input, indices, opt);
    }

    public static Tensor max_unpool3d(Tensor input, Tensor indices, long outputD, long outputH, long outputW) {
        return max_unpool3d(input, indices, new long[]{outputD, outputH, outputW});
    }

    // ========================================================================
    // PADDING
    // ========================================================================

    /** Reflection padding 1D. */
    public static Tensor reflection_pad1d(Tensor input, long[] padding) {
        return torch.reflection_pad1d(input, padding);
    }

    public static Tensor reflection_pad1d(Tensor input, long padding) {
        return reflection_pad1d(input, new long[]{padding});
    }

    /** Reflection padding 2D. */
    public static Tensor reflection_pad2d(Tensor input, long[] padding) {
        return torch.reflection_pad2d(input, padding);
    }

    public static Tensor reflection_pad2d(Tensor input, long left, long right, long top, long bottom) {
        return reflection_pad2d(input, new long[]{left, right, top, bottom});
    }

    /** Reflection padding 3D. */
    public static Tensor reflection_pad3d(Tensor input, long[] padding) {
        return torch.reflection_pad3d(input, padding);
    }

    public static Tensor reflection_pad3d(Tensor input, long left, long right, long top, long bottom, long front, long back) {
        return reflection_pad3d(input, new long[]{left, right, top, bottom, front, back});
    }

    /** Replication padding 1D. */
    public static Tensor replication_pad1d(Tensor input, long[] padding) {
        return torch.replication_pad1d(input, padding);
    }

    public static Tensor replication_pad1d(Tensor input, long padding) {
        return replication_pad1d(input, new long[]{padding});
    }

    /** Replication padding 2D. */
    public static Tensor replication_pad2d(Tensor input, long[] padding) {
        return torch.replication_pad2d(input, padding);
    }

    public static Tensor replication_pad2d(Tensor input, long left, long right, long top, long bottom) {
        return replication_pad2d(input, new long[]{left, right, top, bottom});
    }

    /** Replication padding 3D. */
    public static Tensor replication_pad3d(Tensor input, long[] padding) {
        return torch.replication_pad3d(input, padding);
    }

    public static Tensor replication_pad3d(Tensor input, long left, long right, long top, long bottom, long front, long back) {
        return replication_pad3d(input, new long[]{left, right, top, bottom, front, back});
    }

    /** Constant padding N-D. */
    public static Tensor pad(Tensor input, long[] pad, String mode, double value) {
        PadFuncOptions opt = new PadFuncOptions().mode(mode).value(new Scalar(value));
        return torch.constant_pad_nd(input, pad, opt);
    }

    public static Tensor pad(Tensor input, long[] pad, String mode) {
        return pad(input, pad, mode, 0.0);
    }

    public static Tensor constant_pad_nd(Tensor input, long[] pad, double value) {
        return torch.constant_pad_nd(input, pad, value);
    }

    public static Tensor constant_pad_nd(Tensor input, long[] pad) {
        return torch.constant_pad_nd(input, pad, 0.0);
    }

    // ========================================================================
    // ACTIVATIONS
    // ========================================================================

    /** ReLU. */
    public static Tensor relu(Tensor input, boolean inplace) {
        if (inplace) {
            return torch.relu_(input);
        }
        return torch.relu(input);
    }

    public static Tensor relu(Tensor input) {
        return torch.relu(input);
    }

    /** ReLU6. */
    public static Tensor relu6(Tensor input, boolean inplace) {
        ReLU6Options opt = new ReLU6Options().inplace(inplace);
        return torch.relu6(input, opt);
    }

    public static Tensor relu6(Tensor input) {
        return torch.relu6(input, new ReLU6Options());
    }

    /** LeakyReLU. */
    public static Tensor leaky_relu(Tensor input, double negativeSlope, boolean inplace) {
        LeakyReLUOptions opt = new LeakyReLUOptions().negative_slope(negativeSlope).inplace(inplace);
        return torch.leaky_relu(input, opt);
    }

    public static Tensor leaky_relu(Tensor input, double negativeSlope) {
        return leaky_relu(input, negativeSlope, false);
    }

    public static Tensor leaky_relu(Tensor input) {
        return torch.leaky_relu(input, new LeakyReLUOptions());
    }

    /** PReLU (per-channel learnable). */
    public static Tensor prelu(Tensor input, Tensor weight) {
        return torch.prelu(input, weight);
    }

    /** RReLU. */
    public static Tensor rrelu(Tensor input, double lower, double upper, boolean training, boolean inplace) {
        RReLUOptions opt = new RReLUOptions().lower(lower).upper(upper).training(training).inplace(inplace);
        return torch.rrelu(input, opt);
    }

    public static Tensor rrelu(Tensor input, double lower, double upper, boolean training) {
        return rrelu(input, lower, upper, training, false);
    }

    public static Tensor rrelu(Tensor input) {
        return torch.rrelu(input, new RReLUOptions());
    }

    /** ELU. */
    public static Tensor elu(Tensor input, double alpha, boolean inplace) {
        ELUOptions opt = new ELUOptions().alpha(alpha).inplace(inplace);
        return torch.elu(input, opt);
    }

    public static Tensor elu(Tensor input, double alpha) {
        return elu(input, alpha, false);
    }

    public static Tensor elu(Tensor input) {
        return torch.elu(input, new ELUOptions());
    }

    /** CELU. */
    public static Tensor celu(Tensor input, double alpha, boolean inplace) {
        org.bytedeco.pytorch.nn.options.CELUOptions opt = new org.bytedeco.pytorch.nn.options.CELUOptions().alpha(alpha).inplace(inplace);
        return torch.celu(input, opt);
    }

    public static Tensor celu(Tensor input, double alpha) {
        return celu(input, alpha, false);
    }

    public static Tensor celu(Tensor input) {
        return torch.celu(input, new org.bytedeco.pytorch.nn.options.CELUOptions());
    }

    /** SELU. */
    public static Tensor selu(Tensor input, boolean inplace) {
        SELUOptions opt = new SELUOptions().inplace(inplace);
        return torch.selu(input, opt);
    }

    public static Tensor selu(Tensor input) {
        return torch.selu(input, new SELUOptions());
    }

    /** SiLU (Swish). */
    public static Tensor silu(Tensor input, boolean inplace) {
        if (inplace) {
            return torch.silu_(input);
        }
        return torch.silu(input);
    }

    public static Tensor silu(Tensor input) {
        return torch.silu(input);
    }

    /** Mish. */
    public static Tensor mish(Tensor input, boolean inplace) {
        if (inplace) {
            return torch.mish_(input);
        }
        return torch.mish(input);
    }

    public static Tensor mish(Tensor input) {
        return torch.mish(input);
    }

    /** GELU. */
    public static Tensor gelu(Tensor input, String approximate) {
        GELUOptions opt = new GELUOptions().approximate(approximate);
        return torch.gelu(input, opt);
    }

    public static Tensor gelu(Tensor input) {
        return torch.gelu(input, new GELUOptions());
    }

    /** Hardtanh. */
    public static Tensor hardtanh(Tensor input, double minVal, double maxVal, boolean inplace) {
        HardtanhOptions opt = new HardtanhOptions().min_val(minVal).max_val(maxVal).inplace(inplace);
        return torch.hardtanh(input, opt);
    }

    public static Tensor hardtanh(Tensor input, double minVal, double maxVal) {
        return hardtanh(input, minVal, maxVal, false);
    }

    public static Tensor hardtanh(Tensor input) {
        return torch.hardtanh(input, new HardtanhOptions());
    }

    /** Hardshrink. */
    public static Tensor hardshrink(Tensor input, double lambda) {
        HardshrinkOptions opt = new HardshrinkOptions().lambda(lambda);
        return torch.hardshrink(input, opt);
    }

    public static Tensor hardshrink(Tensor input) {
        return torch.hardshrink(input, new HardshrinkOptions());
    }

    /** Softshrink. */
    public static Tensor softshrink(Tensor input, double lambda) {
        SoftshrinkOptions opt = new SoftshrinkOptions().lambda(lambda);
        return torch.softshrink(input, opt);
    }

    public static Tensor softshrink(Tensor input) {
        return torch.softshrink(input, new SoftshrinkOptions());
    }

    /** Softplus. */
    public static Tensor softplus(Tensor input, double beta, double threshold) {
        SoftplusOptions opt = new SoftplusOptions().beta(beta).threshold(threshold);
        return torch.softplus(input, opt);
    }

    public static Tensor softplus(Tensor input) {
        return torch.softplus(input, new SoftplusOptions());
    }

    /** Softsign. */
    public static Tensor softsign(Tensor input) {
        return torch.softsign(input);
    }

    /** Tanhshrink. */
    public static Tensor tanhshrink(Tensor input) {
        return torch.tanhshrink(input);
    }

    /** Threshold. */
    public static Tensor threshold(Tensor input, double threshold, double value) {
        ThresholdOptions opt = new ThresholdOptions().threshold(threshold).value(value);
        return torch.threshold(input, opt);
    }

    public static Tensor threshold(Tensor input, double threshold, double value, boolean inplace) {
        return threshold(input, threshold, value).clone(); // inplace via clone
    }

    /** GLU (Gated Linear Unit). */
    public static Tensor glu(Tensor input, long dim) {
        GLUOptions opt = new GLUOptions().dim(dim);
        return torch.glu(input, opt);
    }

    public static Tensor glu(Tensor input) {
        return torch.glu(input, new GLUOptions());
    }

    /** LogSigmoid. */
    public static Tensor logsigmoid(Tensor input) {
        return torch.logsigmoid(input);
    }

    /** Softmax. */
    public static Tensor softmax(Tensor input, long dim, org.bytedeco.pytorch.global.torch.ScalarType dtype) {
        SoftmaxFuncOptions opt = new SoftmaxFuncOptions().dim(dim).dtype(dtype);
        return torch.softmax(input, opt);
    }

    public static Tensor softmax(Tensor input, long dim) {
        return softmax(input, dim, (org.bytedeco.pytorch.global.torch.ScalarType) null);
    }

    /** Softmin. */
    public static Tensor softmin(Tensor input, long dim, org.bytedeco.pytorch.global.torch.ScalarType dtype) {
        SoftminFuncOptions opt = new SoftminFuncOptions().dim(dim).dtype(dtype);
        return torch.softmin(input, opt);
    }

    public static Tensor softmin(Tensor input, long dim) {
        return softmin(input, dim, (org.bytedeco.pytorch.global.torch.ScalarType) null);
    }

    /** LogSoftmax. */
    public static Tensor log_softmax(Tensor input, long dim, org.bytedeco.pytorch.global.torch.ScalarType dtype) {
        LogSoftmaxFuncOptions opt = new LogSoftmaxFuncOptions().dim(dim).dtype(dtype);
        return torch.log_softmax(input, opt);
    }

    public static Tensor log_softmax(Tensor input, long dim) {
        return log_softmax(input, dim, (org.bytedeco.pytorch.global.torch.ScalarType) null);
    }

    /** Gumbel softmax. */
    public static Tensor gumbel_softmax(Tensor logits, double tau, boolean hard, long dim) {
        GumbelSoftmaxFuncOptions opt = new GumbelSoftmaxFuncOptions().tau(tau).hard(hard).dim(dim);
        return torch.gumbel_softmax(logits, opt);
    }

    public static Tensor gumbel_softmax(Tensor logits, double tau, boolean hard) {
        return gumbel_softmax(logits, tau, hard, -1);
    }

    /** Tanh. */
    public static Tensor tanh(Tensor input) {
        return torch.tanh(input);
    }

    /** Sigmoid. */
    public static Tensor sigmoid(Tensor input) {
        return torch.sigmoid(input);
    }

    // ========================================================================
    // NORMALIZATION
    // ========================================================================

    /** Batch normalization. */
    public static Tensor batch_norm(Tensor input, Tensor runningMean, Tensor runningVar,
                                    boolean training, Tensor weight, Tensor bias,
                                    double momentum, double eps) {
        BatchNormFuncOptions opt = new BatchNormFuncOptions()
                .training(training)
                .momentum(new DoubleOptional(momentum))
                .eps(eps);
        if (runningMean != null) opt.running_mean(runningMean);
        if (runningVar != null) opt.running_var(runningVar);
        if (weight != null) opt.weight(weight);
        if (bias != null) opt.bias(bias);
        return torch.batch_norm(input, opt);
    }

    public static Tensor batch_norm(Tensor input, Tensor runningMean, Tensor runningVar,
                                    boolean training, double momentum, double eps) {
        return batch_norm(input, runningMean, runningVar, training, null, null, momentum, eps);
    }

    public static Tensor batch_norm(Tensor input, Tensor runningMean, Tensor runningVar, boolean training) {
        return batch_norm(input, runningMean, runningVar, training, 0.1, 1e-5);
    }

    /** Instance normalization. */
    public static Tensor instance_norm(Tensor input, Tensor runningMean, Tensor runningVar,
                                       boolean useInputStats, Tensor weight, Tensor bias,
                                       boolean training, double momentum, double eps) {
        InstanceNormFuncOptions opt = new InstanceNormFuncOptions()
                .use_input_stats(useInputStats)
                .training(training)
                .momentum(new DoubleOptional(momentum))
                .eps(eps);
        if (runningMean != null) opt.running_mean(runningMean);
        if (runningVar != null) opt.running_var(runningVar);
        if (weight != null) opt.weight(weight);
        if (bias != null) opt.bias(bias);
        return torch.instance_norm(input, opt);
    }

    public static Tensor instance_norm(Tensor input, Tensor runningMean, Tensor runningVar,
                                       boolean useInputStats, double momentum, double eps) {
        return instance_norm(input, runningMean, runningVar, useInputStats, null, null, true, momentum, eps);
    }

    /** Layer normalization. */
    public static Tensor layer_norm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias, double eps) {
        return torch.layer_norm(input, new LongVector(normalizedShape), weight, bias, eps);
    }

    public static Tensor layer_norm(Tensor input, long[] normalizedShape, double eps) {
        return layer_norm(input, normalizedShape, null, null, eps);
    }

    public static Tensor layer_norm(Tensor input, long[] normalizedShape) {
        return layer_norm(input, normalizedShape, null, null, 1e-5);
    }

    /** Group normalization. */
    public static Tensor group_norm(Tensor input, long numGroups, Tensor weight, Tensor bias, double eps) {
        return torch.group_norm(input, numGroups, weight, bias, eps);
    }

    public static Tensor group_norm(Tensor input, long numGroups, double eps) {
        return group_norm(input, numGroups, null, null, eps);
    }

    public static Tensor group_norm(Tensor input, long numGroups) {
        return group_norm(input, numGroups, 1e-5);
    }

    /** Local response normalization. */
    public static Tensor local_response_norm(Tensor input, long size, double alpha, double beta, double k) {
        LocalResponseNormOptions opt = new LocalResponseNormOptions().size(size).alpha(alpha).beta(beta).k(k);
        return torch.local_response_norm(input, opt);
    }

    public static Tensor local_response_norm(Tensor input, long size) {
        return local_response_norm(input, size, 1.0, 0.75, 2.0);
    }

    /** Normalize a vector along a dimension with L_p norm. */
    public static Tensor normalize(Tensor input, double p, long dim, double eps) {
        NormalizeFuncOptions opt = new NormalizeFuncOptions().p(new Scalar(p)).dim(dim).eps(eps);
        return torch.normalize(input, opt);
    }

    public static Tensor normalize(Tensor input, double p, long dim) {
        return normalize(input, p, dim, 1e-12);
    }

    public static Tensor normalize(Tensor input, double p) {
        return normalize(input, p, 1);
    }

    // ========================================================================
    // LINEAR
    // ========================================================================

    /** Linear: y = x @ weight.T + bias. */
    public static Tensor linear(Tensor input, Tensor weight, Tensor bias) {
        return torch.linear(input, weight, bias);
    }

    public static Tensor linear(Tensor input, Tensor weight) {
        return torch.linear(input, weight);
    }

    /** Bilinear. */
    public static Tensor bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor bias) {
        return torch.bilinear(input1, input2, weight, bias);
    }

    // ========================================================================
    // DROPOUT
    // ========================================================================

    /** Dropout. */
    public static Tensor dropout(Tensor input, double p, boolean training) {
        return torch.dropout(input, p, training);
    }

    public static Tensor dropout(Tensor input, double p) {
        return dropout(input, p, true);
    }

    public static Tensor dropout(Tensor input) {
        return dropout(input, 0.5, true);
    }

    public static Tensor dropout(Tensor input, double p, boolean training, boolean inplace) {
        if (inplace) {
            return torch.dropout_(input, p, training);
        }
        return torch.dropout(input, p, training);
    }

    /** Alpha dropout. */
    public static Tensor alpha_dropout(Tensor input, double p, boolean training) {
        return torch.alpha_dropout(input, p, training);
    }

    public static Tensor alpha_dropout(Tensor input, double p) {
        return alpha_dropout(input, p, true);
    }

    public static Tensor alpha_dropout(Tensor input) {
        return alpha_dropout(input, 0.5, true);
    }

    /** Feature alpha dropout. */
    public static Tensor feature_alpha_dropout(Tensor input, double p, boolean training) {
        return torch.feature_alpha_dropout(input, p, training);
    }

    public static Tensor feature_alpha_dropout(Tensor input, double p) {
        return feature_alpha_dropout(input, p, true);
    }

    public static Tensor feature_alpha_dropout(Tensor input) {
        return feature_alpha_dropout(input, 0.5, true);
    }

    // ========================================================================
    // LOSS FUNCTIONS
    // ========================================================================

    /** Binary cross entropy. */
    public static Tensor binary_cross_entropy(Tensor input, Tensor target, Tensor weight, boolean reduction) {
        BCELossOptions opt = new BCELossOptions().weight(weight).reduction((byte) (reduction ? 1 : 0));
        return torch.binary_cross_entropy(input, target, opt);
    }

    public static Tensor binary_cross_entropy(Tensor input, Tensor target, Tensor weight, long reduction) {
        return binary_cross_entropy(input, target, weight, reduction != 0);
    }

    public static Tensor binary_cross_entropy(Tensor input, Tensor target, Tensor weight) {
        return binary_cross_entropy(input, target, weight, true);
    }

    public static Tensor binary_cross_entropy(Tensor input, Tensor target) {
        return binary_cross_entropy(input, target, null, true);
    }

    /** Binary cross entropy with logits. */
    public static Tensor binary_cross_entropy_with_logits(Tensor input, Tensor target, Tensor weight,
                                                          Tensor posWeight, boolean reduction) {
        BCEWithLogitsLossOptions opt = new BCEWithLogitsLossOptions().weight(weight).pos_weight(posWeight)
                .reduction((byte) (reduction ? 1 : 0));
        return torch.binary_cross_entropy_with_logits(input, target, opt);
    }

    public static Tensor binary_cross_entropy_with_logits(Tensor input, Tensor target) {
        return binary_cross_entropy_with_logits(input, target, null, null, true);
    }

    /** L1 loss (Mean Absolute Error). */
    public static Tensor l1_loss(Tensor input, Tensor target, long reduction) {
        L1LossOptions opt = new L1LossOptions().reduction(reduction);
        return torch.l1_loss(input, target, opt);
    }

    public static Tensor l1_loss(Tensor input, Tensor target) {
        return l1_loss(input, target, 1);
    }

    public static Tensor l1_loss(Tensor input, Tensor target, boolean reduction) {
        return l1_loss(input, target, reduction ? 1 : 0);
    }

    /** MSE loss. */
    public static Tensor mse_loss(Tensor input, Tensor target, long reduction) {
        MSELossOptions opt = new MSELossOptions().reduction(reduction);
        return torch.mse_loss(input, target, opt);
    }

    public static Tensor mse_loss(Tensor input, Tensor target) {
        return mse_loss(input, target, 1);
    }

    public static Tensor mse_loss(Tensor input, Tensor target, boolean reduction) {
        return mse_loss(input, target, reduction ? 1 : 0);
    }

    /** Smooth L1 loss (Huber). */
    public static Tensor smooth_l1_loss(Tensor input, Tensor target, long reduction, double beta) {
        SmoothL1LossOptions opt = new SmoothL1LossOptions().reduction(reduction).beta(beta);
        return torch._smooth_l1_loss(input, target, opt);
    }

    public static Tensor smooth_l1_loss(Tensor input, Tensor target, long reduction) {
        return smooth_l1_loss(input, target, reduction, 1.0);
    }

    public static Tensor smooth_l1_loss(Tensor input, Tensor target) {
        return smooth_l1_loss(input, target, 1);
    }

    /** Huber loss. */
    public static Tensor huber_loss(Tensor input, Tensor target, long reduction, double delta) {
        org.bytedeco.pytorch.nn.options.HuberLossOptions opt = new org.bytedeco.pytorch.nn.options.HuberLossOptions()
                .reduction(reduction).delta(delta);
        return torch.huber_loss(input, target, opt);
    }

    public static Tensor huber_loss(Tensor input, Tensor target, long reduction) {
        return huber_loss(input, target, reduction, 1.0);
    }

    public static Tensor huber_loss(Tensor input, Tensor target) {
        return huber_loss(input, target, 1);
    }

    /** NLL loss. */
    public static Tensor nll_loss(Tensor input, Tensor target, Tensor weight, long ignoreIndex,
                                  long reduction) {
        NLLLossOptions opt = new NLLLossOptions().weight(weight).ignore_index(ignoreIndex).reduction(reduction);
        return torch.nll_loss(input, target, opt);
    }

    public static Tensor nll_loss(Tensor input, Tensor target, Tensor weight, long ignoreIndex) {
        return nll_loss(input, target, weight, ignoreIndex, 1);
    }

    public static Tensor nll_loss(Tensor input, Tensor target, Tensor weight) {
        return nll_loss(input, target, weight, -100, 1);
    }

    public static Tensor nll_loss(Tensor input, Tensor target) {
        return nll_loss(input, target, null, -100, 1);
    }

    /** Cross entropy loss. */
    public static Tensor cross_entropy(Tensor input, Tensor target, Tensor weight, long ignoreIndex,
                                       long reduction) {
        CrossEntropyLossOptions opt = new CrossEntropyLossOptions().weight(weight).ignore_index(ignoreIndex)
                .reduction(reduction);
        return torch.cross_entropy(input, target, opt);
    }

    public static Tensor cross_entropy(Tensor input, Tensor target, Tensor weight, long ignoreIndex) {
        return cross_entropy(input, target, weight, ignoreIndex, 1);
    }

    public static Tensor cross_entropy(Tensor input, Tensor target, Tensor weight) {
        return cross_entropy(input, target, weight, -100, 1);
    }

    public static Tensor cross_entropy(Tensor input, Tensor target) {
        return cross_entropy(input, target, null, -100, 1);
    }

    /** CTC loss. */
    public static Tensor ctc_loss(Tensor logProbs, Tensor targets, Tensor inputLengths, Tensor targetLengths,
                                  long blank, long reduction) {
        CTCLossOptions opt = new CTCLossOptions().blank(blank).reduction(reduction);
        return torch.ctc_loss(logProbs, targets, inputLengths, targetLengths, opt);
    }

    public static Tensor ctc_loss(Tensor logProbs, Tensor targets, Tensor inputLengths, Tensor targetLengths,
                                  long blank) {
        return ctc_loss(logProbs, targets, inputLengths, targetLengths, blank, 1);
    }

    /** Poisson NLL loss. */
    public static Tensor poisson_nll_loss(Tensor input, Tensor target, Tensor logInput, bool fullLoss,
                                          long reduction) {
        PoissonNLLLossOptions opt = new PoissonNLLLossOptions().log_input(logInput).full(fullLoss).reduction(reduction);
        return torch.poisson_nll_loss(input, target, opt);
    }

    public static Tensor poisson_nll_loss(Tensor input, Tensor target, boolean logInput, boolean full,
                                          long reduction) {
        return poisson_nll_loss(input, target, logInput ? torch.tensor(new int[]{1}) : null,
                full, reduction);
    }

    public static Tensor poisson_nll_loss(Tensor input, Tensor target, boolean logInput, boolean full) {
        return poisson_nll_loss(input, target, logInput, full, 1);
    }

    public static Tensor poisson_nll_loss(Tensor input, Tensor target) {
        return poisson_nll_loss(input, target, false, false, 1);
    }

    /** KL divergence loss. */
    public static Tensor kl_div(Tensor input, Tensor target, long reduction, boolean logTarget) {
        KLDivLossOptions opt = new KLDivLossOptions().reduction(reduction).log_target(logTarget);
        return torch.kl_div(input, target, opt);
    }

    public static Tensor kl_div(Tensor input, Tensor target, long reduction) {
        return kl_div(input, target, reduction, false);
    }

    public static Tensor kl_div(Tensor input, Tensor target) {
        return kl_div(input, target, 1);
    }

    /** Margin ranking loss. */
    public static Tensor margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, double margin,
                                             long reduction) {
        MarginRankingLossOptions opt = new MarginRankingLossOptions().margin(margin).reduction(reduction);
        return torch.margin_ranking_loss(input1, input2, target, opt);
    }

    public static Tensor margin_ranking_loss(Tensor input1, Tensor input2, Tensor target) {
        return margin_ranking_loss(input1, input2, target, 0.0, 1);
    }

    /** Hinge embedding loss. */
    public static Tensor hinge_embedding_loss(Tensor input, Tensor target, double margin, long reduction) {
        HingeEmbeddingLossOptions opt = new HingeEmbeddingLossOptions().margin(margin).reduction(reduction);
        return torch.hinge_embedding_loss(input, target, opt);
    }

    public static Tensor hinge_embedding_loss(Tensor input, Tensor target) {
        return hinge_embedding_loss(input, target, 1.0, 1);
    }

    /** Multi-margin loss. */
    public static Tensor multi_margin_loss(Tensor input, Tensor target, long p, double margin, Tensor weight,
                                           long reduction) {
        MultiMarginLossOptions opt = new MultiMarginLossOptions().p(p).margin(margin).weight(weight).reduction(reduction);
        return torch.multi_margin_loss(input, target, opt);
    }

    public static Tensor multi_margin_loss(Tensor input, Tensor target, long p, double margin) {
        return multi_margin_loss(input, target, p, margin, null, 1);
    }

    /** Cosine embedding loss. */
    public static Tensor cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, double margin,
                                               long reduction) {
        CosineEmbeddingLossOptions opt = new CosineEmbeddingLossOptions().margin(margin).reduction(reduction);
        return torch.cosine_embedding_loss(input1, input2, target, opt);
    }

    public static Tensor cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, double margin) {
        return cosine_embedding_loss(input1, input2, target, margin, 1);
    }

    public static Tensor cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target) {
        return cosine_embedding_loss(input1, input2, target, 0.0);
    }

    /** Multi-label margin loss. */
    public static Tensor multilabel_margin_loss(Tensor input, Tensor target, long reduction) {
        MultiLabelMarginLossOptions opt = new MultiLabelMarginLossOptions().reduction(reduction);
        return torch.multilabel_margin_loss(input, target, opt);
    }

    public static Tensor multilabel_margin_loss(Tensor input, Tensor target) {
        return multilabel_margin_loss(input, target, 1);
    }

    /** Soft margin loss. */
    public static Tensor soft_margin_loss(Tensor input, Tensor target, long reduction) {
        SoftMarginLossOptions opt = new SoftMarginLossOptions().reduction(reduction);
        return torch.soft_margin_loss(input, target, opt);
    }

    public static Tensor soft_margin_loss(Tensor input, Tensor target) {
        return soft_margin_loss(input, target, 1);
    }

    /** Multi-label soft margin loss. */
    public static Tensor multilabel_soft_margin_loss(Tensor input, Tensor target, Tensor weight, long reduction) {
        MultiLabelSoftMarginLossOptions opt = new MultiLabelSoftMarginLossOptions().weight(weight).reduction(reduction);
        return torch.multilabel_soft_margin_loss(input, target, opt);
    }

    public static Tensor multilabel_soft_margin_loss(Tensor input, Tensor target, Tensor weight) {
        return multilabel_soft_margin_loss(input, target, weight, 1);
    }

    public static Tensor multilabel_soft_margin_loss(Tensor input, Tensor target) {
        return multilabel_soft_margin_loss(input, target, null, 1);
    }

    /** Triplet margin loss. */
    public static Tensor triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, double margin,
                                             long p, double eps, boolean swap, long reduction) {
        TripletMarginLossOptions opt = new TripletMarginLossOptions().margin(margin).p(p).eps(eps).swap(swap)
                .reduction(reduction);
        return torch.triplet_margin_loss(anchor, positive, negative, opt);
    }

    public static Tensor triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, double margin) {
        return triplet_margin_loss(anchor, positive, negative, margin, 2, 1e-6, false, 1);
    }

    /** Triplet margin with distance loss. */
    public static Tensor triplet_margin_with_distance_loss(Tensor anchor, Tensor positive, Tensor negative,
                                                          double margin) {
        TripletMarginWithDistanceLossOptions opt = new TripletMarginWithDistanceLossOptions().margin(margin);
        return torch.triplet_margin_with_distance_loss(anchor, positive, negative, opt);
    }

    // ========================================================================
    // DISTANCE
    // ========================================================================

    /** Cosine similarity. */
    public static Tensor cosine_similarity(Tensor x1, Tensor x2, long dim, double eps) {
        CosineSimilarityOptions opt = new CosineSimilarityOptions().dim(dim).eps(eps);
        return torch.cosine_similarity(x1, x2, opt);
    }

    public static Tensor cosine_similarity(Tensor x1, Tensor x2, long dim) {
        return cosine_similarity(x1, x2, dim, 1e-8);
    }

    public static Tensor cosine_similarity(Tensor x1, Tensor x2) {
        return cosine_similarity(x1, x2, 1);
    }

    /** Pairwise distance. */
    public static Tensor pairwise_distance(Tensor x1, Tensor x2, double p, double eps, boolean keepdim) {
        PairwiseDistanceOptions opt = new PairwiseDistanceOptions().p(p).eps(eps).keepdim(keepdim);
        return torch.pairwise_distance(x1, x2, opt);
    }

    public static Tensor pairwise_distance(Tensor x1, Tensor x2, double p, double eps) {
        return pairwise_distance(x1, x2, p, eps, false);
    }

    public static Tensor pairwise_distance(Tensor x1, Tensor x2) {
        return pairwise_distance(x1, x2, 2.0, 1e-6);
    }

    // ========================================================================
    // VISION
    // ========================================================================

    /** Pixel shuffle. */
    public static Tensor pixel_shuffle(Tensor input, long upscaleFactor) {
        return torch.pixel_shuffle(input, upscaleFactor);
    }

    /** Pixel unshuffle. */
    public static Tensor pixel_unshuffle(Tensor input, long downscaleFactor) {
        return torch.pixel_unshuffle(input, downscaleFactor);
    }

    /** Interpolate. */
    public static Tensor interpolate(Tensor input, long[] size, String mode, boolean alignCorners) {
        return torch.interpolate(input, size, new LongVector(size), mode, alignCorners);
    }

    public static Tensor interpolate(Tensor input, long[] size, String mode) {
        return interpolate(input, size, mode, false);
    }

    public static Tensor interpolate(Tensor input, long sizeH, long sizeW, String mode, boolean alignCorners) {
        return interpolate(input, new long[]{sizeH, sizeW}, mode, alignCorners);
    }

    public static Tensor interpolate(Tensor input, long sizeH, long sizeW, String mode) {
        return interpolate(input, new long[]{sizeH, sizeW}, mode);
    }

    /** Upsample bilinear. */
    public static Tensor upsample_bilinear(Tensor input, long[] outputSize, boolean alignCorners) {
        return torch.upsample_bilinear2d(input, outputSize, alignCorners);
    }

    public static Tensor upsample_bilinear(Tensor input, long sizeH, long sizeW, boolean alignCorners) {
        return upsample_bilinear(input, new long[]{sizeH, sizeW}, alignCorners);
    }

    /** Upsample nearest 2D. */
    public static Tensor upsample_nearest(Tensor input, long[] outputSize) {
        return torch.upsample_nearest2d(input, outputSize);
    }

    public static Tensor upsample_nearest(Tensor input, long sizeH, long sizeW) {
        return upsample_nearest(input, new long[]{sizeH, sizeW});
    }

    /** Upsample nearest 1D. */
    public static Tensor upsample_nearest1d(Tensor input, long[] outputSize) {
        return torch.upsample_nearest1d(input, outputSize);
    }

    /** Upsample bilinear 2D (alias). */
    public static Tensor upsample_bilinear2d(Tensor input, long[] outputSize, boolean alignCorners) {
        return torch.upsample_bilinear2d(input, outputSize, alignCorners);
    }

    public static Tensor upsample_bilinear2d(Tensor input, long sizeH, long sizeW, boolean alignCorners) {
        return upsample_bilinear2d(input, new long[]{sizeH, sizeW}, alignCorners);
    }

    /** Upsample nearest 1D. */
    public static Tensor upsample_nearest1d(Tensor input, long size) {
        return torch.upsample_nearest1d(input, new long[]{size});
    }

    /** Upsample trilinear 3D. */
    public static Tensor upsample_trilinear3d(Tensor input, long[] outputSize, boolean alignCorners) {
        return torch.upsample_trilinear3d(input, outputSize, alignCorners);
    }

    public static Tensor upsample_trilinear3d(Tensor input, long d, long h, long w, boolean alignCorners) {
        return upsample_trilinear3d(input, new long[]{d, h, w}, alignCorners);
    }

    /** Upsample bicubic 2D. */
    public static Tensor upsample_bicubic2d(Tensor input, long[] outputSize, boolean alignCorners) {
        return torch.upsample_bicubic2d(input, outputSize, alignCorners);
    }

    public static Tensor upsample_bicubic2d(Tensor input, long sizeH, long sizeW, boolean alignCorners) {
        return upsample_bicubic2d(input, new long[]{sizeH, sizeW}, alignCorners);
    }

    /** Upsample linear 1D. */
    public static Tensor upsample_linear1d(Tensor input, long[] outputSize, boolean alignCorners) {
        return torch.upsample_linear1d(input, outputSize, alignCorners);
    }

    /** Upsample 3D (nearest). */
    public static Tensor upsample_nearest3d(Tensor input, long[] outputSize) {
        return torch.upsample_nearest3d(input, outputSize);
    }

    /** Affine grid. */
    public static Tensor affine_grid(Tensor theta, long[] size, boolean alignCorners) {
        return torch.affine_grid(theta, size, alignCorners);
    }

    public static Tensor affine_grid(Tensor theta, long[] size) {
        return affine_grid(theta, size, false);
    }

    /** Grid sample. */
    public static Tensor grid_sample(Tensor input, Tensor grid, String mode, String paddingMode, boolean alignCorners) {
        return torch.grid_sample(input, grid, mode, paddingMode, alignCorners);
    }

    public static Tensor grid_sample(Tensor input, Tensor grid, String mode, String paddingMode) {
        return grid_sample(input, grid, mode, paddingMode, false);
    }

    public static Tensor grid_sample(Tensor input, Tensor grid, String mode) {
        return grid_sample(input, grid, mode, "zeros");
    }

    // ========================================================================
    // SPARSE
    // ========================================================================

    /** Embedding lookup. */
    public static Tensor embedding(Tensor input, Tensor weight, long paddingIdx, boolean scaleGradByFreq,
                                    boolean sparse) {
        EmbeddingFuncOptions opt = new EmbeddingFuncOptions().padding_idx(paddingIdx)
                .scale_grad_by_freq(scaleGradByFreq).sparse(sparse);
        return torch.embedding(input, weight, opt);
    }

    public static Tensor embedding(Tensor input, Tensor weight, long paddingIdx) {
        return embedding(input, weight, paddingIdx, false, false);
    }

    public static Tensor embedding(Tensor input, Tensor weight) {
        return embedding(input, weight, -1);
    }

    /** Embedding bag. */
    public static Tensor embedding_bag(Tensor input, Tensor weight, Tensor offsets, boolean scaleGradByFreq,
                                       long mode, boolean sparse, Tensor perSampleWeights, boolean includeLastOffset,
                                       long paddingIdx) {
        EmbeddingBagFuncOptions opt = new EmbeddingBagFuncOptions().offsets(offsets)
                .scale_grad_by_freq(scaleGradByFreq).mode(mode).sparse(sparse)
                .per_sample_weights(perSampleWeights).include_last_offset(includeLastOffset)
                .padding_idx(paddingIdx);
        return torch.embedding_bag(input, weight, opt);
    }

    public static Tensor embedding_bag(Tensor input, Tensor weight, Tensor offsets, boolean scaleGradByFreq,
                                       long mode, boolean sparse, Tensor perSampleWeights) {
        return embedding_bag(input, weight, offsets, scaleGradByFreq, mode, sparse, perSampleWeights, false, -1);
    }

    public static Tensor embedding_bag(Tensor input, Tensor weight, Tensor offsets, boolean scaleGradByFreq,
                                       long mode) {
        return embedding_bag(input, weight, offsets, scaleGradByFreq, mode, false, null);
    }

    /** One-hot encoding. */
    public static Tensor one_hot(Tensor input, long numClasses) {
        return torch.one_hot(input, numClasses);
    }

    // ========================================================================
    // FOLD / UNFLATTEN / UNFOLD
    // ========================================================================

    /** Fold. */
    public static Tensor fold(Tensor input, long[] outputSize, long[] kernelSize,
                              long[] dilation, long[] padding, long[] stride) {
        FoldOptions opt = new FoldOptions()
                .output_size(new LongVector(outputSize))
                .kernel_size(new LongVector(kernelSize))
                .dilation(new LongVector(dilation))
                .padding(new LongVector(padding))
                .stride(new LongVector(stride));
        return torch.fold(input, opt);
    }

    public static Tensor fold(Tensor input, long[] outputSize, long[] kernelSize) {
        return fold(input, outputSize, kernelSize, new long[]{1, 1}, new long[]{0, 0}, new long[]{1, 1});
    }

    public static Tensor unfold(Tensor input, long dimension, long size, long step) {
        UnfoldOptions opt = new UnfoldOptions().dimension(dimension).size(size).step(step);
        return torch.unfold(input, opt);
    }

    // ========================================================================
    // MULTI-HEAD ATTENTION (functional)
    // ========================================================================

    /** Scaled dot-product attention. */
    public static Tensor scaled_dot_product_attention(Tensor query, Tensor key, Tensor value,
                                                       Tensor attnMask, double dropoutP, boolean isCausal) {
        return torch.scaled_dot_product_attention(query, key, value, attnMask, dropoutP, isCausal);
    }

    public static Tensor scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor attnMask) {
        return scaled_dot_product_attention(query, key, value, attnMask, 0.0, false);
    }

    public static Tensor scaled_dot_product_attention(Tensor query, Tensor key, Tensor value) {
        return scaled_dot_product_attention(query, key, value, null);
    }

    /** Multi-head attention forward (functional). */
    public static Tensor multi_head_attention_forward(Tensor query, Tensor key, Tensor value,
                                                       Tensor embedDimToCheck, long numHeads,
                                                       Tensor inProjWeight, Tensor inProjBias,
                                                       Tensor outProjWeight, Tensor outProjBias,
                                                       boolean needWeights, double attnMask,
                                                       boolean biasK, boolean addZeroAttn) {
        return torch.multi_head_attention_forward(query, key, value, embedDimToCheck, numHeads,
                inProjWeight, inProjBias, outProjWeight, outProjBias, needWeights, attnMask, biasK, addZeroAttn);
    }

    public static Tensor multi_head_attention_forward(Tensor query, Tensor key, Tensor value,
                                                       Tensor embedDimToCheck, long numHeads,
                                                       Tensor inProjWeight, Tensor inProjBias,
                                                       Tensor outProjWeight, Tensor outProjBias) {
        return multi_head_attention_forward(query, key, value, embedDimToCheck, numHeads,
                inProjWeight, inProjBias, outProjWeight, outProjBias, true, 0.0, false, false);
    }

    // ========================================================================
    // RECURRENT (functional)
    // ========================================================================

    /** RNN functional. */
    public static TensorList rnn(Tensor input, Tensor h0, Tensor[] params, boolean hasBiases,
                                  long numLayers, double dropout, boolean training, boolean bidirectional,
                                  boolean batchFirst) {
        RNNOptions opt = new RNNOptions().has_biases(hasBiases).num_layers(numLayers).dropout(dropout)
                .training(training).bidirectional(bidirectional).batch_first(batchFirst);
        TensorList wList = new TensorList(params.length);
        for (int i = 0; i < params.length; i++) wList.put(i, params[i]);
        return torch.rnn(input, h0, wList, opt);
    }

    /** LSTM functional. */
    public static TensorList lstm(Tensor input, Tensor[] hx, Tensor[] params, boolean hasBiases,
                                   long numLayers, double dropout, boolean training, boolean bidirectional,
                                   boolean batchFirst) {
        LSTMOptions opt = new LSTMOptions().has_biases(hasBiases).num_layers(numLayers).dropout(dropout)
                .training(training).bidirectional(bidirectional).batch_first(batchFirst);
        TensorList wList = new TensorList(params.length);
        for (int i = 0; i < params.length; i++) wList.put(i, params[i]);
        return torch.lstm(input, hx, wList, opt);
    }

    /** GRU functional. */
    public static TensorList gru(Tensor input, Tensor h0, Tensor[] params, boolean hasBiases,
                                  long numLayers, double dropout, boolean training, boolean bidirectional,
                                  boolean batchFirst) {
        GRUOptions opt = new GRUOptions().has_biases(hasBiases).num_layers(numLayers).dropout(dropout)
                .training(training).bidirectional(bidirectional).batch_first(batchFirst);
        TensorList wList = new TensorList(params.length);
        for (int i = 0; i < params.length; i++) wList.put(i, params[i]);
        return torch.gru(input, h0, wList, opt);
    }

    /** RNN cell functional. */
    public static Tensor rnn_cell(Tensor input, Tensor hx, Tensor wIh, Tensor wHh, Tensor bIh, Tensor bHh) {
        RNNCellOptions opt = new RNNCellOptions();
        if (bIh != null) opt.bias_ih(bIh);
        if (bHh != null) opt.bias_hh(bHh);
        return torch.rnn_cell(input, hx, wIh, wHh, opt);
    }

    /** LSTM cell functional. */
    public static Tensor lstm_cell(Tensor input, Tensor[] hx, Tensor wIh, Tensor wHh, Tensor bIh, Tensor bHh) {
        LSTMCellOptions opt = new LSTMCellOptions();
        if (bIh != null) opt.bias_ih(bIh);
        if (bHh != null) opt.bias_hh(bHh);
        Tensor h0 = hx[0];
        Tensor c0 = hx.length > 1 ? hx[1] : torch.zeros_like(h0);
        return torch.lstm_cell(input, h0, c0, wIh, wHh, opt);
    }

    /** GRU cell functional. */
    public static Tensor gru_cell(Tensor input, Tensor hx, Tensor wIh, Tensor wHh, Tensor bIh, Tensor bHh) {
        GRUCellOptions opt = new GRUCellOptions();
        if (bIh != null) opt.bias_ih(bIh);
        if (bHh != null) opt.bias_hh(bHh);
        return torch.gru_cell(input, hx, wIh, wHh, opt);
    }

    // ========================================================================
    // ELEMENT-WISE MATH (at:: functions)
    // ========================================================================

    public static Tensor abs(Tensor input) { return torch.abs(input); }
    public static Tensor absolute(Tensor input) { return torch.absolute(input); }
    public static Tensor acos(Tensor input) { return torch.acos(input); }
    public static Tensor acosh(Tensor input) { return torch.acosh(input); }
    public static Tensor asin(Tensor input) { return torch.asin(input); }
    public static Tensor asinh(Tensor input) { return torch.asinh(input); }
    public static Tensor atan(Tensor input) { return torch.atan(input); }
    public static Tensor atanh(Tensor input) { return torch.atanh(input); }
    public static Tensor ceil(Tensor input) { return torch.ceil(input); }
    public static Tensor clamp(Tensor input, Scalar min, Scalar max) { return torch.clamp(input, min, max); }
    public static Tensor clamp(Tensor input, double min, double max) { return torch.clamp(input, new Scalar(min), new Scalar(max)); }
    public static Tensor clamp_min(Tensor input, Scalar min) { return torch.clamp_min(input, min); }
    public static Tensor clamp_max(Tensor input, Scalar max) { return torch.clamp_max(input, max); }
    public static Tensor clip(Tensor input, double min, double max) { return clamp(input, min, max); }
    public static Tensor conj(Tensor input) { return torch.conj(input); }
    public static Tensor cos(Tensor input) { return torch.cos(input); }
    public static Tensor cosh(Tensor input) { return torch.cosh(input); }
    public static Tensor deg2rad(Tensor input) { return torch.deg2rad(input); }
    public static Tensor digamma(Tensor input) { return torch.digamma(input); }
    public static Tensor erf(Tensor input) { return torch.erf(input); }
    public static Tensor erfc(Tensor input) { return torch.erfc(input); }
    public static Tensor erfinv(Tensor input) { return torch.erfinv(input); }
    public static Tensor exp(Tensor input) { return torch.exp(input); }
    public static Tensor exp2(Tensor input) { return torch.exp2(input); }
    public static Tensor expm1(Tensor input) { return torch.expm1(input); }
    public static Tensor fix(Tensor input) { return torch.fix(input); }
    public static Tensor floor(Tensor input) { return torch.floor(input); }
    public static Tensor frac(Tensor input) { return torch.frac(input); }
    public static Tensor i0(Tensor input) { return torch.i0(input); }
    public static Tensor ldexp(Tensor input, Tensor other) { return torch.ldexp(input, other); }
    public static Tensor lgamma(Tensor input) { return torch.lgamma(input); }
    public static Tensor log(Tensor input) { return torch.log(input); }
    public static Tensor log10(Tensor input) { return torch.log10(input); }
    public static Tensor log1p(Tensor input) { return torch.log1p(input); }
    public static Tensor log2(Tensor input) { return torch.log2(input); }
    public static Tensor logaddexp(Tensor input, Tensor other) { return torch.logaddexp(input, other); }
    public static Tensor logaddexp2(Tensor input, Tensor other) { return torch.logaddexp2(input, other); }
    public static Tensor logical_not(Tensor input) { return torch.logical_not(input); }
    public static Tensor logit(Tensor input, double eps) { return torch.logit(input, eps); }
    public static Tensor logit(Tensor input) { return torch.logit(input, -1.0); }
    public static Tensor mvlgamma(Tensor input, long p) { return torch.mvlgamma(input, p); }
    public static Tensor neg(Tensor input) { return torch.neg(input); }
    public static Tensor negative(Tensor input) { return torch.negative(input); }
    public static Tensor polygamma(Tensor input, long n) { return torch.polygamma(input, n); }
    public static Tensor positive(Tensor input) { return torch.positive(input); }
    public static Tensor rad2deg(Tensor input) { return torch.rad2deg(input); }
    public static Tensor reciprocal(Tensor input) { return torch.reciprocal(input); }
    public static Tensor round(Tensor input) { return torch.round(input); }
    public static Tensor rsqrt(Tensor input) { return torch.rsqrt(input); }
    public static Tensor sgn(Tensor input) { return torch.sgn(input); }
    public static Tensor sign(Tensor input) { return torch.sign(input); }
    public static Tensor signbit(Tensor input) { return torch.signbit(input); }
    public static Tensor sin(Tensor input) { return torch.sin(input); }
    public static Tensor sinc(Tensor input) { return torch.sinc(input); }
    public static Tensor sinh(Tensor input) { return torch.sinh(input); }
    public static Tensor sqrt(Tensor input) { return torch.sqrt(input); }
    public static Tensor square(Tensor input) { return torch.square(input); }
    public static Tensor tan(Tensor input) { return torch.tan(input); }
    public static Tensor tanh(Tensor input) { return torch.tanh(input); }
    public static Tensor trunc(Tensor input) { return torch.trunc(input); }
    public static Tensor xlogy(Tensor input, Tensor other) { return torch.xlogy(input, other); }
    public static Tensor xlogy(Tensor input, Scalar other) { return torch.xlogy(input, other); }

    // ========================================================================
    // BINARY ELEMENT-WISE
    // ========================================================================

    public static Tensor add(Tensor input, Tensor other) { return torch.add(input, other); }
    public static Tensor add(Tensor input, Scalar other) { return torch.add(input, other); }
    public static Tensor sub(Tensor input, Tensor other) { return torch.sub(input, other); }
    public static Tensor sub(Tensor input, Scalar other) { return torch.sub(input, other); }
    public static Tensor mul(Tensor input, Tensor other) { return torch.mul(input, other); }
    public static Tensor mul(Tensor input, Scalar other) { return torch.mul(input, other); }
    public static Tensor multiply(Tensor input, Tensor other) { return torch.multiply(input, other); }
    public static Tensor multiply(Tensor input, Scalar other) { return torch.multiply(input, other); }
    public static Tensor div(Tensor input, Tensor other) { return torch.div(input, other); }
    public static Tensor div(Tensor input, Scalar other) { return torch.div(input, other); }
    public static Tensor divide(Tensor input, Tensor other) { return torch.divide(input, other); }
    public static Tensor divide(Tensor input, Scalar other) { return torch.divide(input, other); }
    public static Tensor floor_divide(Tensor input, Tensor other) { return torch.floor_divide(input, other); }
    public static Tensor floor_divide(Tensor input, Scalar other) { return torch.floor_divide(input, other); }
    public static Tensor true_divide(Tensor input, Tensor other) { return torch.true_divide(input, other); }
    public static Tensor fmod(Tensor input, Tensor other) { return torch.fmod(input, other); }
    public static Tensor fmod(Tensor input, Scalar other) { return torch.fmod(input, other); }
    public static Tensor remainder(Tensor input, Tensor other) { return torch.remainder(input, other); }
    public static Tensor remainder(Tensor input, Scalar other) { return torch.remainder(input, other); }
    public static Tensor pow(Tensor input, Tensor exponent) { return torch.pow(input, exponent); }
    public static Tensor pow(Tensor input, Scalar exponent) { return torch.pow(input, exponent); }
    public static Tensor pow(Tensor input, double exponent) { return torch.pow(input, new Scalar(exponent)); }
    public static Tensor float_power(Tensor input, Tensor exponent) { return torch.float_power(input, exponent); }
    public static Tensor float_power(Tensor input, Scalar exponent) { return torch.float_power(input, exponent); }
    public static Tensor hypot(Tensor input, Tensor other) { return torch.hypot(input, other); }
    public static Tensor atan2(Tensor input, Tensor other) { return torch.atan2(input, other); }
    public static Tensor atan2(Tensor input, Scalar other) { return torch.atan2(input, other); }
    public static Tensor arctan2(Tensor input, Tensor other) { return torch.arctan2(input, other); }
    public static Tensor bitwise_and(Tensor input, Tensor other) { return torch.bitwise_and(input, other); }
    public static Tensor bitwise_and(Tensor input, Scalar other) { return torch.bitwise_and(input, other); }
    public static Tensor bitwise_or(Tensor input, Tensor other) { return torch.bitwise_or(input, other); }
    public static Tensor bitwise_or(Tensor input, Scalar other) { return torch.bitwise_or(input, other); }
    public static Tensor bitwise_xor(Tensor input, Tensor other) { return torch.bitwise_xor(input, other); }
    public static Tensor bitwise_xor(Tensor input, Scalar other) { return torch.bitwise_xor(input, other); }
    public static Tensor bitwise_not(Tensor input) { return torch.bitwise_not(input); }
    public static Tensor bitwise_left_shift(Tensor input, Tensor other) { return torch.bitwise_left_shift(input, other); }
    public static Tensor bitwise_right_shift(Tensor input, Tensor other) { return torch.bitwise_right_shift(input, other); }
    public static Tensor copysign(Tensor input, Tensor other) { return torch.copysign(input, other); }
    public static Tensor copysign(Tensor input, Scalar other) { return torch.copysign(input, other); }
    public static Tensor nextafter(Tensor input, Tensor other) { return torch.nextafter(input, other); }
    public static Tensor gcd(Tensor input, Tensor other) { return torch.gcd(input, other); }
    public static Tensor lcm(Tensor input, Tensor other) { return torch.lcm(input, other); }
    public static Tensor maximum(Tensor input, Tensor other) { return torch.maximum(input, other); }
    public static Tensor minimum(Tensor input, Tensor other) { return torch.minimum(input, other); }
    public static Tensor fmax(Tensor input, Tensor other) { return torch.fmax(input, other); }
    public static Tensor fmin(Tensor input, Tensor other) { return torch.fmin(input, other); }

    // ========================================================================
    // COMPARISON
    // ========================================================================

    public static Tensor eq(Tensor input, Tensor other) { return torch.eq(input, other); }
    public static Tensor eq(Tensor input, Scalar other) { return torch.eq(input, other); }
    public static Tensor ne(Tensor input, Tensor other) { return torch.ne(input, other); }
    public static Tensor ne(Tensor input, Scalar other) { return torch.ne(input, other); }
    public static Tensor neq(Tensor input, Tensor other) { return ne(input, other); }
    public static Tensor lt(Tensor input, Tensor other) { return torch.lt(input, other); }
    public static Tensor lt(Tensor input, Scalar other) { return torch.lt(input, other); }
    public static Tensor less(Tensor input, Tensor other) { return torch.less(input, other); }
    public static Tensor le(Tensor input, Tensor other) { return torch.le(input, other); }
    public static Tensor le(Tensor input, Scalar other) { return torch.le(input, other); }
    public static Tensor less_equal(Tensor input, Tensor other) { return torch.less_equal(input, other); }
    public static Tensor gt(Tensor input, Tensor other) { return torch.gt(input, other); }
    public static Tensor gt(Tensor input, Scalar other) { return torch.gt(input, other); }
    public static Tensor greater(Tensor input, Tensor other) { return torch.greater(input, other); }
    public static Tensor ge(Tensor input, Tensor other) { return torch.ge(input, other); }
    public static Tensor ge(Tensor input, Scalar other) { return torch.ge(input, other); }
    public static Tensor greater_equal(Tensor input, Tensor other) { return torch.greater_equal(input, other); }
    public static Tensor isclose(Tensor input, Tensor other, double rtol, double atol, boolean equalNan) {
        return torch.isclose(input, other, rtol, atol, equalNan);
    }
    public static Tensor isclose(Tensor input, Tensor other) { return torch.isclose(input, other); }
    public static Tensor isfinite(Tensor input) { return torch.isfinite(input); }
    public static Tensor isinf(Tensor input) { return torch.isinf(input); }
    public static Tensor isnan(Tensor input) { return torch.isnan(input); }
    public static Tensor isneginf(Tensor input) { return torch.isneginf(input); }
    public static Tensor isposinf(Tensor input) { return torch.isposinf(input); }
    public static Tensor isreal(Tensor input) { return torch.isreal(input); }
    public static Tensor allclose(Tensor input, Tensor other, double rtol, double atol, boolean equalNan) {
        return torch.allclose(input, other, rtol, atol, equalNan);
    }
    public static Tensor allclose(Tensor input, Tensor other) {
        return torch.allclose(input, other);
    }
    public static Tensor logical_and(Tensor input, Tensor other) { return torch.logical_and(input, other); }
    public static Tensor logical_or(Tensor input, Tensor other) { return torch.logical_or(input, other); }
    public static Tensor logical_xor(Tensor input, Tensor other) { return torch.logical_xor(input, other); }
    public static Tensor where(Tensor condition, Tensor input, Tensor other) {
        return torch.where(condition, input, other);
    }
    public static Tensor where(Tensor condition, Scalar input, Tensor other) {
        return torch.where(condition, input, other);
    }
    public static Tensor where(Tensor condition, Tensor input, Scalar other) {
        return torch.where(condition, input, other);
    }
    public static Tensor where(Tensor condition, Scalar input, Scalar other) {
        return torch.where(condition, input, other);
    }

    // ========================================================================
    // REDUCTION
    // ========================================================================

    public static Tensor sum(Tensor input) { return torch.sum(input); }
    public static Tensor sum(Tensor input, long dim, boolean keepdim) { return torch.sum(input, dim, keepdim); }
    public static Tensor sum(Tensor input, long dim) { return sum(input, dim, false); }
    public static Tensor sum(Tensor input, long[] dim, boolean keepdim) {
        return torch.sum(input, new LongPointer(dim), keepdim);
    }
    public static Tensor sum(Tensor input, long[] dim) { return sum(input, dim, false); }
    public static Tensor mean(Tensor input) { return torch.mean(input); }
    public static Tensor mean(Tensor input, long dim, boolean keepdim) { return torch.mean(input, dim, keepdim); }
    public static Tensor mean(Tensor input, long dim) { return mean(input, dim, false); }
    public static Tensor mean(Tensor input, long[] dim, boolean keepdim) {
        return torch.mean(input, new LongPointer(dim), keepdim);
    }
    public static Tensor mean(Tensor input, long[] dim) { return mean(input, dim, false); }
    public static Tensor prod(Tensor input) { return torch.prod(input); }
    public static Tensor prod(Tensor input, long dim, boolean keepdim) { return torch.prod(input, dim, keepdim); }
    public static Tensor prod(Tensor input, long dim) { return prod(input, dim, false); }
    public static Tensor max(Tensor input) { return torch.max(input); }
    public static Tensor max(Tensor input, long dim, boolean keepdim) { return torch.max(input, dim, keepdim); }
    public static Tensor max(Tensor input, long dim) { return max(input, dim, false); }
    public static Tensor min(Tensor input) { return torch.min(input); }
    public static Tensor min(Tensor input, long dim, boolean keepdim) { return torch.min(input, dim, keepdim); }
    public static Tensor min(Tensor input, long dim) { return min(input, dim, false); }
    public static Tensor amin(Tensor input, long[] dim, boolean keepdim) {
        return torch.amin(input, new LongPointer(dim), keepdim);
    }
    public static Tensor amin(Tensor input, long dim, boolean keepdim) {
        return torch.amin(input, dim, keepdim);
    }
    public static Tensor amax(Tensor input, long[] dim, boolean keepdim) {
        return torch.amax(input, new LongPointer(dim), keepdim);
    }
    public static Tensor amax(Tensor input, long dim, boolean keepdim) {
        return torch.amax(input, dim, keepdim);
    }
    public static Tensor argmax(Tensor input) { return torch.argmax(input); }
    public static Tensor argmax(Tensor input, long dim, boolean keepdim) { return torch.argmax(input, dim, keepdim); }
    public static Tensor argmax(Tensor input, long dim) { return argmax(input, dim, false); }
    public static Tensor argmin(Tensor input) { return torch.argmin(input); }
    public static Tensor argmin(Tensor input, long dim, boolean keepdim) { return torch.argmin(input, dim, keepdim); }
    public static Tensor argmin(Tensor input, long dim) { return argmin(input, dim, false); }
    public static Tensor std(Tensor input) { return torch.std(input); }
    public static Tensor std(Tensor input, long dim, boolean unbiased, boolean keepdim) {
        return torch.std(input, dim, unbiased, keepdim);
    }
    public static Tensor std(Tensor input, long dim, boolean unbiased) { return std(input, dim, unbiased, false); }
    public static Tensor var(Tensor input) { return torch.var(input); }
    public static Tensor var(Tensor input, long dim, boolean unbiased, boolean keepdim) {
        return torch.var(input, dim, unbiased, keepdim);
    }
    public static Tensor var(Tensor input, long dim, boolean unbiased) { return var(input, dim, unbiased, false); }
    public static Tensor median(Tensor input) { return torch.median(input); }
    public static Tensor median(Tensor input, long dim, boolean keepdim) { return torch.median(input, dim, keepdim); }
    public static Tensor median(Tensor input, long dim) { return median(input, dim, false); }
    public static Tensor quantile(Tensor input, Tensor q, long dim, boolean keepdim) {
        return torch.quantile(input, q, dim, keepdim);
    }
    public static Tensor quantile(Tensor input, double q, long dim, boolean keepdim) {
        return torch.quantile(input, new Scalar(q), dim, keepdim);
    }
    public static Tensor nanmean(Tensor input, long[] dim, boolean keepdim) {
        return torch.nanmean(input, new LongPointer(dim), keepdim);
    }
    public static Tensor nanmean(Tensor input) { return torch.nanmean(input); }
    public static Tensor nanmedian(Tensor input) { return torch.nanmedian(input); }
    public static Tensor nansum(Tensor input) { return torch.nansum(input); }
    public static Tensor nansum(Tensor input, long[] dim, boolean keepdim) {
        return torch.nansum(input, new LongPointer(dim), keepdim);
    }
    public static Tensor nansum(Tensor input, long dim, boolean keepdim) {
        return torch.nansum(input, dim, keepdim);
    }
    public static Tensor nan_to_num(Tensor input, double nanVal, double posinfVal, double neginfVal) {
        return torch.nan_to_num(input, new Scalar(nanVal), new Scalar(posinfVal), new Scalar(neginfVal));
    }
    public static Tensor nan_to_num(Tensor input, double nanVal) {
        return torch.nan_to_num(input, new Scalar(nanVal));
    }
    public static Tensor nan_to_num(Tensor input) { return torch.nan_to_num(input); }
    public static Tensor norm(Tensor input, double p) { return torch.norm(input, p); }
    public static Tensor norm(Tensor input, Scalar p) { return torch.norm(input, p); }
    public static Tensor norm(Tensor input, double p, long dim, boolean keepdim) {
        return torch.norm(input, p, dim, keepdim);
    }
    public static Tensor norm(Tensor input, double p, long[] dim, boolean keepdim) {
        return torch.norm(input, p, new LongPointer(dim), keepdim);
    }
    public static Tensor frobenius_norm(Tensor input) { return torch.frobenius_norm(input); }
    public static Tensor frobenius_norm(Tensor input, long[] dim, boolean keepdim) {
        return torch.frobenius_norm(input, new LongPointer(dim), keepdim);
    }
    public static Tensor nuclear_norm(Tensor input) { return torch.nuclear_norm(input); }
    public static Tensor logsumexp(Tensor input, long dim, boolean keepdim) {
        return torch.logsumexp(input, dim, keepdim);
    }
    public static Tensor logsumexp(Tensor input, long dim) { return logsumexp(input, dim, false); }
    public static Tensor logcumsumexp(Tensor input, long dim) {
        return torch.logcumsumexp(input, dim);
    }
    public static Tensor cumsum(Tensor input, long dim) { return torch.cumsum(input, dim); }
    public static Tensor cumprod(Tensor input, long dim) { return torch.cumprod(input, dim); }
    public static Tensor dist(Tensor input, Tensor other, double p) {
        return torch.dist(input, other, p);
    }
    public static Tensor dist(Tensor input, Tensor other) {
        return torch.dist(input, other);
    }
    public static Tensor count_nonzero(Tensor input, long[] dim) {
        return torch.count_nonzero(input, new LongPointer(dim));
    }
    public static Tensor count_nonzero(Tensor input) { return torch.count_nonzero(input); }
    public static Tensor all(Tensor input) { return torch.all(input); }
    public static Tensor all(Tensor input, long dim, boolean keepdim) {
        return torch.all(input, dim, keepdim);
    }
    public static Tensor any(Tensor input) { return torch.any(input); }
    public static Tensor any(Tensor input, long dim, boolean keepdim) {
        return torch.any(input, dim, keepdim);
    }
    public static Tensor histogramc(Tensor input, long bins, double min, double max) {
        return torch.histc(input, bins, new Scalar(min), new Scalar(max));
    }
    public static Tensor trace(Tensor input) { return torch.trace(input); }
    public static Tensor det(Tensor input) { return torch.det(input); }
    public static Tensor logdet(Tensor input) { return torch.logdet(input); }
    public static Tensor matrix_exp(Tensor input) { return torch.matrix_exp(input); }

    // ========================================================================
    // TENSOR SHAPE / MANIPULATION
    // ========================================================================

    public static Tensor reshape(Tensor input, long[] shape) {
        return torch.reshape(input, new LongPointer(shape));
    }
    public static Tensor reshape(Tensor input, long... shape) { return reshape(input, shape); }
    public static Tensor view(Tensor input, long[] shape) {
        return input.view(new LongPointer(shape));
    }
    public static Tensor view(Tensor input, long... shape) { return view(input, shape); }
    public static Tensor flatten(Tensor input, long startDim, long endDim) {
        return torch.flatten(input, startDim, endDim);
    }
    public static Tensor flatten(Tensor input) {
        return torch.flatten(input);
    }
    public static Tensor unflatten(Tensor input, long dim, long[] sizes) {
        return torch.unflatten(input, dim, new LongVector(sizes));
    }
    public static Tensor squeeze(Tensor input) { return torch.squeeze(input); }
    public static Tensor squeeze(Tensor input, long dim) { return torch.squeeze(input, dim); }
    public static Tensor unsqueeze(Tensor input, long dim) { return torch.unsqueeze(input, dim); }
    public static Tensor transpose(Tensor input, long dim0, long dim1) {
        return torch.transpose(input, dim0, dim1);
    }
    public static Tensor t(Tensor input) { return torch.t(input); }
    public static Tensor permute(Tensor input, long[] dims) {
        return torch.permute(input, new LongPointer(dims));
    }
    public static Tensor permute(Tensor input, long... dims) { return permute(input, dims); }
    public static Tensor moveaxis(Tensor input, long source, long destination) {
        return torch.moveaxis(input, source, destination);
    }
    public static Tensor movedim(Tensor input, long source, long destination) {
        return torch.movedim(input, source, destination);
    }
    public static Tensor swapaxes(Tensor input, long axis0, long axis1) {
        return torch.swapaxes(input, axis0, axis1);
    }
    public static Tensor swapdims(Tensor input, long dim0, long dim1) {
        return torch.swapdims(input, dim0, dim1);
    }
    public static Tensor expand(Tensor input, long[] sizes) {
        return input.expand(new LongPointer(sizes));
    }
    public static Tensor expand_as(Tensor input, Tensor other) {
        return input.expand_as(other);
    }
    public static Tensor repeat(Tensor input, long[] repeats) {
        return input.repeat(new LongPointer(repeats));
    }
    public static Tensor broadcast_to(Tensor input, long[] size) {
        return torch.broadcast_to(input, new LongPointer(size));
    }
    public static Tensor cat(Tensor[] tensors, long dim) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.cat(list, dim);
    }
    public static Tensor concat(Tensor[] tensors, long dim) { return cat(tensors, dim); }
    public static Tensor concatenate(Tensor[] tensors, long dim) { return cat(tensors, dim); }
    public static Tensor stack(Tensor[] tensors, long dim) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.stack(list, dim);
    }
    public static Tensor split(Tensor tensor, long splitSize, long dim) {
        return torch.split(tensor, splitSize, dim);
    }
    public static Tensor split(Tensor tensor, long[] splitSizes, long dim) {
        TensorList list = torch.split(tensor, new LongPointer(splitSizes), dim);
        // list is TensorList already
        return list.get(0);
    }
    public static TensorList split_sizes(Tensor tensor, long[] splitSizes, long dim) {
        return torch.split(tensor, new LongPointer(splitSizes), dim);
    }
    public static Tensor chunk(Tensor tensor, long chunks, long dim) {
        return torch.chunk(tensor, chunks, dim).get(0);
    }
    public static TensorList chunks(Tensor tensor, long chunks, long dim) {
        return torch.chunk(tensor, chunks, dim);
    }
    public static TensorList hsplit(Tensor input, long sections) {
        return torch.hsplit(input, sections);
    }
    public static TensorList vsplit(Tensor input, long sections) {
        return torch.vsplit(input, sections);
    }
    public static TensorList dsplit(Tensor input, long sections) {
        return torch.dsplit(input, sections);
    }
    public static TensorList meshgrid(Tensor[] tensors, String indexing) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.meshgrid(list, indexing);
    }
    public static Tensor index_select(Tensor input, long dim, Tensor index) {
        return torch.index_select(input, dim, index);
    }
    public static Tensor masked_select(Tensor input, Tensor mask) {
        return torch.masked_select(input, mask);
    }
    public static Tensor masked_fill(Tensor input, Tensor mask, Scalar value) {
        return input.masked_fill(mask, value);
    }
    public static Tensor masked_fill(Tensor input, Tensor mask, double value) {
        return input.masked_fill(mask, new Scalar(value));
    }
    public static Tensor masked_scatter(Tensor input, Tensor mask, Tensor source) {
        return input.masked_scatter(mask, source);
    }
    public static Tensor nonzero(Tensor input) { return torch.nonzero(input); }
    public static Tensor take(Tensor input, Tensor index) { return torch.take(input, index); }
    public static Tensor take_along_dim(Tensor input, Tensor indices, long dim) {
        return torch.take_along_dim(input, indices, dim);
    }
    public static Tensor gather(Tensor input, long dim, Tensor index) {
        return torch.gather(input, dim, index);
    }
    public static Tensor scatter(Tensor input, long dim, Tensor index, Tensor src) {
        return torch.scatter(input, dim, index, src);
    }
    public static Tensor scatter_add(Tensor input, long dim, Tensor index, Tensor src) {
        return torch.scatter_add(input, dim, index, src);
    }
    public static Tensor diagonal(Tensor input, long offset, long dim1, long dim2) {
        return torch.diagonal(input, offset, dim1, dim2);
    }
    public static Tensor diagonal(Tensor input) {
        return torch.diagonal(input);
    }
    public static Tensor diag(Tensor input, long diagonal) {
        return torch.diag(input, diagonal);
    }
    public static Tensor diag(Tensor input) { return torch.diag(input); }
    public static Tensor diag_embed(Tensor input, long offset, long dim1, long dim2) {
        return torch.diag_embed(input, offset, dim1, dim2);
    }
    public static Tensor diagflat(Tensor input, long offset) {
        return torch.diagflat(input, offset);
    }
    public static Tensor diagflat(Tensor input) { return torch.diagflat(input); }
    public static Tensor triu(Tensor input, long diagonal) { return torch.triu(input, diagonal); }
    public static Tensor triu(Tensor input) { return torch.triu(input); }
    public static Tensor tril(Tensor input, long diagonal) { return torch.tril(input, diagonal); }
    public static Tensor tril(Tensor input) { return torch.tril(input); }
    public static Tensor triu_indices(long row, long col, long offset) {
        return torch.triu_indices(row, col, offset);
    }
    public static Tensor tril_indices(long row, long col, long offset) {
        return torch.tril_indices(row, col, offset);
    }
    public static Tensor roll(Tensor input, long[] shifts, long[] dims) {
        return torch.roll(input, new LongPointer(shifts), new LongPointer(dims));
    }
    public static Tensor roll(Tensor input, long[] shifts) {
        return torch.roll(input, new LongPointer(shifts));
    }
    public static Tensor flip(Tensor input, long[] dims) {
        return torch.flip(input, new LongPointer(dims));
    }
    public static Tensor fliplr(Tensor input) { return torch.fliplr(input); }
    public static Tensor flipud(Tensor input) { return torch.flipud(input); }
    public static Tensor rot90(Tensor input, long k, long[] dims) {
        return torch.rot90(input, k, new LongPointer(dims));
    }
    public static Tensor rot90(Tensor input, long k) {
        return torch.rot90(input, k, new LongPointer(new long[]{0, 1}));
    }
    public static Tensor rot90(Tensor input) {
        return rot90(input, 1);
    }
    public static Tensor tile(Tensor input, long[] dims) {
        return torch.tile(input, new LongPointer(dims));
    }
    public static Tensor repeat_interleave(Tensor input, long repeats, long dim) {
        return torch.repeat_interleave(input, repeats, dim);
    }
    public static Tensor repeat_interleave(Tensor input, long repeats) {
        return torch.repeat_interleave(input, repeats);
    }
    public static Tensor repeat_interleave(Tensor input, Tensor repeats, long dim) {
        return torch.repeat_interleave(input, repeats, dim);
    }
    public static Tensor permute_copy(Tensor input, long[] dims) {
        return torch.permute_copy(input, new LongPointer(dims));
    }
    public static Tensor pad(Tensor input, long[] pad, String mode) {
        return pad(input, pad, mode, 0.0);
    }
    public static Tensor atleast_1d(Tensor[] tensors) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.atleast_1d(list);
    }
    public static Tensor atleast_1d(Tensor input) { return torch.atleast_1d(input); }
    public static Tensor atleast_2d(Tensor[] tensors) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.atleast_2d(list);
    }
    public static Tensor atleast_2d(Tensor input) { return torch.atleast_2d(input); }
    public static Tensor atleast_3d(Tensor[] tensors) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.atleast_3d(list);
    }
    public static Tensor atleast_3d(Tensor input) { return torch.atleast_3d(input); }
    public static Tensor hstack(Tensor[] tensors) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.hstack(list);
    }
    public static Tensor vstack(Tensor[] tensors) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.vstack(list);
    }
    public static Tensor dstack(Tensor[] tensors) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.dstack(list);
    }
    public static Tensor column_stack(Tensor[] tensors) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.column_stack(list);
    }
    public static Tensor row_stack(Tensor[] tensors) {
        return vstack(tensors);
    }
    public static Tensor cartesian_prod(Tensor[] tensors) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.cartesian_prod(list);
    }
    public static Tensor combinations(Tensor input, long r, boolean withReplacement) {
        return torch.combinations(input, r, withReplacement);
    }
    public static Tensor tensordot(Tensor input, Tensor other, long[] dimsA, long[] dimsB) {
        return torch.tensordot(input, other, new LongPointer(dimsA), new LongPointer(dimsB));
    }
    public static Tensor tensordot(Tensor input, Tensor other, long axes) {
        return torch.tensordot(input, other, axes);
    }
    public static Tensor einsum(String equation, Tensor[] operands) {
        TensorList list = new TensorList(operands.length);
        for (int i = 0; i < operands.length; i++) list.put(i, operands[i]);
        return torch.einsum(equation, list);
    }
    public static Tensor matmul(Tensor input, Tensor other) {
        return torch.matmul(input, other);
    }
    public static Tensor bmm(Tensor input, Tensor other) {
        return torch.bmm(input, other);
    }
    public static Tensor baddbmm(Tensor input, Tensor batch1, Tensor batch2, double beta, double alpha) {
        return torch.baddbmm(input, batch1, batch2, new Scalar(beta), new Scalar(alpha));
    }
    public static Tensor baddbmm(Tensor input, Tensor batch1, Tensor batch2) {
        return baddbmm(input, batch1, batch2, 1.0, 1.0);
    }
    public static Tensor mm(Tensor input, Tensor other) { return torch.mm(input, other); }
    public static Tensor addmm(Tensor input, Tensor mat1, Tensor mat2, double beta, double alpha) {
        return torch.addmm(input, mat1, mat2, new Scalar(beta), new Scalar(alpha));
    }
    public static Tensor addmm(Tensor input, Tensor mat1, Tensor mat2) {
        return addmm(input, mat1, mat2, 1.0, 1.0);
    }
    public static Tensor addbmm(Tensor input, Tensor batch1, Tensor batch2, double beta, double alpha) {
        return torch.addbmm(input, batch1, batch2, new Scalar(beta), new Scalar(alpha));
    }
    public static Tensor addbmm(Tensor input, Tensor batch1, Tensor batch2) {
        return addbmm(input, batch1, batch2, 1.0, 1.0);
    }
    public static Tensor addmv(Tensor input, Tensor mat, Tensor vec, double beta, double alpha) {
        return torch.addmv(input, mat, vec, new Scalar(beta), new Scalar(alpha));
    }
    public static Tensor addmv(Tensor input, Tensor mat, Tensor vec) {
        return addmv(input, mat, vec, 1.0, 1.0);
    }
    public static Tensor addr(Tensor input, Tensor vec1, Tensor vec2, double beta, double alpha) {
        return torch.addr(input, vec1, vec2, new Scalar(beta), new Scalar(alpha));
    }
    public static Tensor addr(Tensor input, Tensor vec1, Tensor vec2) {
        return addr(input, vec1, vec2, 1.0, 1.0);
    }
    public static Tensor dot(Tensor input, Tensor other) { return torch.dot(input, other); }
    public static Tensor mv(Tensor input, Tensor other) { return torch.mv(input, other); }
    public static Tensor outer(Tensor input, Tensor vec2) { return torch.outer(input, vec2); }
    public static Tensor inner(Tensor input, Tensor other) { return torch.inner(input, other); }
    public static Tensor ger(Tensor input, Tensor vec2) { return torch.ger(input, vec2); }
    public static Tensor cross(Tensor input, Tensor other, long dim) {
        return torch.cross(input, other, dim);
    }
    public static Tensor cross(Tensor input, Tensor other) {
        return torch.cross(input, other);
    }
    public static Tensor trace(Tensor input) { return torch.trace(input); }
    public static Tensor det(Tensor input) { return torch.det(input); }
    public static Tensor dot_product(Tensor input, Tensor other) { return torch.dot(input, other); }
    public static Tensor vdot(Tensor input, Tensor other) { return torch.vdot(input, other); }
    public static Tensor inverse(Tensor input) { return torch.inverse(input); }
    public static Tensor pinverse(Tensor input) { return torch.pinverse(input); }
    public static Tensor matrix_power(Tensor input, long n) { return torch.matrix_power(input, n); }
    public static Tensor renorm(Tensor input, double p, long dim, double maxnorm) {
        return torch.renorm(input, p, dim, maxnorm);
    }
    public static Tensor kron(Tensor input, Tensor other) { return torch.kron(input, other); }
    public static Tensor cov(Tensor input) { return torch.cov(input); }
    public static Tensor corrcoef(Tensor input) { return torch.corrcoef(input); }
    public static Tensor polar(Tensor abs, Tensor angle) { return torch.polar(abs, angle); }
    public static Tensor angle(Tensor input) { return torch.angle(input); }
    public static Tensor imag(Tensor input) { return torch.imag(input); }
    public static Tensor real(Tensor input) { return torch.real(input); }
    public static Tensor conj_physical(Tensor input) { return torch.conj_physical(input); }
    public static Tensor view_as_complex(Tensor input) { return torch.view_as_complex(input); }
    public static Tensor view_as_real(Tensor input) { return torch.view_as_real(input); }
    public static Tensor broadcast_shapes(long[] shape1, long[] shape2) {
        return torch.broadcast_shapes(new LongPointer(shape1), new LongPointer(shape2));
    }
    public static Tensor broadcast_tensors(Tensor[] tensors) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.broadcast_tensors(list);
    }
    public static Tensor chain_matmul(Tensor[] matrices) {
        TensorList list = new TensorList(matrices.length);
        for (int i = 0; i < matrices.length; i++) list.put(i, matrices[i]);
        return torch.chain_matmul(list);
    }
    public static Tensor t(Tensor input, long dim0, long dim1) {
        return torch.transpose(input, dim0, dim1);
    }

    // ========================================================================
    // TENSOR CREATION
    // ========================================================================

    public static Tensor tensor(float[] data) { return torch.tensor((FloatBuffer) null); }
    public static Tensor tensor(int[] data) { return torch.tensor((IntBuffer) null); }
    public static Tensor tensor(boolean[] data) { return torch.tensor((boolean[]) null); }

    /** Note: For actual tensor creation use {@link #tensor(FloatBuffer)} etc. */
    public static Tensor from_blob(Pointer data, long[] shape, TensorOptions options) {
        return torch.from_blob(data, new LongPointer(shape), new LongPointer(new long[]{shape.length}), options);
    }

    public static Tensor arange(Scalar end, TensorOptions options) {
        return torch.arange(end, options);
    }
    public static Tensor arange(Scalar end) {
        return torch.arange(end);
    }
    public static Tensor arange(Scalar start, Scalar end, TensorOptions options) {
        return torch.arange(start, end, options);
    }
    public static Tensor arange(Scalar start, Scalar end) {
        return torch.arange(start, end);
    }
    public static Tensor arange(Scalar start, Scalar end, Scalar step, TensorOptions options) {
        return torch.arange(start, end, step, options);
    }
    public static Tensor arange(Scalar start, Scalar end, Scalar step) {
        return torch.arange(start, end, step);
    }
    public static Tensor arange(double end) { return arange(new Scalar(end)); }
    public static Tensor arange(double start, double end) {
        return arange(new Scalar(start), new Scalar(end));
    }
    public static Tensor arange(double start, double end, double step) {
        return arange(new Scalar(start), new Scalar(end), new Scalar(step));
    }
    public static Tensor range(Scalar start, Scalar end, Scalar step, TensorOptions options) {
        return torch.range(start, end, step, options);
    }
    public static Tensor range(Scalar start, Scalar end, Scalar step) {
        return torch.range(start, end, step);
    }
    public static Tensor linspace(Scalar start, Scalar end, long steps, TensorOptions options) {
        return torch.linspace(start, end, steps, options);
    }
    public static Tensor linspace(Scalar start, Scalar end, long steps) {
        return torch.linspace(start, end, steps);
    }
    public static Tensor linspace(double start, double end, long steps) {
        return linspace(new Scalar(start), new Scalar(end), steps);
    }
    public static Tensor logspace(Scalar start, Scalar end, long steps, double base, TensorOptions options) {
        return torch.logspace(start, end, steps, base, options);
    }
    public static Tensor logspace(Scalar start, Scalar end, long steps, double base) {
        return torch.logspace(start, end, steps, base);
    }
    public static Tensor logspace(double start, double end, long steps, double base) {
        return logspace(new Scalar(start), new Scalar(end), steps, base);
    }
    public static Tensor eye(long n, TensorOptions options) {
        return torch.eye(n, options);
    }
    public static Tensor eye(long n) {
        return torch.eye(n);
    }
    public static Tensor eye(long n, long m, TensorOptions options) {
        return torch.eye(n, m, options);
    }
    public static Tensor eye(long n, long m) {
        return torch.eye(n, m);
    }
    public static Tensor empty(long[] size, TensorOptions options) {
        return torch.empty(new LongPointer(size), options);
    }
    public static Tensor empty(long[] size) {
        return torch.empty(new LongPointer(size));
    }
    public static Tensor empty(long... size) { return empty(size); }
    public static Tensor zeros(long[] size, TensorOptions options) {
        return torch.zeros(new LongPointer(size), options);
    }
    public static Tensor zeros(long[] size) {
        return torch.zeros(new LongPointer(size));
    }
    public static Tensor zeros(long... size) { return zeros(size); }
    public static Tensor ones(long[] size, TensorOptions options) {
        return torch.ones(new LongPointer(size), options);
    }
    public static Tensor ones(long[] size) {
        return torch.ones(new LongPointer(size));
    }
    public static Tensor ones(long... size) { return ones(size); }
    public static Tensor full(long[] size, Scalar value, TensorOptions options) {
        return torch.full(new LongPointer(size), value, options);
    }
    public static Tensor full(long[] size, Scalar value) {
        return torch.full(new LongPointer(size), value);
    }
    public static Tensor full(long[] size, double value) {
        return full(size, new Scalar(value));
    }
    public static Tensor full(long[] size, double value, TensorOptions options) {
        return full(size, new Scalar(value), options);
    }
    public static Tensor empty_like(Tensor input) { return torch.empty_like(input); }
    public static Tensor empty_like(Tensor input, TensorOptions options) {
        return torch.empty_like(input, options);
    }
    public static Tensor zeros_like(Tensor input) { return torch.zeros_like(input); }
    public static Tensor zeros_like(Tensor input, TensorOptions options) {
        return torch.zeros_like(input, options);
    }
    public static Tensor ones_like(Tensor input) { return torch.ones_like(input); }
    public static Tensor ones_like(Tensor input, TensorOptions options) {
        return torch.ones_like(input, options);
    }
    public static Tensor full_like(Tensor input, Scalar value) { return torch.full_like(input, value); }
    public static Tensor full_like(Tensor input, double value) {
        return full_like(input, new Scalar(value));
    }
    public static Tensor rand(long[] size, TensorOptions options) {
        return torch.rand(new LongPointer(size), options);
    }
    public static Tensor rand(long[] size) {
        return torch.rand(new LongPointer(size));
    }
    public static Tensor rand(long... size) { return rand(size); }
    public static Tensor randn(long[] size, TensorOptions options) {
        return torch.randn(new LongPointer(size), options);
    }
    public static Tensor randn(long[] size) {
        return torch.randn(new LongPointer(size));
    }
    public static Tensor randn(long... size) { return randn(size); }
    public static Tensor randint(long low, long high, long[] size, TensorOptions options) {
        return torch.randint(low, high, new LongPointer(size), options);
    }
    public static Tensor randint(long low, long high, long[] size) {
        return torch.randint(low, high, new LongPointer(size));
    }
    public static Tensor randint(long low, long high, long... size) { return randint(low, high, size); }
    public static Tensor randperm(long n, TensorOptions options) {
        return torch.randperm(n, options);
    }
    public static Tensor randperm(long n) {
        return torch.randperm(n);
    }
    public static Tensor normal(double mean, double std, long[] size, TensorOptions options) {
        return torch.normal(new Scalar(mean), new Scalar(std), new LongPointer(size), options);
    }
    public static Tensor normal(double mean, double std, long[] size) {
        return torch.normal(new Scalar(mean), new Scalar(std), new LongPointer(size));
    }
    public static Tensor normal(Tensor mean, Tensor std, long[] size, TensorOptions options) {
        return torch.normal(mean, std, new LongPointer(size), options);
    }
    public static Tensor normal(Tensor mean, Tensor std, long[] size) {
        return torch.normal(mean, std, new LongPointer(size));
    }
    public static Tensor bernoulli(Tensor input, TensorOptions options) {
        return torch.bernoulli(input, options);
    }
    public static Tensor bernoulli(Tensor input) {
        return torch.bernoulli(input);
    }
    public static Tensor multinomial(Tensor input, long numSamples, boolean replacement) {
        return torch.multinomial(input, numSamples, replacement);
    }
    public static Tensor multinomial(Tensor input, long numSamples) {
        return multinomial(input, numSamples, false);
    }
    public static Tensor poisson(Tensor input) { return torch.poisson(input); }
    public static Tensor exponential(Tensor input) { return torch.exponential(input); }
    public static Tensor geometric(Tensor input, double p) { return torch.geometric(input, p); }
    public static Tensor cauchy(Tensor input) { return torch.cauchy(input); }
    public static Tensor log_normal(Tensor input, double mean, double std) {
        return torch.log_normal(input, new Scalar(mean), new Scalar(std));
    }
    public static Tensor log_normal(Tensor input) { return torch.log_normal(input); }
    public static Tensor random(Tensor input, long from, long to, TensorOptions options) {
        return torch.random(input, from, to, options);
    }
    public static Tensor uniform(Tensor input, double from, double to) {
        return torch.uniform(input, from, to);
    }
    public static Tensor bartlett_window(long windowLength, boolean periodic, TensorOptions options) {
        return torch.bartlett_window(windowLength, periodic, options);
    }
    public static Tensor bartlett_window(long windowLength, boolean periodic) {
        return torch.bartlett_window(windowLength, periodic);
    }
    public static Tensor bartlett_window(long windowLength, TensorOptions options) {
        return torch.bartlett_window(windowLength, options);
    }
    public static Tensor bartlett_window(long windowLength) {
        return torch.bartlett_window(windowLength);
    }
    public static Tensor blackman_window(long windowLength, boolean periodic, TensorOptions options) {
        return torch.blackman_window(windowLength, periodic, options);
    }
    public static Tensor blackman_window(long windowLength, boolean periodic) {
        return torch.blackman_window(windowLength, periodic);
    }
    public static Tensor blackman_window(long windowLength, TensorOptions options) {
        return torch.blackman_window(windowLength, options);
    }
    public static Tensor blackman_window(long windowLength) {
        return torch.blackman_window(windowLength);
    }
    public static Tensor hamming_window(long windowLength, boolean periodic, double alpha, double beta,
                                        TensorOptions options) {
        return torch.hamming_window(windowLength, periodic, alpha, beta, options);
    }
    public static Tensor hamming_window(long windowLength, boolean periodic, double alpha, double beta) {
        return torch.hamming_window(windowLength, periodic, alpha, beta);
    }
    public static Tensor hamming_window(long windowLength, boolean periodic, TensorOptions options) {
        return torch.hamming_window(windowLength, periodic, options);
    }
    public static Tensor hamming_window(long windowLength, boolean periodic) {
        return torch.hamming_window(windowLength, periodic);
    }
    public static Tensor hamming_window(long windowLength, TensorOptions options) {
        return torch.hamming_window(windowLength, options);
    }
    public static Tensor hamming_window(long windowLength) {
        return torch.hamming_window(windowLength);
    }
    public static Tensor hann_window(long windowLength, boolean periodic, TensorOptions options) {
        return torch.hann_window(windowLength, periodic, options);
    }
    public static Tensor hann_window(long windowLength, boolean periodic) {
        return torch.hann_window(windowLength, periodic);
    }
    public static Tensor hann_window(long windowLength, TensorOptions options) {
        return torch.hann_window(windowLength, options);
    }
    public static Tensor hann_window(long windowLength) {
        return torch.hann_window(windowLength);
    }
    public static Tensor kaiser_window(long windowLength, boolean periodic, double beta, TensorOptions options) {
        return torch.kaiser_window(windowLength, periodic, beta, options);
    }
    public static Tensor kaiser_window(long windowLength, boolean periodic, double beta) {
        return torch.kaiser_window(windowLength, periodic, beta);
    }
    public static Tensor kaiser_window(long windowLength, TensorOptions options) {
        return torch.kaiser_window(windowLength, options);
    }
    public static Tensor kaiser_window(long windowLength) {
        return torch.kaiser_window(windowLength);
    }
    public static Tensor vander(Tensor x, long N, boolean increasing) {
        return torch.vander(x, N, increasing);
    }
    public static Tensor vander(Tensor x, long N) {
        return vander(x, N, false);
    }
    public static Tensor vander(Tensor x) {
        return torch.vander(x);
    }
    public static Tensor pad_sequence(Tensor[] sequences, boolean batchFirst, double paddingValue) {
        TensorList list = new TensorList(sequences.length);
        for (int i = 0; i < sequences.length; i++) list.put(i, sequences[i]);
        return torch.pad_sequence(list, batchFirst, new Scalar(paddingValue));
    }
    public static Tensor pad_sequence(Tensor[] sequences, boolean batchFirst) {
        return pad_sequence(sequences, batchFirst, 0.0);
    }

    // ========================================================================
    // INDEXING / ADVANCED INDEXING
    // ========================================================================

    public static Tensor index(Tensor input, Tensor[] indices) {
        TensorList list = new TensorList(indices.length);
        for (int i = 0; i < indices.length; i++) list.put(i, indices[i]);
        return torch.index(input, list);
    }
    public static Tensor index_put(Tensor input, Tensor[] indices, Tensor values, boolean accumulate) {
        TensorList list = new TensorList(indices.length);
        for (int i = 0; i < indices.length; i++) list.put(i, indices[i]);
        return torch.index_put(input, list, values, accumulate);
    }
    public static Tensor index_add(Tensor input, long dim, Tensor index, Tensor source) {
        return torch.index_add(input, dim, index, source);
    }
    public static Tensor index_copy(Tensor input, long dim, Tensor index, Tensor source) {
        return torch.index_copy(input, dim, index, source);
    }
    public static Tensor index_fill(Tensor input, long dim, Tensor index, Scalar value) {
        return torch.index_fill(input, dim, index, value);
    }
    public static Tensor index_fill(Tensor input, long dim, Tensor index, double value) {
        return index_fill(input, dim, index, new Scalar(value));
    }
    public static Tensor gather(Tensor input, long dim, Tensor index, boolean sparseGrad) {
        return torch.gather(input, dim, index, sparseGrad);
    }
    public static Tensor scatter(Tensor input, long dim, Tensor index, Tensor value, boolean reduce) {
        return torch.scatter_reduce(input, dim, index, value,
                org.bytedeco.pytorch.global.torch.ScatterReduceMode().Max());
    }
    public static Tensor scatter_add(Tensor input, long dim, Tensor index, Tensor src) {
        return torch.scatter_add(input, dim, index, src);
    }
    public static Tensor bucketize(Tensor input, Tensor boundaries, boolean outInt32, boolean right) {
        return torch.bucketize(input, boundaries, outInt32, right);
    }
    public static Tensor bucketize(Tensor input, Tensor boundaries, boolean outInt32) {
        return bucketize(input, boundaries, outInt32, false);
    }
    public static Tensor bucketize(Tensor input, Tensor boundaries) {
        return bucketize(input, boundaries, false, false);
    }
    public static Tensor searchsorted(Tensor sortedSequence, Tensor values, boolean outInt32, boolean right) {
        return torch.searchsorted(sortedSequence, values, outInt32, right);
    }
    public static Tensor searchsorted(Tensor sortedSequence, Tensor values) {
        return torch.searchsorted(sortedSequence, values);
    }

    // ========================================================================
    // MISC / UTILITY
    // ========================================================================

    public static Tensor clone(Tensor input) { return torch.clone(input); }
    public static Tensor detach(Tensor input) { return torch.detach(input); }
    public static Tensor contiguous(Tensor input) { return input.contiguous(); }
    public static Tensor to(Tensor input, TensorOptions options) { return input.to(options); }
    public static Tensor to(Tensor input, org.bytedeco.pytorch.Device device) {
        return torch._to_copy(input, org.bytedeco.pytorch.global.torch.TensorOptions().device_(device));
    }
    public static Tensor cpu(Tensor input) {
        return input.cpu();
    }
    public static Tensor cuda(Tensor input) {
        return input.cuda();
    }
    public static Tensor cast(Tensor input, org.bytedeco.pytorch.global.torch.ScalarType dtype) {
        return input.to(dtype);
    }
    public static Tensor type_as(Tensor input, Tensor other) {
        return input.type_as(other);
    }
    public static Tensor copy_(Tensor input, Tensor src) {
        return input.copy_(src);
    }
    public static Tensor dequantize(Tensor input) { return torch.dequantize(input); }
    public static Tensor int_repr(Tensor input) { return torch.int_repr(input); }
    public static Tensor select(Tensor input, long dim, long index) {
        return torch.select(input, dim, index);
    }
    public static Tensor narrow(Tensor input, long dim, long start, long length) {
        return torch.narrow(input, dim, start, length);
    }
    public static Tensor slice(Tensor input, long dim, long start, long end, long step) {
        return torch.slice(input, dim, start, end, step);
    }
    public static TensorList unbind(Tensor input, long dim) {
        return torch.unbind(input, dim);
    }
    public static TensorList tensor_split(Tensor input, long[] sections, long dim) {
        return torch.tensor_split(input, new LongPointer(sections), dim);
    }
    public static TensorList tensor_split(Tensor input, long sections, long dim) {
        return torch.tensor_split(input, sections, dim);
    }
    public static TensorList tensor_split(Tensor input, long[] indices) {
        return torch.tensor_split(input, new LongPointer(indices));
    }
    public static TensorList tensor_split(Tensor input, long indices) {
        return torch.tensor_split(input, indices);
    }
    public static TensorList split_with_sizes(Tensor input, long[] sizes, long dim) {
        return torch.split(input, new LongPointer(sizes), dim);
    }
    public static TensorList split(Tensor input, long[] sizes, long dim) {
        return torch.split(input, new LongPointer(sizes), dim);
    }
    public static TensorList split(Tensor input, long chunks, long dim) {
        return torch.split(input, chunks, dim);
    }
    public static Tensor expand(Tensor input, long[] sizes, boolean implicit) {
        return input.expand(new LongPointer(sizes));
    }
    public static Tensor as_strided(Tensor input, long[] size, long[] stride, long storageOffset) {
        return torch.as_strided(input, new LongPointer(size), new LongPointer(stride), storageOffset);
    }
    public static Tensor as_strided(Tensor input, long[] size, long[] stride) {
        return as_strided(input, size, stride, 0);
    }
    public static Tensor histc(Tensor input, long bins, double min, double max) {
        return histogramc(input, bins, min, max);
    }
    public static Tensor block_diag(Tensor[] tensors) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.block_diag(list);
    }
    public static Tensor cdist(Tensor x1, Tensor x2, double p, boolean computeMode) {
        return torch.cdist(x1, x2, p, computeMode);
    }
    public static Tensor cdist(Tensor x1, Tensor x2, double p) {
        return cdist(x1, x2, p, false);
    }
    public static Tensor cdist(Tensor x1, Tensor x2) {
        return cdist(x1, x2, 2.0);
    }
    public static Tensor dot(Tensor a, Tensor b) { return torch.dot(a, b); }
    public static Tensor pdist(Tensor input, double p) { return torch.pdist(input, p); }
    public static Tensor pdist(Tensor input) { return pdist(input, 2.0); }
    public static Tensor std_mean(Tensor input, long dim, boolean unbiased, boolean keepdim) {
        Tensor[] r = torch.std_mean(input, dim, unbiased, keepdim);
        return r[0];
    }
    public static Tensor std_mean(Tensor input) {
        return torch.std_mean(input);
    }
    public static Tensor var_mean(Tensor input, long dim, boolean unbiased, boolean keepdim) {
        Tensor[] r = torch.var_mean(input, dim, unbiased, keepdim);
        return r[0];
    }
    public static Tensor var_mean(Tensor input) {
        return torch.var_mean(input);
    }

    // ========================================================================
    // SIGNAL / SPECTROGRAM
    // ========================================================================

    public static Tensor stft(Tensor input, long nFFT, long hopLength, long winLength, Tensor window,
                              boolean center, String padMode, boolean normalized, boolean onesided,
                              boolean returnComplex) {
        return torch.stft(input, nFFT, hopLength, winLength, window, center, padMode, normalized, onesided, returnComplex);
    }

    public static Tensor istft(Tensor input, long nFFT, long hopLength, long winLength, Tensor window,
                               boolean center, boolean normalized, boolean onesided, long length) {
        return torch.istft(input, nFFT, hopLength, winLength, window, center, normalized, onesided, length);
    }

    // ========================================================================
    // FFT
    // ========================================================================

    public static Tensor fft(Tensor input, long n, long dim, String norm) {
        return torch.fft_fft(input, n, dim, norm);
    }
    public static Tensor fft(Tensor input) { return torch.fft_fft(input); }
    public static Tensor ifft(Tensor input, long n, long dim, String norm) {
        return torch.fft_ifft(input, n, dim, norm);
    }
    public static Tensor ifft(Tensor input) { return torch.fft_ifft(input); }
    public static Tensor rfft(Tensor input, long n, long dim, String norm) {
        return torch.fft_rfft(input, n, dim, norm);
    }
    public static Tensor rfft(Tensor input) { return torch.fft_rfft(input); }
    public static Tensor irfft(Tensor input, long n, long dim, String norm) {
        return torch.fft_irfft(input, n, dim, norm);
    }
    public static Tensor irfft(Tensor input) { return torch.fft_irfft(input); }
    public static Tensor hfft(Tensor input, long n, long dim, String norm) {
        return torch.fft_hfft(input, n, dim, norm);
    }
    public static Tensor hfft(Tensor input) { return torch.fft_hfft(input); }
    public static Tensor ihfft(Tensor input, long n, long dim, String norm) {
        return torch.fft_ihfft(input, n, dim, norm);
    }
    public static Tensor ihfft(Tensor input) { return torch.fft_ihfft(input); }
    public static Tensor fft2(Tensor input, long[] s, long[] dim, String norm) {
        return torch.fft_fft2(input, new LongPointer(s), new LongPointer(dim), norm);
    }
    public static Tensor fft2(Tensor input) { return torch.fft_fft2(input); }
    public static Tensor ifft2(Tensor input, long[] s, long[] dim, String norm) {
        return torch.fft_ifft2(input, new LongPointer(s), new LongPointer(dim), norm);
    }
    public static Tensor ifft2(Tensor input) { return torch.fft_ifft2(input); }
    public static Tensor rfft2(Tensor input, long[] s, long[] dim, String norm) {
        return torch.fft_rfft2(input, new LongPointer(s), new LongPointer(dim), norm);
    }
    public static Tensor rfft2(Tensor input) { return torch.fft_rfft2(input); }
    public static Tensor irfft2(Tensor input, long[] s, long[] dim, String norm) {
        return torch.fft_irfft2(input, new LongPointer(s), new LongPointer(dim), norm);
    }
    public static Tensor irfft2(Tensor input) { return torch.fft_irfft2(input); }
    public static Tensor hfft2(Tensor input, long[] s, long[] dim, String norm) {
        return torch.fft_hfft2(input, new LongPointer(s), new LongPointer(dim), norm);
    }
    public static Tensor ihfft2(Tensor input, long[] s, long[] dim, String norm) {
        return torch.fft_ihfft2(input, new LongPointer(s), new LongPointer(dim), norm);
    }
    public static Tensor ffts_shift(Tensor input, long[] dim) {
        return torch.fft_fftshift(input, new LongPointer(dim));
    }
    public static Tensor ffts_shift(Tensor input) { return torch.fft_fftshift(input); }
    public static Tensor ifft_shift(Tensor input, long[] dim) {
        return torch.fft_ifftshift(input, new LongPointer(dim));
    }
    public static Tensor ifft_shift(Tensor input) { return torch.fft_ifftshift(input); }
    public static Tensor fftfreq(long n, double d, TensorOptions options) {
        return torch.fft_fftfreq(n, d, options);
    }
    public static Tensor fftfreq(long n, double d) {
        return torch.fft_fftfreq(n, d);
    }
    public static Tensor fftfreq(long n, TensorOptions options) {
        return torch.fft_fftfreq(n, options);
    }
    public static Tensor fftfreq(long n) {
        return torch.fft_fftfreq(n);
    }
    public static Tensor rfftfreq(long n, double d, TensorOptions options) {
        return torch.fft_rfftfreq(n, d, options);
    }
    public static Tensor rfftfreq(long n, double d) {
        return torch.fft_rfftfreq(n, d);
    }
    public static Tensor rfftfreq(long n, TensorOptions options) {
        return torch.fft_rfftfreq(n, options);
    }
    public static Tensor rfftfreq(long n) {
        return torch.fft_rfftfreq(n);
    }

    // ========================================================================
    // LINALG
    // ========================================================================

    public static Tensor linalg_norm(Tensor input, String ord, long[] dim, boolean keepdim) {
        return torch.linalg_norm(input, ord, new LongPointer(dim), keepdim);
    }
    public static Tensor linalg_norm(Tensor input) {
        return torch.linalg_norm(input);
    }
    public static Tensor linalg_norm(Tensor input, String ord) {
        return torch.linalg_norm(input, ord);
    }
    public static Tensor linalg_vector_norm(Tensor input, double ord, long[] dim, boolean keepdim) {
        return torch.linalg_vector_norm(input, ord, new LongPointer(dim), keepdim);
    }
    public static Tensor linalg_vector_norm(Tensor input, double ord) {
        return torch.linalg_vector_norm(input, ord);
    }
    public static Tensor linalg_vector_norm(Tensor input) {
        return torch.linalg_vector_norm(input);
    }
    public static Tensor linalg_matrix_norm(Tensor input, String ord, long[] dim, boolean keepdim) {
        return torch.linalg_matrix_norm(input, ord, new LongPointer(dim), keepdim);
    }
    public static Tensor linalg_matrix_norm(Tensor input, String ord) {
        return torch.linalg_matrix_norm(input, ord);
    }
    public static Tensor linalg_inv(Tensor input) { return torch.linalg_inv(input); }
    public static Tensor linalg_pinv(Tensor input, double rcond, boolean hermitian) {
        return torch.linalg_pinv(input, rcond, hermitian);
    }
    public static Tensor linalg_pinv(Tensor input) { return torch.linalg_pinv(input); }
    public static Tensor linalg_det(Tensor input) { return torch.linalg_det(input); }
    public static Tensor linalg_slogdet(Tensor input) { return torch.linalg_slogdet(input); }
    public static Tensor linalg_eigvals(Tensor input) { return torch.linalg_eigvals(input); }
    public static Tensor linalg_eigvalsh(Tensor input, String UPLO) {
        return torch.linalg_eigvalsh(input, UPLO);
    }
    public static Tensor linalg_eigvalsh(Tensor input) {
        return torch.linalg_eigvalsh(input);
    }
    public static Tensor linalg_cholesky(Tensor input, boolean upper) {
        return torch.linalg_cholesky(input, upper);
    }
    public static Tensor linalg_cholesky(Tensor input) {
        return linalg_cholesky(input, false);
    }
    public static Tensor linalg_cholesky_inverse(Tensor input, boolean upper) {
        return torch.linalg_cholesky_inverse(input, upper);
    }
    public static Tensor linalg_cholesky_inverse(Tensor input) {
        return linalg_cholesky_inverse(input, false);
    }
    public static Tensor linalg_solve(Tensor A, Tensor B, boolean left) {
        return torch.linalg_solve(A, B, left);
    }
    public static Tensor linalg_solve(Tensor A, Tensor B) {
        return linalg_solve(A, B, true);
    }
    public static Tensor linalg_solve_triangular(Tensor B, Tensor A, boolean upper, boolean left, boolean unitriangular) {
        return torch.linalg_solve_triangular(B, A, upper, left, unitriangular);
    }
    public static Tensor linalg_svdvals(Tensor A) { return torch.linalg_svdvals(A); }
    public static Tensor linalg_cond(Tensor input, String p) {
        return torch.linalg_cond(input, p);
    }
    public static Tensor linalg_cond(Tensor input) {
        return torch.linalg_cond(input);
    }
    public static Tensor linalg_cross(Tensor input, Tensor other, long dim) {
        return torch.linalg_cross(input, other, dim);
    }
    public static Tensor linalg_cross(Tensor input, Tensor other) {
        return linalg_cross(input, other, -1);
    }
    public static Tensor linalg_householder_product(Tensor input, Tensor tau) {
        return torch.linalg_householder_product(input, tau);
    }
    public static Tensor linalg_ldl_solve(Tensor LD, Tensor B, Tensor pivots, boolean upper, boolean unitriangular) {
        return torch.linalg_ldl_solve(LD, B, pivots, upper, unitriangular);
    }
    public static Tensor linalg_lu_solve(Tensor LU, Tensor pivots, Tensor B, boolean left, boolean adjoint) {
        return torch.linalg_lu_solve(LU, pivots, B, left, adjoint);
    }
    public static Tensor linalg_matmul(Tensor input, Tensor other) {
        return torch.linalg_matmul(input, other);
    }
    public static Tensor linalg_multi_dot(Tensor[] tensors) {
        TensorList list = new TensorList(tensors.length);
        for (int i = 0; i < tensors.length; i++) list.put(i, tensors[i]);
        return torch.linalg_multi_dot(list);
    }
    public static Tensor linalg_matrix_power(Tensor input, long n) {
        return torch.linalg_matrix_power(input, n);
    }
    public static Tensor linalg_matrix_exp(Tensor input) {
        return torch.linalg_matrix_exp(input);
    }
    public static Tensor linalg_matrix_rank(Tensor input, double tol, boolean hermitian) {
        return torch.linalg_matrix_rank(input, tol, hermitian);
    }
    public static Tensor linalg_matrix_rank(Tensor input, double tol) {
        return linalg_matrix_rank(input, tol, false);
    }
    public static Tensor linalg_matrix_rank(Tensor input) {
        return torch.linalg_matrix_rank(input);
    }
    public static Tensor linalg_diagonal(Tensor input, long offset, long dim1, long dim2) {
        return torch.linalg_diagonal(input, offset, dim1, dim2);
    }
    public static Tensor linalg_vander(Tensor x, long N, boolean increasing) {
        return torch.linalg_vander(x, N, increasing);
    }
    public static Tensor linalg_vecdot(Tensor x, Tensor y, long dim) {
        return torch.linalg_vecdot(x, y, dim);
    }
    public static Tensor linalg_tensorinv(Tensor input, long ind) {
        return torch.linalg_tensorinv(input, ind);
    }
    public static Tensor linalg_tensorsolve(Tensor input, Tensor other, long[] dims) {
        return torch.linalg_tensorsolve(input, other, new LongPointer(dims));
    }
    public static Tensor linalg_tensorsolve(Tensor input, Tensor other) {
        return torch.linalg_tensorsolve(input, other);
    }

    // ========================================================================
    // SPECIAL MATH
    // ========================================================================

    public static Tensor special_entr(Tensor input) { return torch.special_entr(input); }
    public static Tensor special_erf(Tensor input) { return torch.special_erf(input); }
    public static Tensor special_erfc(Tensor input) { return torch.special_erfc(input); }
    public static Tensor special_erfcx(Tensor input) { return torch.special_erfcx(input); }
    public static Tensor special_erfinv(Tensor input) { return torch.special_erfinv(input); }
    public static Tensor special_exp2(Tensor input) { return torch.special_exp2(input); }
    public static Tensor special_expit(Tensor input) { return torch.special_expit(input); }
    public static Tensor special_expm1(Tensor input) { return torch.special_expm1(input); }
    public static Tensor special_gammainc(Tensor input, Tensor other) {
        return torch.special_gammainc(input, other);
    }
    public static Tensor special_gammaincc(Tensor input, Tensor other) {
        return torch.special_gammaincc(input, other);
    }
    public static Tensor special_gammaln(Tensor input) { return torch.special_gammaln(input); }
    public static Tensor special_log1p(Tensor input) { return torch.special_log1p(input); }
    public static Tensor special_logit(Tensor input, double eps) { return torch.special_logit(input, eps); }
    public static Tensor special_logit(Tensor input) { return torch.special_logit(input); }
    public static Tensor special_log_ndtr(Tensor input) { return torch.special_log_ndtr(input); }
    public static Tensor special_log_softmax(Tensor input, long dim, org.bytedeco.pytorch.global.torch.ScalarType dtype) {
        return torch.special_log_softmax(input, dim, dtype);
    }
    public static Tensor special_logsumexp(Tensor input, long dim, boolean keepdim) {
        return torch.special_logsumexp(input, dim, keepdim);
    }
    public static Tensor special_modified_bessel_i0(Tensor input) {
        return torch.special_modified_bessel_i0(input);
    }
    public static Tensor special_modified_bessel_i1(Tensor input) {
        return torch.special_modified_bessel_i1(input);
    }
    public static Tensor special_modified_bessel_k0(Tensor input) {
        return torch.special_modified_bessel_k0(input);
    }
    public static Tensor special_modified_bessel_k1(Tensor input) {
        return torch.special_modified_bessel_k1(input);
    }
    public static Tensor special_multigammaln(Tensor input, long p) {
        return torch.special_multigammaln(input, p);
    }
    public static Tensor special_ndtr(Tensor input) { return torch.special_ndtr(input); }
    public static Tensor special_ndtri(Tensor input) { return torch.special_ndtri(input); }
    public static Tensor special_polygamma(Tensor input, long n) {
        return torch.special_polygamma(input, n);
    }
    public static Tensor special_psi(Tensor input) { return torch.special_psi(input); }
    public static Tensor special_round(Tensor input, long decimals) {
        return torch.special_round(input, decimals);
    }
    public static Tensor special_round(Tensor input) {
        return torch.special_round(input);
    }
    public static Tensor special_sinc(Tensor input) { return torch.special_sinc(input); }
    public static Tensor special_softmax(Tensor input, long dim, org.bytedeco.pytorch.global.torch.ScalarType dtype) {
        return torch.special_softmax(input, dim, dtype);
    }
    public static Tensor special_xlog1py(Tensor input, Tensor other) {
        return torch.special_xlog1py(input, other);
    }
    public static Tensor special_xlogy(Tensor input, Tensor other) {
        return torch.special_xlogy(input, other);
    }
    public static Tensor special_zeta(Tensor input, Tensor other) {
        return torch.special_zeta(input, other);
    }

    // ========================================================================
    // BRO / GINI / METRICS (existing helpers, kept for backward compatibility)
    // ========================================================================

    /**
     * BRO: Batch Representation Orthogonality penalty
     * 迫使特征维度之间去相关 (Disentanglement)。
     * Loss = || M^T M - I || (Frobenius Norm)
     * M: Normalized Batch Features [Batch, Dim]
     */
    public static Tensor bro_penalty(Tensor x) {
        long N = x.size(0);
        long D = x.size(1);

        NormalizeFuncOptions opt = new NormalizeFuncOptions();
        opt.p().put(2);
        opt.dim().put(0);
        Tensor xNorm = torch.normalize(x, opt);

        Tensor corr = xNorm.t().matmul(xNorm);

        Tensor eye = torch.eye(D, x.options());

        return corr.sub(eye).norm();
    }

    /**
     * Gini Coefficient
     * 衡量稀疏性 (0 = complete equality/dense, 1 = complete inequality/sparse)
     */
    public static Tensor gini(Tensor x) {
        Tensor xFlat = x.abs().view(-1).add(new Scalar(1e-6));
        long n = xFlat.size(0);

        Tensor xSorted = torch.sort(xFlat).get0();

        Tensor index = torch.arange(new Scalar(1), new Scalar(n + 1), x.options());

        Tensor num = index.mul(xSorted).sum().mul(new Scalar(2.0));
        Tensor den = xSorted.sum().mul(new Scalar((double) n));

        return num.div(den).sub(new Scalar((double) (n + 1) / n));
    }
}