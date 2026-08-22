package org.bytedeco.pytorch.nn;

import org.bytedeco.javacpp.BooleanPointer;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.pytorch.BoolOptional;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.DoubleOptional;
import org.bytedeco.pytorch.GeneratorOptional;
import org.bytedeco.pytorch.LongArrayRef;
import org.bytedeco.pytorch.LongArrayRefOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.LongExpandingArrayOptional;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.LongVectorOptional;
import org.bytedeco.pytorch.MemoryFormatOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.StringViewOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorList;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorOptional;
import org.bytedeco.pytorch.TensorOptionalList;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.enumtype.Conv1dPadding;
import org.bytedeco.pytorch.enumtype.Conv2dPadding;
import org.bytedeco.pytorch.enumtype.Conv3dPadding;
import org.bytedeco.pytorch.enumtype.kCircular;
import org.bytedeco.pytorch.enumtype.kConstant;
import org.bytedeco.pytorch.enumtype.kReflect;
import org.bytedeco.pytorch.enumtype.kReplicate;
import org.bytedeco.pytorch.enumtype.kMean;
import org.bytedeco.pytorch.enumtype.kNone;
import org.bytedeco.pytorch.enumtype.kSum;
import org.bytedeco.pytorch.enumtype.kBatchMean;
import org.bytedeco.pytorch.enumtype.PaddingMode;
import org.bytedeco.pytorch.enumtype.LossReduction;
import org.bytedeco.pytorch.enumtype.KLDivLossReduction;
import org.bytedeco.pytorch.enumtype.EmbeddingBagMode;
import org.bytedeco.pytorch.enumtype.InterpolateMode;
import org.bytedeco.pytorch.enumtype.GridSampleMode;
import org.bytedeco.pytorch.enumtype.GridSamplePaddingMode;
import org.bytedeco.pytorch.enumtype.kZeros;
import org.bytedeco.pytorch.enumtype.kBorder;
import org.bytedeco.pytorch.enumtype.kReflection;
import org.bytedeco.pytorch.enumtype.kBilinear;
import org.bytedeco.pytorch.enumtype.kBicubic;
import org.bytedeco.pytorch.enumtype.kNearest;
import org.bytedeco.pytorch.enumtype.kLinear;
import org.bytedeco.pytorch.enumtype.kArea;
import org.bytedeco.pytorch.enumtype.kNearestExact;
import org.bytedeco.pytorch.enumtype.kTrilinear;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.modules.EmbeddingBagImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.TensorOptional;

/**
 * PyTorch-style functional API.
 *
 * <p>Mirrors {@code torch.nn.functional} from Python PyTorch. All methods delegate
 * to the native {@code at::} and {@code torch::nn::functional} C++ APIs exposed via
 * {@link torch}.
 */
public class Functional {

    private Functional() {}

    // ========================================================================
    // CONVOLUTION
    // ========================================================================

    public static Tensor conv1d(Tensor input, Tensor weight, Tensor bias,
                                long[] stride, long[] padding, long[] dilation, long groups) {
        Conv1dFuncOptions opt = new Conv1dFuncOptions()
                .stride(new LongPointer(stride))
                .padding(new Conv1dPadding(new LongPointer(padding)))
                .dilation(new LongPointer(dilation))
                .groups(groups);
        if (bias != null) opt.bias(bias);
        return torch.conv1d(input, weight, opt);
    }

    public static Tensor conv1d(Tensor input, Tensor weight, long[] stride, long[] padding, long[] dilation, long groups) {
        return conv1d(input, weight, null, stride, padding, dilation, groups);
    }

    public static Tensor conv1d(Tensor input, Tensor weight, Tensor bias,
                                long stride, long padding, long dilation, long groups) {
        return conv1d(input, weight, bias,
                new long[]{stride}, new long[]{padding}, new long[]{dilation}, groups);
    }

    public static Tensor conv1d(Tensor input, Tensor weight,
                                long stride, long padding, long dilation, long groups) {
        return conv1d(input, weight, null,
                new long[]{stride}, new long[]{padding}, new long[]{dilation}, groups);
    }

    public static Tensor conv1d(Tensor input, Tensor weight) {
        return torch.conv1d(input, weight, new Conv1dFuncOptions());
    }

    public static Tensor conv1d(Tensor input, Tensor weight, Conv1dFuncOptions options) {
        return torch.conv1d(input, weight, options);
    }

    public static Tensor conv2d(Tensor input, Tensor weight, Tensor bias,
                                long[] stride, long[] padding, long[] dilation, long groups) {
        Conv2dFuncOptions opt = new Conv2dFuncOptions()
                .stride(new LongPointer(stride))
                .padding(new Conv2dPadding(new LongPointer(padding)))
                .dilation(new LongPointer(dilation))
                .groups(groups);
        if (bias != null) opt.bias(bias);
        return torch.conv2d(input, weight, opt);
    }

    public static Tensor conv2d(Tensor input, Tensor weight, long[] stride, long[] padding, long[] dilation, long groups) {
        return conv2d(input, weight, null, stride, padding, dilation, groups);
    }

    public static Tensor conv2d(Tensor input, Tensor weight, Tensor bias,
                                long stride, long padding, long dilation, long groups) {
        return conv2d(input, weight, bias,
                new long[]{stride, stride}, new long[]{padding, padding}, new long[]{dilation, dilation}, groups);
    }

    public static Tensor conv2d(Tensor input, Tensor weight,
                                long stride, long padding, long dilation, long groups) {
        return conv2d(input, weight, null,
                new long[]{stride, stride}, new long[]{padding, padding}, new long[]{dilation, dilation}, groups);
    }

    public static Tensor conv2d(Tensor input, Tensor weight) {
        return torch.conv2d(input, weight, new Conv2dFuncOptions());
    }

    public static Tensor conv2d(Tensor input, Tensor weight, Conv2dFuncOptions options) {
        return torch.conv2d(input, weight, options);
    }

    public static Tensor conv3d(Tensor input, Tensor weight, Tensor bias,
                                long[] stride, long[] padding, long[] dilation, long groups) {
        Conv3dFuncOptions opt = new Conv3dFuncOptions()
                .stride(new LongPointer(stride))
                .padding(new Conv3dPadding(new LongPointer(padding)))
                .dilation(new LongPointer(dilation))
                .groups(groups);
        if (bias != null) opt.bias(bias);
        return torch.conv3d(input, weight, opt);
    }

    public static Tensor conv3d(Tensor input, Tensor weight, long[] stride, long[] padding, long[] dilation, long groups) {
        return conv3d(input, weight, null, stride, padding, dilation, groups);
    }

    public static Tensor conv3d(Tensor input, Tensor weight, Tensor bias,
                                long stride, long padding, long dilation, long groups) {
        return conv3d(input, weight, bias,
                new long[]{stride, stride, stride}, new long[]{padding, padding, padding}, new long[]{dilation, dilation, dilation}, groups);
    }

    public static Tensor conv3d(Tensor input, Tensor weight,
                                long stride, long padding, long dilation, long groups) {
        return conv3d(input, weight, null,
                new long[]{stride, stride, stride}, new long[]{padding, padding, padding}, new long[]{dilation, dilation, dilation}, groups);
    }

    public static Tensor conv3d(Tensor input, Tensor weight) {
        return torch.conv3d(input, weight, new Conv3dFuncOptions());
    }

    public static Tensor conv3d(Tensor input, Tensor weight, Conv3dFuncOptions options) {
        return torch.conv3d(input, weight, options);
    }

    public static Tensor conv_transpose1d(Tensor input, Tensor weight, Tensor bias,
                                          long[] stride, long[] padding, long[] outputPadding,
                                          long groups, long[] dilation) {
        ConvTranspose1dFuncOptions opt = new ConvTranspose1dFuncOptions()
                .stride(new LongPointer(stride))
                .padding(new LongPointer(padding))
                .output_padding(new LongPointer(outputPadding))
                .groups(groups)
                .dilation(new LongPointer(dilation));
        if (bias != null) opt.bias(bias);
        return torch.conv_transpose1d(input, weight, opt);
    }

    public static Tensor conv_transpose1d(Tensor input, Tensor weight,
                                          long stride, long padding, long outputPadding, long groups, long dilation) {
        return conv_transpose1d(input, weight, null,
                new long[]{stride}, new long[]{padding}, new long[]{outputPadding}, groups,
                new long[]{dilation});
    }

    public static Tensor conv_transpose1d(Tensor input, Tensor weight) {
        return torch.conv_transpose1d(input, weight, new ConvTranspose1dFuncOptions());
    }

    public static Tensor conv_transpose1d(Tensor input, Tensor weight, ConvTranspose1dFuncOptions options) {
        return torch.conv_transpose1d(input, weight, options);
    }

    public static Tensor conv_transpose2d(Tensor input, Tensor weight, Tensor bias,
                                          long[] stride, long[] padding, long[] outputPadding,
                                          long groups, long[] dilation) {
        ConvTranspose2dFuncOptions opt = new ConvTranspose2dFuncOptions()
                .stride(new LongPointer(stride))
                .padding(new LongPointer(padding))
                .output_padding(new LongPointer(outputPadding))
                .groups(groups)
                .dilation(new LongPointer(dilation));
        if (bias != null) opt.bias(bias);
        return torch.conv_transpose2d(input, weight, opt);
    }

    public static Tensor conv_transpose2d(Tensor input, Tensor weight,
                                          long stride, long padding, long outputPadding, long groups, long dilation) {
        return conv_transpose2d(input, weight, null,
                new long[]{stride, stride}, new long[]{padding, padding}, new long[]{outputPadding, outputPadding}, groups,
                new long[]{dilation, dilation});
    }

    public static Tensor conv_transpose2d(Tensor input, Tensor weight) {
        return torch.conv_transpose2d(input, weight, new ConvTranspose2dFuncOptions());
    }

    public static Tensor conv_transpose2d(Tensor input, Tensor weight, ConvTranspose2dFuncOptions options) {
        return torch.conv_transpose2d(input, weight, options);
    }

    public static Tensor conv_transpose3d(Tensor input, Tensor weight, Tensor bias,
                                          long[] stride, long[] padding, long[] outputPadding,
                                          long groups, long[] dilation) {
        ConvTranspose3dFuncOptions opt = new ConvTranspose3dFuncOptions()
                .stride(new LongPointer(stride))
                .padding(new LongPointer(padding))
                .output_padding(new LongPointer(outputPadding))
                .groups(groups)
                .dilation(new LongPointer(dilation));
        if (bias != null) opt.bias(bias);
        return torch.conv_transpose3d(input, weight, opt);
    }

    public static Tensor conv_transpose3d(Tensor input, Tensor weight,
                                          long stride, long padding, long outputPadding, long groups, long dilation) {
        return conv_transpose3d(input, weight, null,
                new long[]{stride, stride, stride}, new long[]{padding, padding, padding},
                new long[]{outputPadding, outputPadding, outputPadding}, groups,
                new long[]{dilation, dilation, dilation});
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

    public static Tensor avg_pool1d(Tensor input, long kernelSize, long stride, long padding,
                                    boolean ceilMode, boolean countIncludePad) {
        return torch.avg_pool1d(input, new long[]{kernelSize},
                stride > 0 ? new long[]{stride} : null,
                new long[]{padding}, ceilMode, countIncludePad);
    }

    public static Tensor avg_pool1d(Tensor input, long kernelSize) {
        return avg_pool1d(input, kernelSize, 0, 0, false, true);
    }

    public static Tensor avg_pool2d(Tensor input, long kernelSizeH, long kernelSizeW,
                                    long strideH, long strideW,
                                    long padH, long padW,
                                    boolean ceilMode, boolean countIncludePad, long divisorOverride) {
        long[] ks = new long[]{kernelSizeH, kernelSizeW};
        long[] ss = (strideH > 0 || strideW > 0) ? new long[]{strideH, strideW} : null;
        long[] ps = new long[]{padH, padW};
        return torch.avg_pool2d(input, ks, ss, ps, ceilMode, countIncludePad, new LongOptional(divisorOverride));
    }

    public static Tensor avg_pool2d(Tensor input, long[] kernelSize, long[] stride, long[] padding) {
        return torch.avg_pool2d(input, kernelSize, stride, padding, false, true, new LongOptional(0));
    }

    public static Tensor avg_pool2d(Tensor input, long kernelSize) {
        return avg_pool2d(input, kernelSize, kernelSize, 0, 0, 0, 0, false, true, 0);
    }

    public static Tensor avg_pool3d(Tensor input, long[] kernelSize, long[] stride, long[] padding,
                                    boolean ceilMode, boolean countIncludePad, long divisorOverride) {
        return torch.avg_pool3d(input, kernelSize, stride, padding, ceilMode, countIncludePad, new LongOptional(divisorOverride));
    }

    public static Tensor avg_pool3d(Tensor input, long[] kernelSize, long[] stride, long[] padding) {
        return avg_pool3d(input, kernelSize, stride, padding, false, true, 0);
    }

    public static Tensor avg_pool3d(Tensor input, long kernelSize) {
        long[] k = new long[]{kernelSize, kernelSize, kernelSize};
        long[] s = new long[]{kernelSize, kernelSize, kernelSize};
        long[] p = new long[]{0, 0, 0};
        return avg_pool3d(input, k, s, p);
    }

    public static Tensor max_pool1d(Tensor input, long kernelSize, long stride, long padding,
                                    long dilation, boolean ceilMode) {
        return torch.max_pool1d(input, new long[]{kernelSize},
                stride > 0 ? new long[]{stride} : null,
                new long[]{padding},
                new long[]{dilation}, ceilMode);
    }

    public static Tensor max_pool1d(Tensor input, long kernelSize) {
        return max_pool1d(input, kernelSize, 0, 0, 1, false);
    }

    public static Tensor max_pool2d(Tensor input, long kernelSizeH, long kernelSizeW,
                                    long strideH, long strideW,
                                    long padH, long padW,
                                    long dilationH, long dilationW, boolean ceilMode) {
        long[] ks = new long[]{kernelSizeH, kernelSizeW};
        long[] ss = (strideH > 0 || strideW > 0) ? new long[]{strideH, strideW} : null;
        long[] ps = new long[]{padH, padW};
        long[] ds = new long[]{dilationH, dilationW};
        MaxPool2dOptions opt = new MaxPool2dOptions(new LongPointer(ks));
        opt.stride(ss != null ? new LongPointer(ss) : new LongPointer());
        opt.padding(new LongPointer(ps));
        opt.dilation(new LongPointer(ds));
        opt.ceil_mode(ceilMode);
        return torch.max_pool2d(input, opt);
    }

    public static Tensor max_pool2d(Tensor input, long[] kernelSize, long[] stride, long[] padding) {
        MaxPool2dOptions opt = new MaxPool2dOptions(new LongPointer(kernelSize));
        opt.stride(stride != null ? new LongPointer(stride) : new LongPointer());
        opt.padding(new LongPointer(padding));
        opt.dilation(new LongPointer(new long[]{1, 1}));
        opt.ceil_mode(false);
        return torch.max_pool2d(input, opt);
    }

    public static Tensor max_pool2d(Tensor input, long kernelSize) {
        return max_pool2d(input, kernelSize, kernelSize, 0L, 0L, 0L, 0L, 1L, 1L, false);
    }

    public static Tensor max_pool3d(Tensor input, long[] kernelSize, long[] stride, long[] padding,
                                    long[] dilation, boolean ceilMode) {
        MaxPool3dOptions opt = new MaxPool3dOptions(new LongPointer(kernelSize));
        opt.stride(stride != null ? new LongPointer(stride) : new LongPointer());
        opt.padding(new LongPointer(padding));
        opt.dilation(dilation != null ? new LongPointer(dilation) : new LongPointer(new long[]{1,1,1}));
        opt.ceil_mode(ceilMode);
        return torch.max_pool3d(input, opt);
    }

    public static Tensor max_pool3d(Tensor input, long[] kernelSize, long[] stride, long[] padding) {
        long[] d = new long[]{1, 1, 1};
        MaxPool3dOptions opt = new MaxPool3dOptions(new LongPointer(kernelSize));
        opt.stride(stride != null ? new LongPointer(stride) : new LongPointer());
        opt.padding(new LongPointer(padding));
        opt.dilation(new LongPointer(d));
        opt.ceil_mode(false);
        return torch.max_pool3d(input, opt);
    }

    public static Tensor max_pool3d(Tensor input, long kernelSize) {
        long[] k = new long[]{kernelSize, kernelSize, kernelSize};
        long[] s = new long[]{kernelSize, kernelSize, kernelSize};
        long[] p = new long[]{0, 0, 0};
        return max_pool3d(input, k, s, p);
    }

    public static Tensor adaptive_avg_pool1d(Tensor input, long[] outputSize) {
        return torch.adaptive_avg_pool1d(input, outputSize);
    }

    public static Tensor adaptive_avg_pool1d(Tensor input, long outputSize) {
        return adaptive_avg_pool1d(input, new long[]{outputSize});
    }

    public static Tensor adaptive_avg_pool2d(Tensor input, long[] outputSize) {
        return torch.adaptive_avg_pool2d(input, outputSize);
    }

    public static Tensor adaptive_avg_pool2d(Tensor input, long outputH, long outputW) {
        return adaptive_avg_pool2d(input, new long[]{outputH, outputW});
    }

    public static Tensor adaptive_avg_pool3d(Tensor input, long[] outputSize) {
        return torch.adaptive_avg_pool3d(input, outputSize);
    }

    public static Tensor adaptive_avg_pool3d(Tensor input, long outputD, long outputH, long outputW) {
        return adaptive_avg_pool3d(input, new long[]{outputD, outputH, outputW});
    }

    public static Tensor adaptive_max_pool1d(Tensor input, long[] outputSize) {
        return torch.adaptive_max_pool1d(input, outputSize).get0();
    }

    public static Tensor adaptive_max_pool1d(Tensor input, long outputSize) {
        return adaptive_max_pool1d(input, new long[]{outputSize});
    }

    public static Tensor adaptive_max_pool2d(Tensor input, long[] outputSize) {
        return torch.adaptive_max_pool2d(input, outputSize).get0();
    }

    public static Tensor adaptive_max_pool2d(Tensor input, long outputH, long outputW) {
        return adaptive_max_pool2d(input, new long[]{outputH, outputW});
    }

    public static Tensor adaptive_max_pool3d(Tensor input, long[] outputSize) {
        return torch.adaptive_max_pool3d(input, outputSize).get0();
    }

    public static Tensor adaptive_max_pool3d(Tensor input, long outputD, long outputH, long outputW) {
        return adaptive_max_pool3d(input, new long[]{outputD, outputH, outputW});
    }

    public static Tensor fractional_max_pool2d(Tensor input, long[] kernelSize, long[] outputSize,
                                                Tensor randomH, Tensor randomW) {
        FractionalMaxPool2dOptions opt = new FractionalMaxPool2dOptions(new LongPointer(kernelSize));
        opt.output_size(new LongExpandingArrayOptional(new LongPointer(outputSize)));
        if (randomH != null && randomW != null) {
            // Combine into single tensor
            TensorVector r = new TensorVector();
            r.push_back(randomH);
            r.push_back(randomW);
            opt._random_samples(torch.cat(r, 0L));
        }
        return torch.fractional_max_pool2d(input, opt);
    }

    public static Tensor fractional_max_pool2d(Tensor input, long kernelSize, long outputSize) {
        return fractional_max_pool2d(input, new long[]{kernelSize, kernelSize},
                new long[]{outputSize, outputSize}, null, null);
    }

    public static Tensor fractional_max_pool3d(Tensor input, long[] kernelSize, long[] outputSize,
                                                Tensor randomH, Tensor randomW) {
        FractionalMaxPool3dOptions opt = new FractionalMaxPool3dOptions(new LongPointer(kernelSize));
        opt.output_size(new LongExpandingArrayOptional(new LongPointer(outputSize)));
        if (randomH != null && randomW != null) {
            // Combine into single tensor
            TensorVector r = new TensorVector();
            r.push_back(randomH);
            r.push_back(randomW);
            opt._random_samples(torch.cat(r, 0L));
        }
        return torch.fractional_max_pool3d(input, opt);
    }

    public static Tensor lp_pool1d(Tensor input, double normType, long kernelSize, long stride, boolean ceilMode) {
        LPPool1dOptions opt = new LPPool1dOptions(normType, new LongPointer(new long[]{kernelSize}));
        opt.stride(new LongPointer(new long[]{stride}));
        opt.ceil_mode(ceilMode);
        return torch.lp_pool1d(input, opt);
    }

    public static Tensor lp_pool2d(Tensor input, double normType, long kernelSize, long stride, boolean ceilMode) {
        LPPool2dOptions opt = new LPPool2dOptions(normType, new LongPointer(new long[]{kernelSize, kernelSize}));
        opt.stride(new LongPointer(new long[]{stride, stride}));
        opt.ceil_mode(ceilMode);
        return torch.lp_pool2d(input, opt);
    }

    public static Tensor lp_pool3d(Tensor input, double normType, long kernelSize, long stride, boolean ceilMode) {
        LPPool3dOptions opt = new LPPool3dOptions(normType, new LongPointer(new long[]{kernelSize, kernelSize, kernelSize}));
        opt.stride(new LongPointer(new long[]{stride, stride, stride}));
        opt.ceil_mode(ceilMode);
        return torch.lp_pool3d(input, opt);
    }

    public static Tensor max_unpool1d(Tensor input, Tensor indices, long[] outputSize) {
        MaxUnpool1dFuncOptions opt = new MaxUnpool1dFuncOptions(new LongPointer(new long[]{1}));
        opt.output_size(new LongVectorOptional(new LongVector(outputSize)));
        return torch.max_unpool1d(input, indices, opt);
    }

    public static Tensor max_unpool1d(Tensor input, Tensor indices, long outputSize) {
        return max_unpool1d(input, indices, new long[]{outputSize});
    }

    public static Tensor max_unpool2d(Tensor input, Tensor indices, long[] outputSize) {
        MaxUnpool2dFuncOptions opt = new MaxUnpool2dFuncOptions(new LongPointer(new long[]{1, 1}));
        opt.output_size(new LongVectorOptional(new LongVector(outputSize)));
        return torch.max_unpool2d(input, indices, opt);
    }

    public static Tensor max_unpool2d(Tensor input, Tensor indices, long outputH, long outputW) {
        return max_unpool2d(input, indices, new long[]{outputH, outputW});
    }

    public static Tensor max_unpool3d(Tensor input, Tensor indices, long[] outputSize) {
        MaxUnpool3dFuncOptions opt = new MaxUnpool3dFuncOptions(new LongPointer(new long[]{1, 1, 1}));
        opt.output_size(new LongVectorOptional(new LongVector(outputSize)));
        return torch.max_unpool3d(input, indices, opt);
    }

    public static Tensor max_unpool3d(Tensor input, Tensor indices, long outputD, long outputH, long outputW) {
        return max_unpool3d(input, indices, new long[]{outputD, outputH, outputW});
    }

    // ========================================================================
    // PADDING
    // ========================================================================

    public static Tensor reflection_pad1d(Tensor input, long[] padding) {
        return torch.reflection_pad1d(input, padding);
    }

    public static Tensor reflection_pad1d(Tensor input, long padding) {
        return reflection_pad1d(input, new long[]{padding});
    }

    public static Tensor reflection_pad2d(Tensor input, long[] padding) {
        return torch.reflection_pad2d(input, padding);
    }

    public static Tensor reflection_pad2d(Tensor input, long left, long right, long top, long bottom) {
        return reflection_pad2d(input, new long[]{left, right, top, bottom});
    }

    public static Tensor reflection_pad3d(Tensor input, long[] padding) {
        return torch.reflection_pad3d(input, padding);
    }

    public static Tensor reflection_pad3d(Tensor input, long left, long right, long top, long bottom, long front, long back) {
        return reflection_pad3d(input, new long[]{left, right, top, bottom, front, back});
    }

    public static Tensor replication_pad1d(Tensor input, long[] padding) {
        return torch.replication_pad1d(input, padding);
    }

    public static Tensor replication_pad1d(Tensor input, long padding) {
        return replication_pad1d(input, new long[]{padding});
    }

    public static Tensor replication_pad2d(Tensor input, long[] padding) {
        return torch.replication_pad2d(input, padding);
    }

    public static Tensor replication_pad2d(Tensor input, long left, long right, long top, long bottom) {
        return replication_pad2d(input, new long[]{left, right, top, bottom});
    }

    public static Tensor replication_pad3d(Tensor input, long[] padding) {
        return torch.replication_pad3d(input, padding);
    }

    public static Tensor replication_pad3d(Tensor input, long left, long right, long top, long bottom, long front, long back) {
        return replication_pad3d(input, new long[]{left, right, top, bottom, front, back});
    }

    public static Tensor circular_pad1d(Tensor input, long[] padding) {
        PadFuncOptions opt = new PadFuncOptions(new LongVector(padding));
        opt.mode(new PaddingMode(new kCircular()));
        return torch.pad(input, opt);
    }
    public static Tensor circular_pad1d(Tensor input, long padding) {
        return circular_pad1d(input, new long[]{padding});
    }

    public static Tensor circular_pad1d(Tensor input, PadFuncOptions opt) {
        return circular_pad1d(input, opt);
    }
    public static Tensor circular_pad2d(Tensor input, long[] padding) {
        PadFuncOptions opt = new PadFuncOptions(new LongVector(padding));
        opt.mode(new PaddingMode(new kCircular()));
        return torch.pad(input, opt);
    }

    public static Tensor circular_pad2d(Tensor input, PadFuncOptions opt) {
        return torch.pad(input, opt);
    }
    public static Tensor circular_pad2d(Tensor input, long left, long right, long top, long bottom) {
        return circular_pad2d(input, new long[]{left, right, top, bottom});
    }
    public static Tensor circular_pad3d(Tensor input, long[] padding) {
        PadFuncOptions opt = new PadFuncOptions(new LongVector(padding));
        opt.mode(new PaddingMode(new kCircular()));
        return torch.pad(input, opt);
    }

    public static Tensor circular_pad3d(Tensor input, PadFuncOptions opt) {
        return torch.pad(input, opt);
    }
    public static Tensor circular_pad3d(Tensor input, long left, long right, long top, long bottom, long front, long back) {
        return circular_pad3d(input, new long[]{left, right, top, bottom, front, back});
    }

    public static Tensor constant_pad_nd(Tensor input, long[] pad, Scalar value) {
        return torch.constant_pad_nd(input, pad, value);
    }

    public static Tensor constant_pad_nd(Tensor input, long[] pad, double value) {
        return constant_pad_nd(input, pad, new Scalar(value));
    }

    public static Tensor pad(Tensor input, long[] pad, String mode, Scalar value) {
        PadFuncOptions opt = new PadFuncOptions(new LongVector(pad));
        opt.mode(toPaddingMode(mode));
        opt.value(value.toDouble());
        return torch.pad(input, opt);
    }


    public static Tensor pad(Tensor input, long[] pad, String mode, double value) {
        return pad(input, pad, mode, new Scalar(value));
    }

    private static InterpolateMode toInterpolateMode(String name) {
        switch (name) {
            case "nearest":  return new InterpolateMode(new kNearest());
            case "linear":   return new InterpolateMode(new kLinear());
            case "bilinear": return new InterpolateMode(new kBilinear());
            case "bicubic":  return new InterpolateMode(new kBicubic());
            case "trilinear":return new InterpolateMode(new kTrilinear());
            case "area":     return new InterpolateMode(new kArea());
            case "nearest-exact": return new InterpolateMode(new kNearestExact());
            default: return new InterpolateMode(new kNearest());
        }
    }

    private static GridSampleMode toGridSampleMode(String name) {
        switch (name) {
            case "bilinear": return new GridSampleMode(new kBilinear());
            case "bicubic":  return new GridSampleMode(new kBicubic());
            case "nearest":  return new GridSampleMode(new kNearest());
            default: return new GridSampleMode(new kBilinear());
        }
    }

    private static GridSamplePaddingMode toGridSamplePaddingMode(String name) {
        switch (name) {
            case "zeros":     return new GridSamplePaddingMode(new kZeros());
            case "border":    return new GridSamplePaddingMode(new kBorder());
            case "reflection":return new GridSamplePaddingMode(new kReflection());
            default: return new GridSamplePaddingMode(new kZeros());
        }
    }

    private static PaddingMode toPaddingMode(String name) {
        switch (name) {
            case "constant": return new PaddingMode(new kConstant());
            case "reflect":  return new PaddingMode(new kReflect());
            case "replicate":return new PaddingMode(new kReplicate());
            case "circular": return new PaddingMode(new kCircular());
            default: return new PaddingMode(new kConstant());
        }
    }

    // ========================================================================
    // ACTIVATIONS
    // ========================================================================

    public static Tensor relu(Tensor input) {
        return torch.relu(input);
    }

    public static Tensor relu6(Tensor input) {
        return torch.relu6(input, new ReLU6Options());
    }

    public static Tensor leaky_relu(Tensor input) {
        return torch.leaky_relu(input, new LeakyReLUOptions());
    }

    public static Tensor leaky_relu(Tensor input, double negativeSlope) {
        LeakyReLUOptions opt = new LeakyReLUOptions().negative_slope(negativeSlope);
        return torch.leaky_relu(input, opt);
    }

    public static Tensor prelu(Tensor input, Tensor weight) {
        return torch.prelu(input, weight);
    }

    public static Tensor rrelu(Tensor input) {
        return torch.rrelu(input, new RReLUFuncOptions());
    }

    public static Tensor rrelu(Tensor input, double lower, double upper, boolean training) {
        RReLUFuncOptions opt = new RReLUFuncOptions().lower(lower).upper(upper).training(training);
        return torch.rrelu(input, opt);
    }

    public static Tensor elu(Tensor input) {
        return torch.elu(input, new ELUOptions());
    }

    public static Tensor elu(Tensor input, double alpha) {
        ELUOptions opt = new ELUOptions().alpha(alpha);
        return torch.elu(input, opt);
    }

    public static Tensor selu(Tensor input) {
        return torch.selu(input, new SELUOptions());
    }

    public static Tensor celu(Tensor input) {
        return torch.celu(input, new org.bytedeco.pytorch.nn.options.CELUOptions());
    }

    public static Tensor celu(Tensor input, double alpha) {
        org.bytedeco.pytorch.nn.options.CELUOptions opt = new org.bytedeco.pytorch.nn.options.CELUOptions().alpha(alpha);
        return torch.celu(input, opt);
    }

    public static Tensor silu(Tensor input) {
        return torch.silu(input);
    }

    public static Tensor mish(Tensor input) {
        return torch.mish(input);
    }

    public static Tensor gelu(Tensor input) {
        return torch.gelu(input, new GELUOptions());
    }

    public static Tensor gelu(Tensor input, String approximate) {
        GELUOptions opt = new GELUOptions().approximate(approximate);
        return torch.gelu(input, opt);
    }

    public static Tensor hardtanh(Tensor input) {
        return torch.hardtanh(input, new HardtanhOptions());
    }

    public static Tensor hardtanh(Tensor input, double minVal, double maxVal) {
        HardtanhOptions opt = new HardtanhOptions().min_val(minVal).max_val(maxVal);
        return torch.hardtanh(input, opt);
    }

    public static Tensor hardshrink(Tensor input) {
        return torch.hardshrink(input, new HardshrinkOptions());
    }

    public static Tensor hardshrink(Tensor input, double lambda) {
        HardshrinkOptions opt = new HardshrinkOptions().lambda(lambda);
        return torch.hardshrink(input, opt);
    }

    public static Tensor softshrink(Tensor input) {
        return torch.softshrink(input, new SoftshrinkOptions());
    }

    public static Tensor softshrink(Tensor input, double lambda) {
        SoftshrinkOptions opt = new SoftshrinkOptions().lambda(lambda);
        return torch.softshrink(input, opt);
    }

    public static Tensor softplus(Tensor input) {
        return torch.softplus(input, new SoftplusOptions());
    }

    public static Tensor softplus(Tensor input, double beta, double threshold) {
        SoftplusOptions opt = new SoftplusOptions().beta(beta).threshold(threshold);
        return torch.softplus(input, opt);
    }

    public static Tensor softsign(Tensor input) {
        return torch.softsign(input);
    }

    public static Tensor tanhshrink(Tensor input) {
        return torch.tanhshrink(input);
    }

    public static Tensor threshold(Tensor input, double threshold, double value) {
        ThresholdOptions opt = new ThresholdOptions(threshold, value);
        return torch.threshold(input, opt);
    }

    public static Tensor glu(Tensor input) {
        return torch.glu(input, new GLUOptions());
    }

    public static Tensor glu(Tensor input, long dim) {
        GLUOptions opt = new GLUOptions().dim(dim);
        return torch.glu(input, opt);
    }

    public static Tensor logsigmoid(Tensor input) {
        return torch.logsigmoid(input);
    }

    public static Tensor softmax(Tensor input, long dim) {
        SoftmaxFuncOptions opt = new SoftmaxFuncOptions(dim);
        return torch.softmax(input, opt);
    }

    public static Tensor softmax(Tensor input, long dim, ScalarTypeOptional dtype) {
        SoftmaxFuncOptions opt = new SoftmaxFuncOptions(dim);
        opt.dtype(dtype);
        return torch.softmax(input, opt);
    }

    public static Tensor softmin(Tensor input, long dim) {
        SoftminFuncOptions opt = new SoftminFuncOptions(dim);
        return torch.softmin(input, opt);
    }

    public static Tensor softmin(Tensor input, long dim, ScalarTypeOptional dtype) {
        SoftminFuncOptions opt = new SoftminFuncOptions(dim);
        opt.dtype(dtype);
        return torch.softmin(input, opt);
    }

    public static Tensor log_softmax(Tensor input, long dim) {
        LogSoftmaxFuncOptions opt = new LogSoftmaxFuncOptions(dim);
        return torch.log_softmax(input, opt);
    }

    public static Tensor log_softmax(Tensor input, long dim, ScalarTypeOptional dtype) {
        LogSoftmaxFuncOptions opt = new LogSoftmaxFuncOptions(dim);
        opt.dtype(dtype);
        return torch.log_softmax(input, opt);
    }

    public static Tensor gumbel_softmax(Tensor logits, double tau, boolean hard, int dim) {
        GumbelSoftmaxFuncOptions opt = new GumbelSoftmaxFuncOptions().tau(tau).hard(hard).dim(dim);
        return torch.gumbel_softmax(logits, opt);
    }

    public static Tensor gumbel_softmax(Tensor logits, double tau, boolean hard) {
        return gumbel_softmax(logits, tau, hard, -1);
    }

    public static Tensor tanh(Tensor input) {
        return torch.tanh(input);
    }

    public static Tensor sigmoid(Tensor input) {
        return torch.sigmoid(input);
    }

    public static Tensor hard_sigmoid(Tensor input) {
        return torch.hardsigmoid(input);
    }
    public static Tensor hardsigmoid(Tensor input) { return hard_sigmoid(input); }

    public static Tensor hard_swish(Tensor input) {
        return torch.hardswish(input);
    }
    public static Tensor hardswish(Tensor input) { return hard_swish(input); }

    // ========================================================================
    // NORMALIZATION
    // ========================================================================

    public static Tensor batch_norm(Tensor input, Tensor runningMean, Tensor runningVar,
                                    boolean training, Tensor weight, Tensor bias,
                                    double momentum, double eps) {
        BatchNormFuncOptions opt = new BatchNormFuncOptions()
                .training(training)
                .momentum(momentum)
                .eps(eps);
        if (weight != null) opt.weight(weight);
        if (bias != null) opt.bias(bias);
        return torch.batch_norm(input, runningMean, runningVar, opt);
    }

    public static Tensor batch_norm(Tensor input, Tensor runningMean, Tensor runningVar, boolean training) {
        return batch_norm(input, runningMean, runningVar, training, null, null, 0.1, 1e-5);
    }

    public static Tensor batch_norm(Tensor input, Tensor runningMean, Tensor runningVar) {
        return batch_norm(input, runningMean, runningVar, false);
    }

    public static Tensor instance_norm(Tensor input, Tensor runningMean, Tensor runningVar,
                                       boolean useInputStats, Tensor weight, Tensor bias,
                                       boolean training, double momentum, double eps) {
        InstanceNormFuncOptions opt = new InstanceNormFuncOptions()
                .use_input_stats(useInputStats)
                .momentum(momentum)
                .eps(eps);
        if (runningMean != null) opt.running_mean(runningMean);
        if (runningVar != null) opt.running_var(runningVar);
        if (weight != null) opt.weight(weight);
        if (bias != null) opt.bias(bias);
        return torch.instance_norm(input, opt);
    }

    public static Tensor instance_norm(Tensor input, boolean useInputStats, double momentum, double eps) {
        return instance_norm(input, null, null, useInputStats, null, null, true, momentum, eps);
    }

    public static Tensor layer_norm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias, double eps) {
        LayerNormFuncOptions opt = new LayerNormFuncOptions(new LongVector(normalizedShape));
        if (weight != null) opt.weight(weight);
        if (bias != null) opt.bias(bias);
        opt.eps(eps);
        return torch.layer_norm(input, opt);
    }

    public static Tensor layer_norm(Tensor input, long[] normalizedShape, Tensor weight, Tensor bias) {
        return layer_norm(input, normalizedShape, weight, bias, 1e-5);
    }

    public static Tensor layer_norm(Tensor input, long[] normalizedShape) {
        return layer_norm(input, normalizedShape, null, null, 1e-5);
    }

    public static Tensor group_norm(Tensor input, long numGroups, Tensor weight, Tensor bias, double eps) {
        GroupNormFuncOptions opt = new GroupNormFuncOptions(numGroups);
        if (weight != null) opt.weight(weight);
        if (bias != null) opt.bias(bias);
        opt.eps(eps);
        return torch.group_norm(input, opt);
    }

    public static Tensor group_norm(Tensor input, long numGroups, double eps) {
        return group_norm(input, numGroups, null, null, eps);
    }

    public static Tensor group_norm(Tensor input, long numGroups) {
        return group_norm(input, numGroups, null, null, 1e-5);
    }

    public static Tensor local_response_norm(Tensor input, long size) {
        LocalResponseNormOptions opt = new LocalResponseNormOptions(size);
        return torch.local_response_norm(input, opt);
    }

    public static Tensor local_response_norm(Tensor input, long size, double alpha, double beta, double k) {
        LocalResponseNormOptions opt = new LocalResponseNormOptions(size);
        opt.alpha(alpha).beta(beta).k(k);
        return torch.local_response_norm(input, opt);
    }

    public static Tensor normalize(Tensor input, double p, long dim, double eps) {
        NormalizeFuncOptions opt = new NormalizeFuncOptions().p(p).dim(dim).eps(eps);
        return torch.normalize(input, opt);
    }

    public static Tensor normalize(Tensor input, double p, long dim) {
        return normalize(input, p, dim, 1e-12);
    }

    public static Tensor normalize(Tensor input, double p) {
        return normalize(input, p, 1, 1e-12);
    }

    public static Tensor rms_norm(Tensor input, long[] normalizedShape, Tensor weight, double eps) {
        if (weight != null) {
            return torch.rms_norm(input, normalizedShape, new TensorOptional(weight), new DoubleOptional(eps));
        } else {
            return torch.rms_norm(input, normalizedShape);
        }
    }

    public static Tensor rms_norm(Tensor input, long[] normalizedShape) {
        return rms_norm(input, normalizedShape, null, 1e-6);
    }

    // ========================================================================
    // LINEAR
    // ========================================================================

    public EmbeddingImpl from_pretrained_embedding(Tensor embeddings, EmbeddingFromPretrainedOptions options){
        return EmbeddingImpl.from_pretrained(embeddings,options);
    }

    public EmbeddingImpl from_pretrained_embedding(Tensor embeddings){
        return EmbeddingImpl.from_pretrained(embeddings);
    }
    public EmbeddingBagImpl from_pretrained_embedding(Tensor embeddings, EmbeddingBagFromPretrainedOptions options){
        return EmbeddingBagImpl.from_pretrained(embeddings,options);
    }

    public EmbeddingBagImpl from_pretrained_embedding_bag(Tensor embeddings){
        return EmbeddingBagImpl.from_pretrained(embeddings);
    }

    public static Tensor linear(Tensor input, Tensor weight, Tensor bias) {
        return torch.linear(input, weight, bias);
    }

    public static Tensor linear(Tensor input, Tensor weight) {
        return torch.linear(input, weight);
    }

    public static Tensor bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor bias) {
        return torch.bilinear(input1, input2, weight, bias);
    }

    // ========================================================================
    // DROPOUT
    // ========================================================================

    public static Tensor dropout(Tensor input, double p, boolean training) {
        return torch.dropout(input, p, training);
    }

    public static Tensor dropout(Tensor input, double p) {
        return dropout(input, p, true);
    }

    public static Tensor dropout(Tensor input) {
        return dropout(input, 0.5, true);
    }

    public static Tensor dropout1d(Tensor input, double p, boolean training) {
        DropoutFuncOptions opt = new DropoutFuncOptions().p(p).training(training);
        // dropout1d uses Dropout2dFuncOptions in PyTorch C++
        return torch.dropout(input, p, training);
    }

    public static Tensor dropout1d(Tensor input, double p) {
        return dropout1d(input, p, true);
    }

    public static Tensor dropout2d(Tensor input, double p, boolean training) {
        return torch.dropout(input, p, training);
    }

    public static Tensor dropout2d(Tensor input, double p) {
        return dropout2d(input, p, true);
    }

    public static Tensor dropout3d(Tensor input, double p, boolean training) {
        return torch.dropout(input, p, training);
    }

    public static Tensor dropout3d(Tensor input, double p) {
        return dropout3d(input, p, true);
    }

    public static Tensor alpha_dropout(Tensor input, double p, boolean training) {
        return torch.alpha_dropout(input, p, training);
    }

    public static Tensor alpha_dropout(Tensor input, double p) {
        return alpha_dropout(input, p, true);
    }

    public static Tensor alpha_dropout(Tensor input) {
        return alpha_dropout(input, 0.5, true);
    }

    public static Tensor feature_alpha_dropout(Tensor input, double p, boolean training) {
        FeatureAlphaDropoutFuncOptions opt = new FeatureAlphaDropoutFuncOptions().p(p).training(training);
        return torch.feature_alpha_dropout(input, opt);
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

    public static Tensor binary_cross_entropy(Tensor input, Tensor target, Tensor weight, long reduction) {
        BCELossOptions opt = new BCELossOptions().weight(weight).reduction(toLossReduction(reduction));
        return torch.binary_cross_entropy(input, target, opt);
    }

    public static Tensor binary_cross_entropy(Tensor input, Tensor target, Tensor weight) {
        return binary_cross_entropy(input, target, weight, 1);
    }

    public static Tensor binary_cross_entropy(Tensor input, Tensor target) {
        return binary_cross_entropy(input, target, null, 1);
    }

    public static Tensor binary_cross_entropy_with_logits(Tensor input, Tensor target, Tensor weight,
                                                          Tensor posWeight, long reduction) {
        BCEWithLogitsLossOptions opt = new BCEWithLogitsLossOptions()
                .weight(weight).pos_weight(posWeight).reduction(toLossReduction(reduction));
        return torch.binary_cross_entropy_with_logits(input, target, opt);
    }

    public static Tensor binary_cross_entropy_with_logits(Tensor input, Tensor target) {
        return binary_cross_entropy_with_logits(input, target, null, null, 1);
    }

    public static Tensor l1_loss(Tensor input, Tensor target, long reduction) {
        L1LossOptions opt = new L1LossOptions().reduction(toLossReduction(reduction));
        return torch.l1_loss(input, target, opt);
    }

    public static Tensor l1_loss(Tensor input, Tensor target) {
        return l1_loss(input, target, 1);
    }

    public static Tensor mse_loss(Tensor input, Tensor target, long reduction) {
        MSELossOptions opt = new MSELossOptions().reduction(toLossReduction(reduction));
        return torch.mse_loss(input, target, opt);
    }

    public static Tensor mse_loss(Tensor input, Tensor target) {
        return mse_loss(input, target, 1);
    }

    public static Tensor smooth_l1_loss(Tensor input, Tensor target, long reduction, double beta) {
        SmoothL1LossOptions opt = new SmoothL1LossOptions().reduction(toLossReduction(reduction)).beta(new DoubleOptional(beta));
        return torch.smooth_l1_loss(input, target, opt);
    }

    public static Tensor smooth_l1_loss(Tensor input, Tensor target, long reduction) {
        return smooth_l1_loss(input, target, reduction, 1.0);
    }

    public static Tensor smooth_l1_loss(Tensor input, Tensor target) {
        return smooth_l1_loss(input, target, 1);
    }

    public static Tensor huber_loss(Tensor input, Tensor target, long reduction, double delta) {
        org.bytedeco.pytorch.nn.options.HuberLossOptions opt =
                new org.bytedeco.pytorch.nn.options.HuberLossOptions()
                .reduction(toLossReduction(reduction)).delta(delta);
        return torch.huber_loss(input, target, opt);
    }

    public static Tensor huber_loss(Tensor input, Tensor target, long reduction) {
        return huber_loss(input, target, reduction, 1.0);
    }

    public static Tensor huber_loss(Tensor input, Tensor target) {
        return huber_loss(input, target, 1);
    }

    public static Tensor nll_loss(Tensor input, Tensor target, Tensor weight, long ignoreIndex, long reduction) {
        NLLLossOptions opt = new NLLLossOptions()
                .weight(weight).ignore_index(ignoreIndex).reduction(toLossReduction(reduction));
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

    public static Tensor nll_loss2d(Tensor input, Tensor target, Tensor weight, long ignoreIndex, long reduction) {
        // nll_loss2d uses the same backend as nll_loss in C++
        NLLLossOptions opt = new NLLLossOptions()
                .weight(weight).ignore_index(ignoreIndex).reduction(toLossReduction(reduction));
        return torch.nll_loss(input, target, opt);
    }

    public static Tensor cross_entropy(Tensor input, Tensor target, Tensor weight, long ignoreIndex, long reduction) {
        CrossEntropyLossOptions opt = new CrossEntropyLossOptions()
                .weight(weight).ignore_index(ignoreIndex).reduction(toLossReduction(reduction));
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

    public static Tensor kl_div(Tensor input, Tensor target, long reduction, boolean logTarget) {
        KLDivLossOptions opt = new KLDivLossOptions().reduction(toKLDivLossReduction(reduction)).log_target(logTarget);
        return torch.kl_div(input, target, opt);
    }

    public static Tensor kl_div(Tensor input, Tensor target, long reduction) {
        return kl_div(input, target, reduction, false);
    }

    public static Tensor kl_div(Tensor input, Tensor target) {
        return kl_div(input, target, 1);
    }

    public static Tensor poisson_nll_loss(Tensor input, Tensor target, boolean logInput, boolean full,
                                          long reduction) {
        PoissonNLLLossOptions opt = new PoissonNLLLossOptions()
                .log_input(logInput).full(full).reduction(toLossReduction(reduction));
        return torch.poisson_nll_loss(input, target, opt);
    }

    public static Tensor poisson_nll_loss(Tensor input, Tensor target) {
        return poisson_nll_loss(input, target, true, false, 1);
    }

    public static Tensor cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target,
                                               double margin, long reduction) {
        CosineEmbeddingLossOptions opt = new CosineEmbeddingLossOptions()
                .margin(margin).reduction(toLossReduction(reduction));
        return torch.cosine_embedding_loss(input1, input2, target, opt);
    }

    public static Tensor cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, double margin) {
        return cosine_embedding_loss(input1, input2, target, margin, 1);
    }

    public static Tensor cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target) {
        return cosine_embedding_loss(input1, input2, target, 0.0, 1);
    }

    public static Tensor margin_ranking_loss(Tensor input1, Tensor input2, Tensor target,
                                             double margin, long reduction) {
        MarginRankingLossOptions opt = new MarginRankingLossOptions()
                .margin(margin).reduction(toLossReduction(reduction));
        return torch.margin_ranking_loss(input1, input2, target, opt);
    }

    public static Tensor margin_ranking_loss(Tensor input1, Tensor input2, Tensor target) {
        return margin_ranking_loss(input1, input2, target, 0.0, 1);
    }

    public static Tensor multilabel_margin_loss(Tensor input, Tensor target, long reduction) {
        MultiLabelMarginLossOptions opt = new MultiLabelMarginLossOptions().reduction(toLossReduction(reduction));
        return torch.multilabel_margin_loss(input, target, opt);
    }

    public static Tensor multilabel_margin_loss(Tensor input, Tensor target) {
        return multilabel_margin_loss(input, target, 1L);
    }

    public static Tensor multilabel_soft_margin_loss(Tensor input, Tensor target, Tensor weight,
                                                     long reduction) {
        MultiLabelSoftMarginLossOptions opt = new MultiLabelSoftMarginLossOptions()
                .weight(weight).reduction(toLossReduction(reduction));
        return torch.multilabel_soft_margin_loss(input, target, opt);
    }

    public static Tensor multilabel_soft_margin_loss(Tensor input, Tensor target) {
        return multilabel_soft_margin_loss(input, target, null, 1);
    }

    public static Tensor multi_margin_loss(Tensor input, Tensor target, long p, double margin,
                                          Tensor weight, long reduction) {
        MultiMarginLossOptions opt = new MultiMarginLossOptions()
                .p(p).margin(margin).weight(weight).reduction(toLossReduction(reduction));
        return torch.multi_margin_loss(input, target, opt);
    }

    public static Tensor multi_margin_loss(Tensor input, Tensor target, long p, double margin) {
        return multi_margin_loss(input, target, p, margin, null, 1);
    }

    public static Tensor hinge_embedding_loss(Tensor input, Tensor target, double margin, long reduction) {
        HingeEmbeddingLossOptions opt = new HingeEmbeddingLossOptions()
                .margin(margin).reduction(toLossReduction(reduction));
        return torch.hinge_embedding_loss(input, target, opt);
    }

    public static Tensor hinge_embedding_loss(Tensor input, Tensor target) {
        return hinge_embedding_loss(input, target, 1.0, 1);
    }

    public static Tensor triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative,
                                             double margin, long p, double eps, boolean swap, long reduction) {
        TripletMarginLossOptions opt = new TripletMarginLossOptions()
                .margin(margin).p(p).eps(eps).swap(swap).reduction(toLossReduction(reduction));
        return torch.triplet_margin_loss(anchor, positive, negative, opt);
    }

    public static Tensor triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, double margin) {
        return triplet_margin_loss(anchor, positive, negative, margin, 2, 1e-6, false, 1);
    }

    public static Tensor triplet_margin_with_distance_loss(Tensor anchor, Tensor positive, Tensor negative,
                                                          double margin) {
        TripletMarginWithDistanceLossOptions opt = new TripletMarginWithDistanceLossOptions().margin(margin);
        return torch.triplet_margin_with_distance_loss(anchor, positive, negative, opt);
    }

    public static Tensor triplet_margin_with_distance_loss(Tensor anchor, Tensor positive, Tensor negative) {
        return triplet_margin_with_distance_loss(anchor, positive, negative, 1.0);
    }

    public static Tensor soft_margin_loss(Tensor input, Tensor target, long reduction) {
        SoftMarginLossOptions opt = new SoftMarginLossOptions().reduction(toLossReduction(reduction));
        return torch.soft_margin_loss(input, target, opt);
    }

    public static Tensor soft_margin_loss(Tensor input, Tensor target) {
        return soft_margin_loss(input, target, 1);
    }

    public static Tensor ctc_loss(Tensor logProbs, Tensor targets, Tensor inputLengths, Tensor targetLengths,
                                  long blank, long reduction) {
        CTCLossOptions opt = new CTCLossOptions().blank(blank).reduction(toLossReduction(reduction));
        return torch.ctc_loss(logProbs, targets, inputLengths, targetLengths, opt);
    }

    public static Tensor ctc_loss(Tensor logProbs, Tensor targets, Tensor inputLengths, Tensor targetLengths,
                                  long blank) {
        return ctc_loss(logProbs, targets, inputLengths, targetLengths, blank, 1);
    }

    public static Tensor gaussian_nll_loss(Tensor input, Tensor target, Tensor var,
                                           boolean full, double eps, long reduction) {
        // gaussian_nll_loss not directly exposed in torch::nn::functional in C++
        // Approximate via manual computation
        Tensor diff = input.sub(target);
        Tensor log_var = var.log();
        Tensor loss = diff.mul(diff).div(var.add(new Scalar(eps))).add(log_var).add(new Scalar(Math.log(2 * Math.PI)));
        if (reduction == 1) return loss.mean();
        if (reduction == 0) return loss.sum();
        return loss.mean();
    }

    // ========================================================================
    // DISTANCE
    // ========================================================================

    public static Tensor cosine_similarity(Tensor x1, Tensor x2, long dim, double eps) {
        CosineSimilarityOptions opt = new CosineSimilarityOptions().dim(dim).eps(eps);
        return torch.cosine_similarity(x1, x2, opt);
    }

    public static Tensor cosine_similarity(Tensor x1, Tensor x2, long dim) {
        return cosine_similarity(x1, x2, dim, 1e-8);
    }

    public static Tensor cosine_similarity(Tensor x1, Tensor x2) {
        return cosine_similarity(x1, x2, 1, 1e-8);
    }

    public static Tensor pairwise_distance(Tensor x1, Tensor x2, double p, double eps, boolean keepdim) {
        PairwiseDistanceOptions opt = new PairwiseDistanceOptions().p(p).eps(eps).keepdim(keepdim);
        return torch.pairwise_distance(x1, x2, opt);
    }

    public static Tensor pairwise_distance(Tensor x1, Tensor x2, double p, double eps) {
        return pairwise_distance(x1, x2, p, eps, false);
    }

    public static Tensor pairwise_distance(Tensor x1, Tensor x2) {
        return pairwise_distance(x1, x2, 2.0, 1e-6, false);
    }

    public static Tensor lp_distance(Tensor x1, Tensor x2, double p) {
        return pairwise_distance(x1, x2, p, 1e-6, false);
    }

    // ========================================================================
    // VISION
    // ========================================================================

    public static Tensor pixel_shuffle(Tensor input, long upscaleFactor) {
        return torch.pixel_shuffle(input, upscaleFactor);
    }

    public static Tensor pixel_unshuffle(Tensor input, long downscaleFactor) {
        return torch.pixel_unshuffle(input, downscaleFactor);
    }

    public static Tensor interpolate(Tensor input, long[] size, String mode, boolean alignCorners) {
        InterpolateFuncOptions opt = new InterpolateFuncOptions()
                .size(new LongVectorOptional(new LongVector(size)))
                .mode(toInterpolateMode(mode))
                .align_corners(alignCorners ? new BoolOptional(true) : new BoolOptional(false));
        return torch.interpolate(input, opt);
    }

    public static Tensor interpolate(Tensor input, long[] size, String mode) {
        return interpolate(input, size, mode, false);
    }

    public static Tensor interpolate(Tensor input, long sizeH, long sizeW, String mode, boolean alignCorners) {
        return interpolate(input, new long[]{sizeH, sizeW}, mode, alignCorners);
    }

    public static Tensor interpolate(Tensor input, long sizeH, long sizeW, String mode) {
        return interpolate(input, sizeH, sizeW, mode, false);
    }

    public static Tensor upsample_bilinear(Tensor input, long[] outputSize, boolean alignCorners) {
        return torch.upsample_bilinear2d(input, outputSize, alignCorners);
    }

    public static Tensor upsample_bilinear(Tensor input, long sizeH, long sizeW, boolean alignCorners) {
        return upsample_bilinear(input, new long[]{sizeH, sizeW}, alignCorners);
    }

    public static Tensor upsample_nearest(Tensor input, long[] outputSize) {
        return torch.upsample_nearest2d(input, outputSize);
    }

    public static Tensor upsample_nearest(Tensor input, long sizeH, long sizeW) {
        return upsample_nearest(input, new long[]{sizeH, sizeW});
    }

    public static Tensor upsample_nearest1d(Tensor input, long[] outputSize) {
        return torch.upsample_nearest1d(input, outputSize);
    }

    public static Tensor upsample_nearest1d(Tensor input, long size) {
        return upsample_nearest1d(input, new long[]{size});
    }

    public static Tensor upsample_nearest2d(Tensor input, long[] outputSize) {
        return torch.upsample_nearest2d(input, outputSize);
    }

    public static Tensor upsample_nearest3d(Tensor input, long[] outputSize) {
        return torch.upsample_nearest3d(input, outputSize);
    }

    public static Tensor upsample_bilinear2d(Tensor input, long[] outputSize, boolean alignCorners) {
        return torch.upsample_bilinear2d(input, outputSize, alignCorners);
    }

    public static Tensor upsample_bilinear2d(Tensor input, long sizeH, long sizeW, boolean alignCorners) {
        return upsample_bilinear2d(input, new long[]{sizeH, sizeW}, alignCorners);
    }

    public static Tensor upsample_linear1d(Tensor input, long[] outputSize, boolean alignCorners) {
        return torch.upsample_linear1d(input, outputSize, alignCorners);
    }

    public static Tensor upsample_trilinear3d(Tensor input, long[] outputSize, boolean alignCorners) {
        return torch.upsample_trilinear3d(input, outputSize, alignCorners);
    }

    public static Tensor upsample_trilinear3d(Tensor input, long d, long h, long w, boolean alignCorners) {
        return upsample_trilinear3d(input, new long[]{d, h, w}, alignCorners);
    }

    public static Tensor upsample_bicubic2d(Tensor input, long[] outputSize, boolean alignCorners) {
        return torch.upsample_bicubic2d(input, outputSize, alignCorners);
    }

    public static Tensor upsample_bicubic2d(Tensor input, long sizeH, long sizeW, boolean alignCorners) {
        return upsample_bicubic2d(input, new long[]{sizeH, sizeW}, alignCorners);
    }

    public static Tensor upsample(Tensor input, long[] size, String mode, boolean alignCorners) {
        return interpolate(input, size, mode, alignCorners);
    }

    public static Tensor upsample(Tensor input, long[] size, String mode) {
        return upsample(input, size, mode, false);
    }

    public static Tensor affine_grid(Tensor theta, long[] size, boolean alignCorners) {
        return torch.affine_grid(theta, size, alignCorners);
    }

    public static Tensor affine_grid(Tensor theta, long[] size) {
        return affine_grid(theta, size, false);
    }

    public static Tensor grid_sample(Tensor input, Tensor grid, String mode, String paddingMode, boolean alignCorners) {
        GridSampleFuncOptions opt = new GridSampleFuncOptions()
                .mode(toGridSampleMode(mode))
                .padding_mode(toGridSamplePaddingMode(paddingMode))
                .align_corners(new BoolOptional(alignCorners));
        return torch.grid_sample(input, grid, opt);
    }

    public static Tensor grid_sample(Tensor input, Tensor grid, String mode, String paddingMode) {
        return grid_sample(input, grid, mode, paddingMode, false);
    }

    public static Tensor grid_sample(Tensor input, Tensor grid, String mode) {
        return grid_sample(input, grid, mode, "zeros", false);
    }

    // ========================================================================
    // SPARSE
    // ========================================================================

    public static Tensor embedding(Tensor input, Tensor weight, EmbeddingFuncOptions opt) {
        return torch.embedding(input, weight, opt);
    }

    public static Tensor embedding(Tensor input, Tensor weight, long paddingIdx,
                                   boolean scaleGradByFreq, boolean sparse) {
        EmbeddingFuncOptions opt = new EmbeddingFuncOptions()
                .padding_idx(new LongOptional(paddingIdx))
                .scale_grad_by_freq(scaleGradByFreq)
                .sparse(sparse);
        return torch.embedding(input, weight, opt);
    }

    public static Tensor embedding(Tensor input, Tensor weight, long paddingIdx) {
        return embedding(input, weight, paddingIdx, false, false);
    }

    public static Tensor embedding(Tensor input, Tensor weight) {
        return embedding(input, weight, -1, false, false);
    }

    public static Tensor embedding_bag(Tensor input, Tensor weight, Tensor offsets,
                                       boolean scaleGradByFreq, long mode, boolean sparse,
                                       Tensor perSampleWeights, boolean includeLastOffset, long paddingIdx) {
        EmbeddingBagFuncOptions opt = new EmbeddingBagFuncOptions()
                .offsets(offsets)
                .scale_grad_by_freq(scaleGradByFreq)
                .mode(toEmbeddingBagMode(mode))
                .sparse(sparse)
                .per_sample_weights(perSampleWeights)
                .include_last_offset(includeLastOffset)
                .padding_idx(new LongOptional(paddingIdx));
        return torch.embedding_bag(input, weight, opt);
    }

    public static Tensor embedding_bag(Tensor input, Tensor weight, EmbeddingBagFuncOptions opt) {

        return torch.embedding_bag(input, weight, opt);
    }

    public static Tensor embedding_bag(Tensor input, Tensor weight, Tensor offsets,
                                       boolean scaleGradByFreq, long mode) {
        return embedding_bag(input, weight, offsets, scaleGradByFreq, mode, false, null, false, -1);
    }

    public static Tensor embedding_bag(Tensor input, Tensor weight, Tensor offsets, long mode) {
        return embedding_bag(input, weight, offsets, false, mode);
    }

    public static Tensor one_hot(Tensor input, long numClasses) {
        return torch.one_hot(input, numClasses);
    }

    // ========================================================================
    // FOLD / UNFOLD
    // ========================================================================

    public static Tensor fold(Tensor input, long[] outputSize, long[] kernelSize,
                              long[] dilation, long[] padding, long[] stride) {
        FoldOptions opt = new FoldOptions(new LongPointer(outputSize), new LongPointer(kernelSize))
                .dilation(new LongPointer(dilation))
                .padding(new LongPointer(padding))
                .stride(new LongPointer(stride));
        return torch.fold(input, opt);
    }

    public static Tensor fold(Tensor input, long[] outputSize, long[] kernelSize) {
        return fold(input, outputSize, kernelSize, new long[]{1, 1}, new long[]{0, 0}, new long[]{1, 1});
    }

    public static Tensor unfold(Tensor input, long dimension, long size, long step) {
        return torch.unfold_copy(input, dimension, size, step);
    }

    // ========================================================================
    // MULTI-HEAD ATTENTION
    // ========================================================================

    public static Tensor scaled_dot_product_attention(Tensor query, Tensor key, Tensor value,
                                                       Tensor attnMask, double dropoutP, boolean isCausal) {
        return torch.scaled_dot_product_attention(query, key, value, new TensorOptional(attnMask), dropoutP, isCausal, new DoubleOptional(), false);
    }

    public static Tensor scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor attnMask) {
        return scaled_dot_product_attention(query, key, value, attnMask, 0.0, false);
    }

    public static Tensor scaled_dot_product_attention(Tensor query, Tensor key, Tensor value) {
        return scaled_dot_product_attention(query, key, value, null, 0.0, false);
    }

    // ========================================================================
    // ELEMENT-WISE MATH (at:: namespace)
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
    public static Tensor clamp(Tensor input, Scalar min, Scalar max) {
        return torch.clamp(input,
                min != null ? new ScalarOptional(min) : new ScalarOptional(),
                max != null ? new ScalarOptional(max) : new ScalarOptional());
    }
    public static Tensor clamp(Tensor input, double min, double max) {
        return clamp(input, new Scalar(min), new Scalar(max));
    }
    public static Tensor clamp_min(Tensor input, Scalar min) {
        return torch.clamp_min(input, min);
    }
    public static Tensor clamp_max(Tensor input, Scalar max) {
        return torch.clamp_max(input, max);
    }
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
    public static Tensor floor_divide(Tensor input, Tensor other) { return torch.floor_divide(input, other); }
    public static Tensor floor_divide(Tensor input, Scalar other) { return torch.floor_divide(input, other); }
    public static Tensor frac(Tensor input) { return torch.frac(input); }
    public static Tensor i0(Tensor input) { return torch.i0(input); }
    public static Tensor lgamma(Tensor input) { return torch.lgamma(input); }
    public static Tensor log(Tensor input) { return torch.log(input); }
    public static Tensor log10(Tensor input) { return torch.log10(input); }
    public static Tensor log1p(Tensor input) { return torch.log1p(input); }
    public static Tensor log2(Tensor input) { return torch.log2(input); }
    public static Tensor logit(Tensor input) { return torch.logit(input, new DoubleOptional(-1.0)); }
    public static Tensor logit(Tensor input, double eps) { return torch.logit(input, new DoubleOptional(eps)); }
    public static Tensor mvlgamma(Tensor input, long p) { return torch.mvlgamma(input, p); }
    public static Tensor neg(Tensor input) { return torch.neg(input); }
    public static Tensor negative(Tensor input) { return torch.negative(input); }
    public static Tensor polygamma(Tensor input, long n) { return torch.polygamma(n, input); }
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
    public static Tensor tan (Tensor input) { return torch.tan(input); }
    public static Tensor trunc(Tensor input) { return torch.trunc(input); }

    // ========================================================================
    // BINARY ELEMENT-WISE
    // ========================================================================

    public static Tensor add(Tensor input, Tensor other) { return torch.add(input, other); }
    public static Tensor add(Tensor input, Scalar other) { return torch.add(input, other); }
    public static Tensor add(Tensor input, double other) { return torch.add(input, new Scalar(other)); }
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
    public static Tensor atan2(Tensor input, double other) { return torch.atan2(input, torch.tensor(new Scalar(other), new TensorOptions())); }
    public static Tensor bitwise_and(Tensor input, Tensor other) { return torch.bitwise_and(input, other); }
    public static Tensor bitwise_and(Tensor input, Scalar other) { return torch.bitwise_and(input, other); }
    public static Tensor bitwise_or(Tensor input, Tensor other) { return torch.bitwise_or(input, other); }
    public static Tensor bitwise_or(Tensor input, Scalar other) { return torch.bitwise_or(input, other); }
    public static Tensor bitwise_xor(Tensor input, Tensor other) { return torch.bitwise_xor(input, other); }
    public static Tensor bitwise_xor(Tensor input, Scalar other) { return torch.bitwise_xor(input, other); }
    public static Tensor bitwise_not(Tensor input) { return torch.bitwise_not(input); }
    public static Tensor copysign(Tensor input, Tensor other) { return torch.copysign(input, other); }
    public static Tensor copysign(Tensor input, Scalar other) { return torch.copysign(input, other); }
    public static Tensor nextafter(Tensor input, Tensor other) { return torch.nextafter(input, other); }
    public static Tensor gcd(Tensor input, Tensor other) { return torch.gcd(input, other); }
    public static Tensor lcm(Tensor input, Tensor other) { return torch.lcm(input, other); }
    public static Tensor maximum(Tensor input, Tensor other) { return torch.maximum(input, other); }
    public static Tensor minimum(Tensor input, Tensor other) { return torch.minimum(input, other); }
    public static Tensor fmax(Tensor input, Tensor other) { return torch.fmax(input, other); }
    public static Tensor fmin(Tensor input, Tensor other) { return torch.fmin(input, other); }
    public static Tensor ldexp(Tensor input, Tensor other) { return torch.ldexp(input, other); }
    public static Tensor logaddexp(Tensor input, Tensor other) { return torch.logaddexp(input, other); }
    public static Tensor logaddexp2(Tensor input, Tensor other) { return torch.logaddexp2(input, other); }
    public static Tensor logical_not(Tensor input) { return torch.logical_not(input); }
    public static Tensor xlogy(Tensor input, Tensor other) { return torch.xlogy(input, other); }
    public static Tensor xlogy(Tensor input, Scalar other) { return torch.xlogy(input, other); }

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
    public static boolean allclose(Tensor input, Tensor other, double rtol, double atol, boolean equalNan) {
        return torch.allclose(input, other, rtol, atol, equalNan);
    }
    public static boolean allclose(Tensor input, Tensor other) { return torch.allclose(input, other); }
    public static Tensor logical_and(Tensor input, Tensor other) { return torch.logical_and(input, other); }
    public static Tensor logical_or(Tensor input, Tensor other) { return torch.logical_or(input, other); }
    public static Tensor logical_xor(Tensor input, Tensor other) { return torch.logical_xor(input, other); }
    public static Tensor where(Tensor condition, Tensor input, Tensor other) {
        return torch.where(condition, input, other);
    }

    // ========================================================================
    // REDUCTION
    // ========================================================================

    public static Tensor sum(Tensor input) { return torch.sum(input); }
    public static Tensor sum(Tensor input, long dim, boolean keepdim) {
        return torch.sum(input, new LongArrayRefOptional(new long[]{dim}), keepdim, new ScalarTypeOptional());
    }
    public static Tensor sum(Tensor input, long dim) { return sum(input, dim, false); }
    public static Tensor sum(Tensor input, long[] dim, boolean keepdim) {
        return torch.sum(input, new LongArrayRefOptional(dim), keepdim, new ScalarTypeOptional());
    }
    public static Tensor sum(Tensor input, long[] dim) { return sum(input, dim, false); }

    public static Tensor mean(Tensor input) { return torch.mean(input); }
    public static Tensor mean(Tensor input, long dim, boolean keepdim) {
        return torch.mean(input, new LongArrayRefOptional(new long[]{dim}), keepdim, new ScalarTypeOptional());
    }
    public static Tensor mean(Tensor input, long dim) { return mean(input, dim, false); }
    public static Tensor mean(Tensor input, long[] dim, boolean keepdim) {
        return torch.mean(input, new LongArrayRefOptional(dim), keepdim, new ScalarTypeOptional());
    }
    public static Tensor mean(Tensor input, long[] dim) { return mean(input, dim, false); }

    public static Tensor prod(Tensor input) { return torch.prod(input); }
    public static Tensor prod(Tensor input, long dim, boolean keepdim) {
        return torch.prod(input, dim, keepdim, new ScalarTypeOptional());
    }
    public static Tensor prod(Tensor input, long dim) { return torch.prod(input, dim); }

    public static Tensor max(Tensor input) { return torch.max(input); }
    public static Tensor max(Tensor input, long dim, boolean keepdim) {
        return torch.max(input, dim, keepdim).get0();
    }
    public static Tensor max(Tensor input, long dim) { return max(input, dim, false); }

    public static Tensor min(Tensor input) { return torch.min(input); }
    public static Tensor min(Tensor input, long dim, boolean keepdim) {
        return torch.min(input, dim, keepdim).get0();
    }
    public static Tensor min(Tensor input, long dim) { return min(input, dim, false); }

    public static Tensor amin(Tensor input, long[] dim, boolean keepdim) {
        return torch.amin(input, dim, keepdim);
    }
    public static Tensor amin(Tensor input, long dim, boolean keepdim) {
        return torch.amin(input, new long[]{dim}, keepdim);
    }
    public static Tensor amin(Tensor input, long[] dim) { return amin(input, dim, false); }
    public static Tensor amin(Tensor input, long dim) { return amin(input, dim, false); }
    public static Tensor amin(Tensor input) { return torch.amin(input); }
    public static Tensor amax(Tensor input, long[] dim, boolean keepdim) {
        return torch.amax(input, dim, keepdim);
    }
    public static Tensor amax(Tensor input, long dim, boolean keepdim) {
        return torch.amax(input, new long[]{dim}, keepdim);
    }
    public static Tensor amax(Tensor input, long[] dim) { return amax(input, dim, false); }
    public static Tensor amax(Tensor input, long dim) { return amax(input, dim, false); }
    public static Tensor amax(Tensor input) { return torch.amax(input); }
    public static Tensor argmax(Tensor input) { return torch.argmax(input); }
    public static Tensor argmax(Tensor input, long dim, boolean keepdim) {
        return torch.argmax(input, new LongOptional(dim), keepdim);
    }
    public static Tensor argmax(Tensor input, long dim) { return argmax(input, dim, false); }
    public static Tensor argmin(Tensor input) { return torch.argmin(input); }
    public static Tensor argmin(Tensor input, long dim, boolean keepdim) {
        return torch.argmin(input, new LongOptional(dim), keepdim);
    }
    public static Tensor argmin(Tensor input, long dim) { return argmin(input, dim, false); }

    public static Tensor std(Tensor input) { return torch.std(input); }
    public static Tensor std(Tensor input, long dim, boolean unbiased, boolean keepdim) {
        return torch.std(input, new LongArrayRefOptional(new long[]{dim}), unbiased, keepdim);
    }
    public static Tensor std(Tensor input, long dim, boolean unbiased) { return std(input, dim, unbiased, false); }

    public static Tensor var(Tensor input) { return torch.var(input); }
    public static Tensor var(Tensor input, long dim, boolean unbiased, boolean keepdim) {
        return torch.var(input, new LongArrayRefOptional(new long[]{dim}), unbiased, keepdim);
    }
    public static Tensor var(Tensor input, long dim, boolean unbiased) { return var(input, dim, unbiased, false); }

    public static Tensor median(Tensor input) { return torch.median(input); }
    public static Tensor median(Tensor input, long dim, boolean keepdim) {
        return torch.median(input, dim, keepdim).get0();
    }
    public static Tensor median(Tensor input, long dim) { return median(input, dim, false); }

    public static Tensor quantile(Tensor input, double q, long dim, boolean keepdim) {
        return torch.quantile(input, q, new LongOptional(dim), keepdim, "linear");
    }
    public static Tensor quantile(Tensor input, Tensor q, long dim, boolean keepdim) {
        return torch.quantile(input, q, new LongOptional(dim), keepdim, "linear");
    }
    public static Tensor nanmean(Tensor input) { return torch.nanmean(input); }
    public static Tensor nanmean(Tensor input, long[] dim, boolean keepdim) {
        return torch.nanmean(input, new LongArrayRefOptional(dim), keepdim, new ScalarTypeOptional());
    }
    public static Tensor nanmedian(Tensor input) { return torch.nanmedian(input); }
    public static Tensor nansum(Tensor input) { return torch.nansum(input); }
    public static Tensor nansum(Tensor input, long[] dim, boolean keepdim) {
        return torch.nansum(input, new LongArrayRefOptional(dim), keepdim, new ScalarTypeOptional());
    }
    public static Tensor nansum(Tensor input, long dim, boolean keepdim) {
        return torch.nansum(input, new long[]{dim}, keepdim, new ScalarTypeOptional());
    }

    public static Tensor nan_to_num(Tensor input, double nanVal, double posinfVal, double neginfVal) {
        return torch.nan_to_num(input, new DoubleOptional(nanVal), new DoubleOptional(posinfVal), new DoubleOptional(neginfVal));
    }
    public static Tensor nan_to_num(Tensor input) {
        return torch.nan_to_num(input);
    }

    public static Tensor norm(Tensor input, double p) { return torch.norm(input, new Scalar(p)); }
    public static Tensor norm(Tensor input, Scalar p) { return torch.norm(input, p); }
    public static Tensor norm(Tensor input, double p, long dim, boolean keepdim) {
        return torch.norm(input, new ScalarOptional(new Scalar(p)), new long[]{dim}, keepdim);
    }
    public static Tensor norm(Tensor input, double p, long[] dim, boolean keepdim) {
        return torch.norm(input, new ScalarOptional(new Scalar(p)), dim, keepdim);
    }

    public static Tensor frobenius_norm(Tensor input) { return torch.frobenius_norm(input); }
    public static Tensor frobenius_norm(Tensor input, long[] dim, boolean keepdim) {
        return torch.frobenius_norm(input, dim, keepdim);
    }
    public static Tensor nuclear_norm(Tensor input) { return torch.nuclear_norm(input); }
    public static Tensor logsumexp(Tensor input, long dim, boolean keepdim) {
        return torch.logsumexp(input, new long[]{dim}, keepdim);
    }
    public static Tensor logsumexp(Tensor input, long dim) { return logsumexp(input, dim, false); }
    public static Tensor logcumsumexp(Tensor input, long dim) { return torch.logcumsumexp(input, dim); }
    public static Tensor cumsum(Tensor input, long dim) { return torch.cumsum(input, dim); }
    public static Tensor cumprod(Tensor input, long dim) { return torch.cumprod(input, dim); }
    public static Tensor dist(Tensor input, Tensor other, double p) { return torch.dist(input, other, new Scalar(p)); }
    public static Tensor dist(Tensor input, Tensor other) { return torch.dist(input, other); }
    public static Tensor count_nonzero(Tensor input, long[] dim) {
        return torch.count_nonzero(input, dim);
    }
    public static Tensor count_nonzero(Tensor input) { return torch.count_nonzero(input); }
    public static Tensor all(Tensor input) { return torch.all(input); }
    public static Tensor all(Tensor input, long dim, boolean keepdim) { return torch.all(input, dim, keepdim); }
    public static Tensor any(Tensor input) { return torch.any(input); }
    public static Tensor any(Tensor input, long dim, boolean keepdim) { return torch.any(input, dim, keepdim); }
    public static Tensor histc(Tensor input, long bins, double min, double max) {
        return torch.histc(input, bins, new Scalar(min), new Scalar(max));
    }
    public static Tensor trace(Tensor input) { return torch.trace(input); }
    public static Tensor det(Tensor input) { return torch.det(input); }
    public static Tensor logdet(Tensor input) { return torch.logdet(input); }
    public static Tensor matrix_exp(Tensor input) { return torch.matrix_exp(input); }

    // ========================================================================
    // TENSOR SHAPE / MANIPULATION
    // ========================================================================

    public static Tensor reshape(Tensor input, long... shape) { return input.reshape(shape); }
    public static Tensor view(Tensor input, long... shape) { return input.view(shape); }
    public static Tensor flatten(Tensor input, long startDim, long endDim) {
        return torch.flatten(input, startDim, endDim);
    }
    public static Tensor flatten(Tensor input) { return torch.flatten(input); }
    public static Tensor unflatten(Tensor input, long dim, long[] sizes) {
        return torch.unflatten(input, dim, sizes);
    }
    public static Tensor squeeze(Tensor input) { return torch.squeeze(input); }
    public static Tensor squeeze(Tensor input, long dim) { return torch.squeeze(input, dim); }
    public static Tensor unsqueeze(Tensor input, long dim) { return torch.unsqueeze(input, dim); }
    public static Tensor transpose(Tensor input, long dim0, long dim1) { return torch.transpose(input, dim0, dim1); }
    public static Tensor t(Tensor input) { return torch.t(input); }
    public static Tensor permute(Tensor input, long... dims) { return torch.permute(input, dims); }
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
    public static Tensor expand(Tensor input, long... sizes) { return input.expand(sizes); }
    public static Tensor expand_as(Tensor input, Tensor other) { return input.expand_as(other); }
    public static Tensor repeat(Tensor input, long... repeats) { return input.repeat(repeats); }
    public static Tensor broadcast_to(Tensor input, long... size) { return torch.broadcast_to(input, size); }

    public static Tensor cat(Tensor[] tensors, long dim) {
        TensorVector list = new TensorVector();
        for (int i = 0; i < tensors.length; i++) list.push_back(tensors[i]);
        return torch.cat(list, dim);
    }
    public static Tensor concat(Tensor[] tensors, long dim) { return cat(tensors, dim); }
    public static Tensor concatenate(Tensor[] tensors, long dim) { return cat(tensors, dim); }
    public static Tensor stack(Tensor[] tensors, long dim) {
        TensorVector list = new TensorVector();
        for (int i = 0; i < tensors.length; i++) list.push_back(tensors[i]);
        return torch.stack(list, dim);
    }
    public static Tensor split(Tensor tensor, long splitSize, long dim) {
        TensorVector v = torch.split(tensor, splitSize, dim);
        return v.get(0);
    }
    public static TensorVector split(Tensor tensor, long[] splitSizes, long dim) {
        return torch.split(tensor, splitSizes, dim);
    }
    public static Tensor chunk(Tensor tensor, long chunks, long dim) {
        TensorVector v = torch.chunk(tensor, chunks, dim);
        return v.get(0);
    }
    public static TensorVector chunks(Tensor tensor, long chunks, long dim) {
        return torch.chunk(tensor, chunks, dim);
    }
    public static TensorVector hsplit(Tensor input, long sections) { return torch.hsplit(input, sections); }
    public static TensorVector vsplit(Tensor input, long sections) { return torch.vsplit(input, sections); }
    public static TensorVector dsplit(Tensor input, long sections) { return torch.dsplit(input, sections); }
    public static TensorVector meshgrid(Tensor[] tensors, String indexing) {
        TensorVector list = new TensorVector();
        for (int i = 0; i < tensors.length; i++) list.push_back(tensors[i]);
        return torch.meshgrid(list, indexing);
    }
    public static TensorVector meshgrid(Tensor[] tensors) { return meshgrid(tensors, "ij"); }

    public static Tensor index_select(Tensor input, long dim, Tensor index) {
        return torch.index_select(input, dim, index);
    }
    public static Tensor masked_select(Tensor input, Tensor mask) { return torch.masked_select(input, mask); }
    public static Tensor masked_fill(Tensor input, Tensor mask, Scalar value) { return input.masked_fill(mask, value); }
    public static Tensor masked_fill(Tensor input, Tensor mask, double value) {
        return masked_fill(input, mask, new Scalar(value));
    }
    public static Tensor masked_scatter(Tensor input, Tensor mask, Tensor source) {
        return input.masked_scatter(mask, source);
    }
    public static Tensor nonzero(Tensor input) { return torch.nonzero(input); }
    public static Tensor take(Tensor input, Tensor index) { return torch.take(input, index); }
    public static Tensor take_along_dim(Tensor input, Tensor indices, long dim) {
        return torch.take_along_dim(input, indices, new LongOptional(dim));
    }
    public static Tensor gather(Tensor input, long dim, Tensor index, boolean sparseGrad) {
        return torch.gather(input, dim, index, sparseGrad);
    }
    public static Tensor gather(Tensor input, long dim, Tensor index) {
        return torch.gather(input, dim, index, false);
    }
    public static Tensor scatter(Tensor input, long dim, Tensor index, Tensor src) {
        return torch.scatter(input, dim, index, src);
    }
    public static Tensor scatter_add(Tensor input, long dim, Tensor index, Tensor src) {
        return torch.scatter_add(input, dim, index, src);
    }
    public static Tensor scatter_reduce(Tensor input, long dim, Tensor index, Tensor src, String reduce) {
        return torch.scatter_reduce(input, dim, index, src, reduce);
    }
    public static Tensor diagonal(Tensor input, long offset, long dim1, long dim2) {
        return torch.diagonal(input, offset, dim1, dim2);
    }
    public static Tensor diagonal(Tensor input) { return torch.diagonal(input); }
    public static Tensor diag(Tensor input, long diagonal) { return torch.diag(input, diagonal); }
    public static Tensor diag(Tensor input) { return torch.diag(input); }
    public static Tensor diag_embed(Tensor input, long offset, long dim1, long dim2) {
        return torch.diag_embed(input, offset, dim1, dim2);
    }
    public static Tensor diagflat(Tensor input, long offset) { return torch.diagflat(input, offset); }
    public static Tensor diagflat(Tensor input) { return torch.diagflat(input); }
    public static Tensor triu(Tensor input, long diagonal) { return torch.triu(input, diagonal); }
    public static Tensor triu(Tensor input) { return torch.triu(input); }
    public static Tensor tril(Tensor input, long diagonal) { return torch.tril(input, diagonal); }
    public static Tensor tril(Tensor input) { return torch.tril(input); }
    public static Tensor triu_indices(long row, long col, long offset) {
        return torch.triu_indices(row, col, offset, new TensorOptions());
    }
    public static Tensor triu_indices(long row, long col) { return torch.triu_indices(row, col); }
    public static Tensor tril_indices(long row, long col, long offset) {
        return torch.tril_indices(row, col, offset, new TensorOptions());
    }
    public static Tensor tril_indices(long row, long col) { return torch.tril_indices(row, col); }
    public static Tensor roll(Tensor input, long[] shifts, long[] dims) {
        return torch.roll(input, shifts, dims);
    }
    public static Tensor roll(Tensor input, long[] shifts) {
        return torch.roll(input, shifts);
    }
    public static Tensor flip(Tensor input, long[] dims) { return torch.flip(input, dims); }
    public static Tensor fliplr(Tensor input) { return torch.fliplr(input); }
    public static Tensor flipud(Tensor input) { return torch.flipud(input); }
    public static Tensor rot90(Tensor input, long k, long[] dims) {
        return torch.rot90(input, k, dims);
    }
    public static Tensor rot90(Tensor input, long k) { return rot90(input, k, new long[]{0, 1}); }
    public static Tensor rot90(Tensor input) { return rot90(input, 1); }
    public static Tensor tile(Tensor input, long[] dims) { return torch.tile(input, dims); }
    public static Tensor repeat_interleave(Tensor input, long repeats, long dim) {
        return torch.repeat_interleave(input, repeats, new LongOptional(dim), new LongOptional());
    }
    public static Tensor repeat_interleave(Tensor input, long repeats) {
        return torch.repeat_interleave(input, repeats);
    }
    public static Tensor repeat_interleave(Tensor input, Tensor repeats, long dim) {
        return torch.repeat_interleave(input, repeats, new LongOptional(dim), new LongOptional());
    }
    public static Tensor atleast_1d(Tensor input) { return torch.atleast_1d(input); }
    public static Tensor atleast_2d(Tensor input) { return torch.atleast_2d(input); }
    public static Tensor atleast_3d(Tensor input) { return torch.atleast_3d(input); }
    public static Tensor hstack(Tensor[] tensors) {
        TensorVector list = new TensorVector();
        for (int i = 0; i < tensors.length; i++) list.push_back(tensors[i]);
        return torch.hstack(list);
    }
    public static Tensor vstack(Tensor[] tensors) {
        TensorVector list = new TensorVector();
        for (int i = 0; i < tensors.length; i++) list.push_back(tensors[i]);
        return torch.vstack(list);
    }
    public static Tensor dstack(Tensor[] tensors) {
        TensorVector list = new TensorVector();
        for (int i = 0; i < tensors.length; i++) list.push_back(tensors[i]);
        return torch.dstack(list);
    }
    public static Tensor column_stack(Tensor[] tensors) {
        TensorVector list = new TensorVector();
        for (int i = 0; i < tensors.length; i++) list.push_back(tensors[i]);
        return torch.column_stack(list);
    }
    public static Tensor row_stack(Tensor[] tensors) { return vstack(tensors); }
    public static Tensor cartesian_prod(Tensor[] tensors) {
        TensorVector list = new TensorVector();
        for (int i = 0; i < tensors.length; i++) list.push_back(tensors[i]);
        return torch.cartesian_prod(list);
    }
    public static Tensor combinations(Tensor input, long r, boolean withReplacement) {
        return torch.combinations(input, r, withReplacement);
    }
    public static Tensor block_diag(Tensor[] tensors) {
        TensorVector list = new TensorVector();
        for (int i = 0; i < tensors.length; i++) list.push_back(tensors[i]);
        return torch.block_diag(list);
    }

    // ========================================================================
    // LINEAR ALGEBRA
    // ========================================================================

    public static Tensor matmul(Tensor input, Tensor other) { return torch.matmul(input, other); }
    public static Tensor mm(Tensor input, Tensor other) { return torch.mm(input, other); }
    public static Tensor bmm(Tensor input, Tensor other) { return torch.bmm(input, other); }
    public static Tensor baddbmm(Tensor input, Tensor batch1, Tensor batch2, double beta, double alpha) {
        return torch.baddbmm(input, batch1, batch2, new Scalar(beta), new Scalar(alpha));
    }
    public static Tensor baddbmm(Tensor input, Tensor batch1, Tensor batch2) {
        return baddbmm(input, batch1, batch2, 1.0, 1.0);
    }
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
    public static Tensor vdot(Tensor input, Tensor other) { return torch.vdot(input, other); }
    public static Tensor mv(Tensor input, Tensor other) { return torch.mv(input, other); }
    public static Tensor outer(Tensor input, Tensor vec2) { return torch.outer(input, vec2); }
    public static Tensor inner(Tensor input, Tensor other) { return torch.inner(input, other); }
    public static Tensor ger(Tensor input, Tensor vec2) { return torch.ger(input, vec2); }
    public static Tensor cross(Tensor input, Tensor other, long dim) {
        return torch.cross(input, other, new LongOptional(dim));
    }
    public static Tensor cross(Tensor input, Tensor other) { return torch.cross(input, other); }
    public static Tensor tensordot(Tensor input, Tensor other, long[] dimsA, long[] dimsB) {
        return torch.tensordot(input, other, dimsA, dimsB);
    }
    public static Tensor tensordot(Tensor input, Tensor other, long axes) {
        long[] dimsA = new long[(int)axes];
        long[] dimsB = new long[(int)axes];
        for (int i = 0; i < axes; i++) { dimsA[i] = i; dimsB[i] = i; }
        return torch.tensordot(input, other, dimsA, dimsB);
    }
    public static Tensor einsum(String equation, Tensor... operands) {
        TensorVector list = new TensorVector();
        for (int i = 0; i < operands.length; i++) list.push_back(operands[i]);
        return torch.einsum(equation, list);
    }
    public static Tensor chain_matmul(Tensor... matrices) {
        TensorVector list = new TensorVector();
        for (int i = 0; i < matrices.length; i++) list.push_back(matrices[i]);
        return torch.chain_matmul(list);
    }
    public static Tensor inverse(Tensor input) { return torch.inverse(input); }
    public static Tensor pinverse(Tensor input) { return torch.pinverse(input); }
    public static Tensor matrix_power(Tensor input, long n) { return torch.matrix_power(input, n); }
    public static Tensor renorm(Tensor input, double p, long dim, double maxnorm) {
        return torch.renorm(input, new Scalar(p), dim, new Scalar(maxnorm));
    }
    public static Tensor kron(Tensor input, Tensor other) { return torch.kron(input, other); }
    public static Tensor cov(Tensor input) { return torch.cov(input); }
    public static Tensor corrcoef(Tensor input) { return torch.corrcoef(input); }
    public static Tensor polar(Tensor abs, Tensor angle) { return torch.polar(abs, angle); }
    public static Tensor angle(Tensor input) { return torch.angle(input); }
    public static Tensor imag(Tensor input) { return torch.imag(input); }
    public static Tensor real(Tensor input) { return torch.real(input); }
    public static Tensor view_as_complex(Tensor input) { return torch.view_as_complex(input); }
    public static Tensor view_as_real(Tensor input) { return torch.view_as_real(input); }

    // ========================================================================
    // TENSOR CREATION
    // ========================================================================

    public static Tensor arange(Scalar end) { return torch.arange(end); }
    public static Tensor arange(double end) { return torch.arange(new Scalar(end)); }
    public static Tensor arange(Scalar start, Scalar end) { return torch.arange(start, end); }
    public static Tensor arange(double start, double end) { return torch.arange(new Scalar(start), new Scalar(end)); }
    public static Tensor arange(Scalar start, Scalar end, Scalar step) { return torch.arange(start, end, step); }
    public static Tensor arange(double start, double end, double step) {
        return torch.arange(new Scalar(start), new Scalar(end), new Scalar(step), new TensorOptions());
    }
    public static Tensor range(Scalar start, Scalar end, Scalar step) {
        return torch.range(start, end, step, new TensorOptions());
    }
    public static Tensor linspace(Scalar start, Scalar end, long steps) {
        return torch.linspace(start, end, steps);
    }
    public static Tensor linspace(double start, double end, long steps) {
        return torch.linspace(new Scalar(start), new Scalar(end), steps);
    }
    public static Tensor logspace(Scalar start, Scalar end, long steps, double base) {
        return torch.logspace(start, end, steps, base, new TensorOptions());
    }
    public static Tensor logspace(double start, double end, long steps, double base) {
        return torch.logspace(new Scalar(start), new Scalar(end), steps, base, new TensorOptions());
    }
    public static Tensor eye(long n) { return torch.eye(n); }
    public static Tensor eye(long n, long m) { return torch.eye(n, m); }
    public static Tensor empty(long... size) { return torch.empty(size); }
    public static Tensor zeros(long... size) { return torch.zeros(size); }
    public static Tensor ones(long... size) { return torch.ones(size); }
    public static Tensor full(Scalar value, long... size) { return torch.full(size, value); }
    public static Tensor full(double value, long... size) { return full(new Scalar(value), size); }
    public static Tensor empty_like(Tensor input) { return torch.empty_like(input); }
    public static Tensor zeros_like(Tensor input) { return torch.zeros_like(input); }
    public static Tensor ones_like(Tensor input) { return torch.ones_like(input); }
    public static Tensor full_like(Tensor input, Scalar value) { return torch.full_like(input, value); }
    public static Tensor full_like(Tensor input, double value) { return full_like(input, new Scalar(value)); }
    public static Tensor rand(long... size) { return torch.rand(size); }
    public static Tensor randn(long... size) { return torch.randn(size); }
    public static Tensor randint(long low, long high, long... size) {
        return torch.randint(low, high, size);
    }
    public static Tensor randperm(long n) { return torch.randperm(n); }
    public static Tensor normal(double mean, double std, long... size) {
        return torch.normal(mean, std, size);
    }
    public static Tensor normal(Tensor mean, Tensor std, long... size) {
        return torch.normal(mean, std);
    }
    public static Tensor bernoulli(Tensor input) { return torch.bernoulli(input); }
    public static Tensor multinomial(Tensor input, long numSamples, boolean replacement) {
        return torch.multinomial(input, numSamples, replacement, new GeneratorOptional());
    }
    public static Tensor multinomial(Tensor input, long numSamples) {
        return multinomial(input, numSamples, false);
    }
    public static Tensor poisson(Tensor input) { return torch.poisson(input); }
    public static Tensor exponential(Tensor input) { return torch.exponential(input); }
    public static Tensor geometric(Tensor input, double p) { return torch.geometric(input, p); }
    public static Tensor cauchy(Tensor input) { return torch.cauchy(input); }
    public static Tensor log_normal(Tensor input) { return torch.log_normal(input); }
    public static Tensor log_normal(Tensor input, double mean, double std) {
        return torch.log_normal(input, mean, std, new GeneratorOptional());
    }
    public static Tensor uniform(Tensor input, double from, double to) {
        return torch.uniform(input, from, to, new GeneratorOptional());
    }
    public static Tensor bartlett_window(long windowLength, boolean periodic) {
        return torch.bartlett_window(windowLength, periodic);
    }
    public static Tensor bartlett_window(long windowLength) {
        return torch.bartlett_window(windowLength);
    }
    public static Tensor blackman_window(long windowLength, boolean periodic) {
        return torch.blackman_window(windowLength, periodic);
    }
    public static Tensor blackman_window(long windowLength) {
        return torch.blackman_window(windowLength);
    }
    public static Tensor hamming_window(long windowLength, boolean periodic) {
        return torch.hamming_window(windowLength, periodic);
    }
    public static Tensor hamming_window(long windowLength) {
        return torch.hamming_window(windowLength);
    }
    public static Tensor hann_window(long windowLength, boolean periodic) {
        return torch.hann_window(windowLength, periodic);
    }
    public static Tensor hann_window(long windowLength) {
        return torch.hann_window(windowLength);
    }
    public static Tensor kaiser_window(long windowLength, boolean periodic, double beta) {
        return torch.kaiser_window(windowLength, periodic, beta);
    }
    public static Tensor kaiser_window(long windowLength) {
        return torch.kaiser_window(windowLength);
    }
    public static Tensor vander(Tensor x, long N, boolean increasing) {
        return torch.vander(x, new LongOptional(N), increasing);
    }
    public static Tensor vander(Tensor x, long N) { return vander(x, N, false); }
    public static Tensor vander(Tensor x) { return torch.vander(x); }

    // ========================================================================
    // INDEXING / ADVANCED INDEXING
    // ========================================================================

    public static Tensor index(Tensor input, Tensor... indices) {
        TensorOptionalList list = new TensorOptionalList();
        for (int i = 0; i < indices.length; i++) list.push_back(new TensorOptional(indices[i]));
        return torch.index(input, list);
    }
    public static Tensor index_put(Tensor input, Tensor[] indices, Tensor values, boolean accumulate) {
        TensorOptionalList list = new TensorOptionalList();
        for (int i = 0; i < indices.length; i++) list.push_back(new TensorOptional(indices[i]));
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
    public static Tensor bucketize(Tensor input, Tensor boundaries, boolean right) {
        return torch.bucketize(input, boundaries, false, right);
    }
    public static Tensor bucketize(Tensor input, Tensor boundaries) {
        return bucketize(input, boundaries, false);
    }
    public static Tensor searchsorted(Tensor sortedSequence, Tensor values, boolean right) {
        return torch.searchsorted(sortedSequence, values, false, right, new StringViewOptional(), new TensorOptional());
    }
    public static Tensor searchsorted(Tensor sortedSequence, Tensor values) {
        return searchsorted(sortedSequence, values, false);
    }

    // ========================================================================
    // MISC / UTILITY
    // ========================================================================

    public static Tensor contiguous(Tensor input) { return input.contiguous(); }
    public static Tensor to(Tensor input, TensorOptions options) { return torch.torch__to_copy(input, options, false, new MemoryFormatOptional()); }
    public static Tensor to(Tensor input, Device device) {
        return torch.torch__to_copy(input, new TensorOptions(device), false, new MemoryFormatOptional());
    }
    public static Tensor cpu(Tensor input) { return input.cpu(); }
    public static Tensor cuda(Tensor input) { return input.cuda(); }
    public static Tensor type_as(Tensor input, Tensor other) { return input.type_as(other); }
    public static Tensor copy_(Tensor input, Tensor src) { return input.copy_(src); }
    public static Tensor dequantize(Tensor input) { return torch.dequantize(input); }
    public static Tensor int_repr(Tensor input) { return torch.int_repr(input); }
    public static Tensor select(Tensor input, long dim, long index) {
        return torch.select(input, dim, index);
    }
    public static Tensor narrow(Tensor input, long dim, long start, long length) {
        return torch.narrow(input, dim, start, length);
    }
    public static Tensor slice(Tensor input, long dim, long start, long end, long step) {
        return torch.slice(input, dim, new LongOptional(start), new LongOptional(end), step);
    }
    public static TensorVector unbind(Tensor input, long dim) { return torch.unbind(input, dim); }
    public static TensorVector tensor_split(Tensor input, long sections, long dim) {
        return torch.tensor_split(input, sections, dim);
    }
    public static TensorVector tensor_split(Tensor input, long[] indices, long dim) {
        return torch.tensor_split(input, indices, dim);
    }
    public static TensorVector split_with_sizes(Tensor input, long[] sizes, long dim) {
        return torch.split(input, sizes, dim);
    }
    public static Tensor expand(Tensor input, long[] sizes, boolean implicit) {
        // implicit is not directly supported; use regular expand
        return input.expand(sizes, implicit);
    }
    public static Tensor as_strided(Tensor input, long[] size, long[] stride, long storageOffset) {
        return torch.as_strided(input, size, stride, new LongOptional(storageOffset));
    }
    public static Tensor as_strided(Tensor input, long[] size, long[] stride) {
        return as_strided(input, size, stride, 0);
    }
    public static Tensor cdist(Tensor x1, Tensor x2, double p) {
        return torch.cdist(x1, x2, p, new LongOptional());
    }
    public static Tensor cdist(Tensor x1, Tensor x2) { return cdist(x1, x2, 2.0); }
    public static Tensor pdist(Tensor input, double p) { return torch.pdist(input, p); }
    public static Tensor pdist(Tensor input) { return pdist(input, 2.0); }

    // ========================================================================
    // SIGNAL / SPECTROGRAM
    // ========================================================================

    public static Tensor stft(Tensor input, long nFFT, long hopLength, long winLength,
                              Tensor window, boolean center, String padMode,
                              boolean normalized, boolean onesided, long length) {
        return torch.stft(input, nFFT, new LongOptional(hopLength), new LongOptional(winLength),
                new TensorOptional(window), center, padMode, normalized,
                new BoolOptional(onesided), new BoolOptional(true), new BoolOptional(false));
    }
    public static Tensor stft(Tensor input, long nFFT, long hopLength, long winLength, Tensor window) {
        return stft(input, nFFT, hopLength, winLength, window, true, "reflect", false, true, -1);
    }
    public static Tensor istft(Tensor input, long nFFT, long hopLength, long winLength,
                               Tensor window, boolean center, boolean normalized,
                               boolean onesided, long length) {
        return torch.istft(input, nFFT, new LongOptional(hopLength), new LongOptional(winLength),
                new TensorOptional(window), center, normalized,
                new BoolOptional(onesided), new LongOptional(length), false);
    }

    // ========================================================================
    // FFT
    // ========================================================================

    public static Tensor fft_fft(Tensor input) { return torch.fft_fft(input); }
    public static Tensor fft_ifft(Tensor input) { return torch.fft_ifft(input); }
    public static Tensor fft_rfft(Tensor input) { return torch.fft_rfft(input); }
    public static Tensor fft_irfft(Tensor input) { return torch.fft_irfft(input); }
    public static Tensor fft_hfft(Tensor input) { return torch.fft_hfft(input); }
    public static Tensor fft_ihfft(Tensor input) { return torch.fft_ihfft(input); }
    public static Tensor fft_fft2(Tensor input) { return torch.fft_fft2(input); }
    public static Tensor fft_ifft2(Tensor input) { return torch.fft_ifft2(input); }
    public static Tensor fft_rfft2(Tensor input) { return torch.fft_rfft2(input); }
    public static Tensor fft_irfft2(Tensor input) { return torch.fft_irfft2(input); }
    public static Tensor fft_hfft2(Tensor input) { return torch.fft_hfft2(input); }
    public static Tensor fft_ihfft2(Tensor input) { return torch.fft_ihfft2(input); }
    public static Tensor fft_fftshift(Tensor input) { return torch.fft_fftshift(input); }
    public static Tensor fft_ifftshift(Tensor input) { return torch.fft_ifftshift(input); }
    public static Tensor fft_fftfreq(long n, double d) { return torch.fft_fftfreq(n, d, new TensorOptions()); }
    public static Tensor fft_fftfreq(long n) { return torch.fft_fftfreq(n); }
    public static Tensor fft_rfftfreq(long n, double d) { return torch.fft_rfftfreq(n, d, new TensorOptions()); }
    public static Tensor fft_rfftfreq(long n) { return torch.fft_rfftfreq(n); }

    // ========================================================================
    // LINALG
    // ========================================================================

    public static Tensor linalg_norm(Tensor input) { return torch.linalg_norm(input); }
    public static Tensor linalg_norm(Tensor input, Scalar ord) {
        return torch.linalg_norm(input, new ScalarOptional(ord), new LongArrayRefOptional(), false, new ScalarTypeOptional());
    }
    public static Tensor linalg_norm(Tensor input, String ord) { return torch.linalg_norm(input, new BytePointer(ord)); }
    public static Tensor linalg_vector_norm(Tensor input, double ord) {
        return torch.linalg_vector_norm(input, new Scalar(ord), new LongArrayRefOptional(), false, new ScalarTypeOptional());
    }
    public static Tensor linalg_vector_norm(Tensor input) { return torch.linalg_vector_norm(input); }
    public static Tensor linalg_matrix_norm(Tensor input, Scalar ord) { return torch.linalg_matrix_norm(input, ord); }
    public static Tensor linalg_matrix_norm(Tensor input, String ord) { return torch.linalg_matrix_norm(input, new BytePointer(ord), new LongArrayRef(new LongPointer(new long[]{-2, -1}), 2), false, new ScalarTypeOptional()); }
    public static Tensor linalg_inv(Tensor input) { return torch.linalg_inv(input); }
    public static Tensor linalg_pinv(Tensor input) { return torch.linalg_pinv(input); }
    public static Tensor linalg_det(Tensor input) { return torch.linalg_det(input); }
    public static Tensor linalg_slogdet(Tensor input) { return torch.linalg_slogdet(input).get0(); }
    public static Tensor linalg_eigvals(Tensor input) { return torch.linalg_eigvals(input); }
    public static Tensor linalg_eigvalsh(Tensor input) { return torch.linalg_eigvalsh(input); }
    public static Tensor linalg_cholesky(Tensor input) { return torch.linalg_cholesky(input, false); }
    public static Tensor linalg_cholesky(Tensor input, boolean upper) {
        return torch.linalg_cholesky(input, upper);
    }
    public static Tensor linalg_solve(Tensor A, Tensor B) { return torch.linalg_solve(A, B, true); }
    public static Tensor linalg_solve(Tensor A, Tensor B, boolean left) {
        return torch.linalg_solve(A, B, left);
    }
    public static Tensor linalg_svdvals(Tensor A) { return torch.linalg_svdvals(A); }
    public static Tensor linalg_cond(Tensor input) { return torch.linalg_cond(input); }
    public static Tensor linalg_cond(Tensor input, String p) { return torch.linalg_cond(input, p); }
    public static Tensor linalg_cross(Tensor input, Tensor other, long dim) {
        return torch.linalg_cross(input, other, dim);
    }
    public static Tensor linalg_cross(Tensor input, Tensor other) {
        return linalg_cross(input, other, -1);
    }
    public static Tensor linalg_householder_product(Tensor input, Tensor tau) {
        return torch.linalg_householder_product(input, tau);
    }
    public static Tensor linalg_matrix_rank(Tensor input) { return torch.linalg_matrix_rank(input); }
    public static Tensor linalg_matrix_exp(Tensor input) { return torch.linalg_matrix_exp(input); }
    public static Tensor linalg_matrix_power(Tensor input, long n) {
        return torch.linalg_matrix_power(input, n);
    }
    public static Tensor linalg_multi_dot(Tensor... tensors) {
        TensorVector list = new TensorVector();
        for (int i = 0; i < tensors.length; i++) list.push_back(tensors[i]);
        return torch.linalg_multi_dot(list);
    }
    public static Tensor linalg_matmul(Tensor input, Tensor other) {
        return torch.linalg_matmul(input, other);
    }
    public static Tensor linalg_diagonal(Tensor input, long offset, long dim1, long dim2) {
        return torch.linalg_diagonal(input, offset, dim1, dim2);
    }
    public static Tensor linalg_vecdot(Tensor x, Tensor y, long dim) {
        return torch.linalg_vecdot(x, y, dim);
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
    public static Tensor special_gammainc(Tensor input, Tensor other) { return torch.special_gammainc(input, other); }
    public static Tensor special_gammaincc(Tensor input, Tensor other) { return torch.special_gammaincc(input, other); }
    public static Tensor special_gammaln(Tensor input) { return torch.special_gammaln(input); }
    public static Tensor special_log1p(Tensor input) { return torch.special_log1p(input); }
    public static Tensor special_logit(Tensor input) { return torch.special_logit(input, new DoubleOptional(-1.0)); }
    public static Tensor special_logit(Tensor input, double eps) { return torch.special_logit(input, new DoubleOptional(eps)); }
    public static Tensor special_log_ndtr(Tensor input) { return torch.special_log_ndtr(input); }
    public static Tensor special_log_softmax(Tensor input, long dim, ScalarTypeOptional dtype) {
        return torch.special_log_softmax(input, dim, dtype);
    }
    public static Tensor special_logsumexp(Tensor input, long dim, boolean keepdim) {
        return torch.special_logsumexp(input, new long[]{dim}, keepdim);
    }
    public static Tensor special_modified_bessel_i0(Tensor input) { return torch.special_modified_bessel_i0(input); }
    public static Tensor special_modified_bessel_i1(Tensor input) { return torch.special_modified_bessel_i1(input); }
    public static Tensor special_modified_bessel_k0(Tensor input) { return torch.special_modified_bessel_k0(input); }
    public static Tensor special_modified_bessel_k1(Tensor input) { return torch.special_modified_bessel_k1(input); }
    public static Tensor special_multigammaln(Tensor input, long p) { return torch.special_multigammaln(input, p); }
    public static Tensor special_ndtr(Tensor input) { return torch.special_ndtr(input); }
    public static Tensor special_ndtri(Tensor input) { return torch.special_ndtri(input); }
    public static Tensor special_polygamma(Tensor input, long n) { return torch.special_polygamma(n, input); }
    public static Tensor special_psi(Tensor input) { return torch.special_psi(input); }
    public static Tensor special_round(Tensor input) { return torch.special_round(input); }
    public static Tensor special_sinc(Tensor input) { return torch.special_sinc(input); }
    public static Tensor special_softmax(Tensor input, long dim, ScalarTypeOptional dtype) {
        return torch.special_softmax(input, dim, dtype);
    }
    public static Tensor special_xlog1py(Tensor input, Tensor other) { return torch.special_xlog1py(input, other); }
    public static Tensor special_xlogy(Tensor input, Tensor other) { return torch.special_xlogy(input, other); }
    public static Tensor special_zeta(Tensor input, Tensor other) { return torch.special_zeta(input, other); }

    // ========================================================================
    // BRO / GINI (existing helpers)
    // ========================================================================

    public static Tensor bro_penalty(Tensor x) {
        NormalizeFuncOptions opt = new NormalizeFuncOptions();
        opt.p().put(2);
        opt.dim().put(0);
        Tensor xNorm = torch.normalize(x, opt);
        Tensor corr = xNorm.t().matmul(xNorm);
        Tensor eye = torch.eye(x.size(1), x.options());
        return corr.sub(eye).norm();
    }

    public static Tensor gini(Tensor x) {
        Tensor xFlat = x.abs().view(-1).add(new Scalar(1e-6));
        long n = xFlat.size(0);
        Tensor xSorted = torch.sort(xFlat).get0();
        Tensor index = torch.arange(new Scalar(1), new Scalar(n + 1), x.options());
        Tensor num = index.mul(xSorted).sum().mul(new Scalar(2.0));
        Tensor den = xSorted.sum().mul(new Scalar((double) n));
        return num.div(den).sub(new Scalar((double) (n + 1) / n));
    }

    private static LossReduction toLossReduction(long reduction) {
        if (reduction == 0) return new LossReduction(new kNone());
        if (reduction == 2) return new LossReduction(new kSum());
        return new LossReduction(new kMean());  // 1 or default
    }

    private static KLDivLossReduction toKLDivLossReduction(long reduction) {
        if (reduction == 0) return new KLDivLossReduction(new kNone());
        if (reduction == 2) return new KLDivLossReduction(new kSum());
        if (reduction == 3) return new KLDivLossReduction(new kMean());
        return new KLDivLossReduction(new kBatchMean());  // 1 or default
    }

    private static EmbeddingBagMode toEmbeddingBagMode(long mode) {
        switch ((int) mode) {
            case 0: return new EmbeddingBagMode(new org.bytedeco.pytorch.enumtype.kSum());
            case 1: return new EmbeddingBagMode(new org.bytedeco.pytorch.enumtype.kMean());
            case 2: return new EmbeddingBagMode(new org.bytedeco.pytorch.enumtype.kMax());
            default: return new EmbeddingBagMode(new org.bytedeco.pytorch.enumtype.kMean());
        }
    }

    // ========================================================================
    // OPTIONS-BASED OVERLOADS - Single Tensor Input
    // ========================================================================

    // ========================================================================
    // OPTIONS-BASED OVERLOADS - (input, options)
    // ========================================================================

    public static Tensor elu(Tensor input, ELUOptions options) {
        return torch.elu(input, options);
    }

    public static Tensor selu(Tensor input, SELUOptions options) {
        return torch.selu(input, options);
    }

    public static Tensor hardshrink(Tensor input, HardshrinkOptions options) {
        return torch.hardshrink(input, options);
    }

    public static Tensor hardtanh(Tensor input, HardtanhOptions options) {
        return torch.hardtanh(input, options);
    }

    public static Tensor leaky_relu(Tensor input, LeakyReLUOptions options) {
        return torch.leaky_relu(input, options);
    }

    public static Tensor gumbel_softmax(Tensor input, GumbelSoftmaxFuncOptions options) {
        return torch.gumbel_softmax(input, options);
    }

    public static Tensor softmax(Tensor input, SoftmaxFuncOptions options) {
        return torch.softmax(input, options);
    }

    public static Tensor softmin(Tensor input, SoftminFuncOptions options) {
        return torch.softmin(input, options);
    }

    public static Tensor log_softmax(Tensor input, LogSoftmaxFuncOptions options) {
        return torch.log_softmax(input, options);
    }

    public static Tensor glu(Tensor input, GLUOptions options) {
        return torch.glu(input, options);
    }

    public static Tensor gelu(Tensor input, GELUOptions options) {
        return torch.gelu(input, options);
    }

    public static Tensor relu(Tensor input, ReLUOptions options) {
        return torch.relu(input, options);
    }

    public static Tensor relu6(Tensor input, ReLU6Options options) {
        return torch.relu6(input, options);
    }

    public static Tensor rrelu(Tensor input, RReLUFuncOptions options) {
        return torch.rrelu(input, options);
    }

    public static Tensor celu(Tensor input, CELUOptions options) {
        return torch.celu(input, options);
    }

    public static Tensor softplus(Tensor input, SoftplusOptions options) {
        return torch.softplus(input, options);
    }

    public static Tensor softshrink(Tensor input, SoftshrinkOptions options) {
        return torch.softshrink(input, options);
    }

    public static Tensor threshold(Tensor input, ThresholdOptions options) {
        return torch.threshold(input, options);
    }

    public static Tensor instance_norm(Tensor input, InstanceNormFuncOptions options) {
        return torch.instance_norm(input, options);
    }

    public static Tensor layer_norm(Tensor input, LayerNormFuncOptions options) {
        return torch.layer_norm(input, options);
    }

    public static Tensor group_norm(Tensor input, GroupNormFuncOptions options) {
        return torch.group_norm(input, options);
    }

    public static Tensor normalize(Tensor input, NormalizeFuncOptions options) {
        return torch.normalize(input, options);
    }

    public static Tensor dropout(Tensor input, DropoutFuncOptions options) {
        return torch.dropout(input, options);
    }

    public static Tensor alpha_dropout(Tensor input, AlphaDropoutFuncOptions options) {
        return torch.alpha_dropout(input, options);
    }

    public static Tensor feature_alpha_dropout(Tensor input, FeatureAlphaDropoutFuncOptions options) {
        return torch.feature_alpha_dropout(input, options);
    }

    public static Tensor pad(Tensor input, PadFuncOptions options) {
        return torch.pad(input, options);
    }

    public static Tensor interpolate(Tensor input, InterpolateFuncOptions options) {
        return torch.interpolate(input, options);
    }

    public static Tensor grid_sample(Tensor input, Tensor grid, GridSampleFuncOptions options) {
        return torch.grid_sample(input, grid, options);
    }

    public static Tensor avg_pool1d(Tensor input, AvgPool1dOptions options) {
        return torch.avg_pool1d(input, options);
    }

    public static Tensor avg_pool2d(Tensor input, AvgPool2dOptions options) {
        return torch.avg_pool2d(input, options);
    }

    public static Tensor avg_pool3d(Tensor input, AvgPool3dOptions options) {
        return torch.avg_pool3d(input, options);
    }

    public static Tensor max_pool1d(Tensor input, MaxPool1dOptions options) {
        return torch.max_pool1d(input, options);
    }

    public static Tensor max_pool2d(Tensor input, MaxPool2dOptions options) {
        return torch.max_pool2d(input, options);
    }

    public static Tensor max_pool3d(Tensor input, MaxPool3dOptions options) {
        return torch.max_pool3d(input, options);
    }

    public static Tensor adaptive_max_pool1d(Tensor input, AdaptiveMaxPool1dOptions options) {
        return torch.adaptive_max_pool1d(input, options);
    }

    public static Tensor adaptive_max_pool2d(Tensor input, AdaptiveMaxPool2dOptions options) {
        return torch.adaptive_max_pool2d(input, options);
    }

    public static Tensor adaptive_max_pool3d(Tensor input, AdaptiveMaxPool3dOptions options) {
        return torch.adaptive_max_pool3d(input, options);
    }

    public static Tensor adaptive_avg_pool1d(Tensor input, AdaptiveAvgPool1dOptions options) {
        return torch.adaptive_avg_pool1d(input, options);
    }

    public static Tensor adaptive_avg_pool2d(Tensor input, AdaptiveAvgPool2dOptions options) {
        return torch.adaptive_avg_pool2d(input, options);
    }

    public static Tensor adaptive_avg_pool3d(Tensor input, AdaptiveAvgPool3dOptions options) {
        return torch.adaptive_avg_pool3d(input, options);
    }

    public static Tensor fractional_max_pool2d(Tensor input, FractionalMaxPool2dOptions options) {
        return torch.fractional_max_pool2d(input, options);
    }

    public static Tensor fractional_max_pool3d(Tensor input, FractionalMaxPool3dOptions options) {
        return torch.fractional_max_pool3d(input, options);
    }

    public static Tensor lp_pool1d(Tensor input, LPPool1dOptions options) {
        return torch.lp_pool1d(input, options);
    }

    public static Tensor lp_pool2d(Tensor input, LPPool2dOptions options) {
        return torch.lp_pool2d(input, options);
    }

    public static Tensor lp_pool3d(Tensor input, LPPool3dOptions options) {
        return torch.lp_pool3d(input, options);
    }

    public static Tensor fold(Tensor input, FoldOptions options) {
        return torch.fold(input, options);
    }

    public static Tensor unfold(Tensor input, UnfoldOptions options) {
        return torch.unfold(input, options);
    }


    // ========================================================================
    // OPTIONS-BASED OVERLOADS - (input, indices, options)
    // ========================================================================

    public static Tensor max_unpool1d(Tensor input, Tensor indices, MaxUnpool1dFuncOptions options) {
        return torch.max_unpool1d(input, indices, options);
    }

    public static Tensor max_unpool2d(Tensor input, Tensor indices, MaxUnpool2dFuncOptions options) {
        return torch.max_unpool2d(input, indices, options);
    }

    public static Tensor max_unpool3d(Tensor input, Tensor indices, MaxUnpool3dFuncOptions options) {
        return torch.max_unpool3d(input, indices, options);
    }

    // ========================================================================
    // OPTIONS-BASED OVERLOADS - (input, target, options)
    // ========================================================================

    public static Tensor cosine_similarity(Tensor input, Tensor target, CosineSimilarityOptions options) {
        return torch.cosine_similarity(input, target, options);
    }

    public static Tensor pairwise_distance(Tensor input, Tensor target, PairwiseDistanceOptions options) {
        return torch.pairwise_distance(input, target, options);
    }

    public static Tensor binary_cross_entropy(Tensor input, Tensor target, BCELossOptions options) {
        return torch.binary_cross_entropy(input, target, options);
    }

    public static Tensor kl_div(Tensor input, Tensor target, KLDivLossOptions options) {
        return torch.kl_div(input, target, options);
    }

    public static Tensor mse_loss(Tensor input, Tensor target, MSELossOptions options) {
        return torch.mse_loss(input, target, options);
    }

    public static Tensor hinge_embedding_loss(Tensor input, Tensor target, HingeEmbeddingLossOptions options) {
        return torch.hinge_embedding_loss(input, target, options);
    }

    public static Tensor multi_margin_loss(Tensor input, Tensor target, MultiMarginLossOptions options) {
        return torch.multi_margin_loss(input, target, options);
    }

    public static Tensor l1_loss(Tensor input, Tensor target, L1LossOptions options) {
        return torch.l1_loss(input, target, options);
    }

    public static Tensor smooth_l1_loss(Tensor input, Tensor target, SmoothL1LossOptions options) {
        return torch.smooth_l1_loss(input, target, options);
    }

    public static Tensor huber_loss(Tensor input, Tensor target, HuberLossOptions options) {
        return torch.huber_loss(input, target, options);
    }

    public static Tensor soft_margin_loss(Tensor input, Tensor target, SoftMarginLossOptions options) {
        return torch.soft_margin_loss(input, target, options);
    }

    public static Tensor multilabel_soft_margin_loss(Tensor input, Tensor target, MultiLabelSoftMarginLossOptions options) {
        return torch.multilabel_soft_margin_loss(input, target, options);
    }

    public static Tensor margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, MarginRankingLossOptions options) {
        return torch.margin_ranking_loss(input1, input2, target, options);
    }

    public static Tensor nll_loss(Tensor input, Tensor target, NLLLossOptions options) {
        return torch.nll_loss(input, target, options);
    }

    public static Tensor cross_entropy(Tensor input, Tensor target, CrossEntropyLossOptions options) {
        return torch.cross_entropy(input, target, options);
    }

    public static Tensor binary_cross_entropy_with_logits(Tensor input, Tensor target, BCEWithLogitsLossOptions options) {
        return torch.binary_cross_entropy_with_logits(input, target, options);
    }

    public static Tensor multilabel_margin_loss(Tensor input, Tensor target, MultiLabelMarginLossOptions options) {
        return torch.multilabel_margin_loss(input, target, options);
    }

    public static Tensor poisson_nll_loss(Tensor input, Tensor target, PoissonNLLLossOptions options) {
        return torch.poisson_nll_loss(input, target, options);
    }

    // ========================================================================
    // OPTIONS-BASED OVERLOADS - (input1, input2, target, options)
    // ========================================================================

    public static Tensor cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, CosineEmbeddingLossOptions options) {
        return torch.cosine_embedding_loss(input1, input2, target, options);
    }

    // ========================================================================
    // OPTIONS-BASED OVERLOADS - (anchor, positive, negative, options)
    // ========================================================================

    public static Tensor triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, TripletMarginLossOptions options) {
        return torch.triplet_margin_loss(anchor, positive, negative, options);
    }

    public static Tensor triplet_margin_with_distance_loss(Tensor anchor, Tensor positive, Tensor negative, TripletMarginWithDistanceLossOptions options) {
        return torch.triplet_margin_with_distance_loss(anchor, positive, negative, options);
    }

    // ========================================================================
    // OPTIONS-BASED OVERLOADS - (input, runningMean, runningVar, options)
    // ========================================================================

    public static Tensor batch_norm(Tensor input, Tensor runningMean, Tensor runningVar, BatchNormFuncOptions options) {
        return torch.batch_norm(input, runningMean, runningVar, options);
    }

    // ========================================================================
    // OPTIONS-BASED OVERLOADS - CTC Loss
    // ========================================================================

    public static Tensor ctc_loss(Tensor logProbs, Tensor targets, Tensor inputLengths, Tensor targetLengths, CTCLossOptions options) {
        return torch.ctc_loss(logProbs, targets, inputLengths, targetLengths, options);
    }

    // ========================================================================
    // OPTIONS-BASED OVERLOADS - Multihead Attention
    // ========================================================================

    public static T_TensorTensor_T multi_head_attention_forward(Tensor query, Tensor key, Tensor value, MultiheadAttentionForwardFuncOptions options) {
        return torch.multi_head_attention_forward(query, key, value, options);
    }

}
