/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option)
 * any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/licenses/
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.diffusion.modeling;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.nn.modules.GroupNormImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.SiLUImpl;
import org.bytedeco.pytorch.nn.options.Conv2dOptions;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.pytorch.enumtype.Conv2dPadding;



/**
 * UNet2DConditionModel — the denoising U-Net used in Stable Diffusion-style
 * text-to-image diffusion models.
 *
 * <p>Architecture overview:
 * <pre>
 *  Input: [B, 4, H/8, W/8] noisy latent + timestep embedding + text conditioning
 *
 *  Down blocks (encoder):
 *    Conv2d → ResBlock (with group norm + conv2d) → ... × 3
 *    Each level halves spatial dims
 *
 *  Middle:
 *    ResBlock → Attention → ResBlock
 *
 *  Up blocks (decoder):
 *    Upsample → Conv2d → ResBlock → ... × 3
 *    Skip connections from encoder
 *
 *  Output: [B, 4, H/8, W/8] predicted noise
 * </pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class UNet2DConditionModel extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    // Conv2d helper: returns Conv2dOptions (pass directly to new Conv2dImpl or register_module).
    static Conv2dOptions conv2d(int inCh, int outCh, int kernel) {
        long k2 = kernel;
        long p2 = kernel / 2;
        Conv2dOptions opt = new Conv2dOptions(inCh, outCh,
            new org.bytedeco.javacpp.LongPointer(new long[]{k2, k2}));
        opt.stride(new org.bytedeco.javacpp.LongPointer(new long[]{1, 1}));
        opt.padding().put(new org.bytedeco.javacpp.LongPointer(new long[]{p2, p2}));
        opt.bias(false);
        return opt;
    }

    // Conv2d helper without bias
    static Conv2dOptions conv2dNoBias(int inCh, int outCh, int kernel) {
        long k2 = kernel;
        long p2 = kernel / 2;
        Conv2dOptions opt = new Conv2dOptions(inCh, outCh,
            new org.bytedeco.javacpp.LongPointer(new long[]{k2, k2}));
        opt.stride(new org.bytedeco.javacpp.LongPointer(new long[]{1, 1}));
        opt.padding().put(new org.bytedeco.javacpp.LongPointer(new long[]{p2, p2}));
        opt.bias(false);
        return opt;
    }

    // Strided conv used for downsampling (kernel, stride>=2)
    static Conv2dOptions stridedConv(int inCh, int outCh, int kernel, int stride) {
        long k2 = kernel;
        long s2 = stride;
        long p2 = kernel / 2;
        Conv2dOptions opt = new Conv2dOptions(inCh, outCh,
            new org.bytedeco.javacpp.LongPointer(new long[]{k2, k2}));
        opt.stride(new org.bytedeco.javacpp.LongPointer(new long[]{s2, s2}));
        opt.padding().put(new org.bytedeco.javacpp.LongPointer(new long[]{p2, p2}));
        opt.bias(false);
        return opt;
    }

    // ── Config ─────────────────────────────────────────────────────

    public static class DiffusionUnetConfig {
        private int inChannels = 4;
        private int outChannels = 4;
        private int[] blockOutChannels = new int[]{320, 640, 1280};
        private int[] numLayers = new int[]{1, 1, 1};
        private int attentionHeadDim = 40; // single-head dim; #heads = blockOut / headDim
        private int[] crossAttentionDim = new int[]{1024};

        public int inChannels() { return inChannels; }
        public void inChannels(int v) { this.inChannels = v; }
        public int outChannels() { return outChannels; }
        public void outChannels(int v) { this.outChannels = v; }
        public int[] blockOutChannels() { return blockOutChannels; }
        public void blockOutChannels(int[] v) { this.blockOutChannels = v; }
        public int[] numLayers() { return numLayers; }
        public void numLayers(int[] v) { this.numLayers = v; }
        public int attentionHeadDim() { return attentionHeadDim; }
        public void attentionHeadDim(int v) { this.attentionHeadDim = v; }
        public int[] crossAttentionDim() { return crossAttentionDim; }
        public void crossAttentionDim(int[] v) { this.crossAttentionDim = v; }
    }

    private final DiffusionUnetConfig config;
    private final int timeEmbedDim;
    private final Module timeEmbedding;
    private final Module timeFc1;
    private final Module timeAct;
    private final Module timeFc2;
    private final Module convIn;
    private final List<Module> downBlocks = new ArrayList<>();
    private final Module midBlock;
    private final List<Module> upBlocks = new ArrayList<>();
    private final Module convOut;

    // ── Inner Classes ─────────────────────────────────────────────

    public static class SinusoidalPosEmb extends Module {
        private final int dim;

        public SinusoidalPosEmb(int dim) {
            super("SinusoidalPosEmb");
            this.dim = dim;
        }

        @Override
        public Tensor forward(Tensor x) {
            long b = x.size(0);
            int half = dim / 2;
            // freqs = exp(-log(10000) / (2*half) * i)  for i in [0, half)
            Tensor invFreq = torch.arange(new Scalar(0), new Scalar(half), new Scalar(1),
                    new TensorOptions().dtype(new ScalarTypeOptional(ScalarType.Float)));
            invFreq = invFreq.mul(new Scalar(2.0 * Math.PI / Math.log(10000.0)));
            invFreq = torch.neg(invFreq).exp();
            invFreq = invFreq.reshape(new long[]{1, half});

            Tensor t = x.reshape(new long[]{b, 1});  // [b, 1]
            Tensor a = t.mul(invFreq);              // [b, half]
            Tensor b_emb = torch.neg(t).mul(invFreq).exp();  // [b, half]
            return cat(new TensorVector(new Tensor[]{a, b_emb}), -1);  // [b, dim]
        }
    }

    public static class ResnetBlock2D extends Module {
        private final int inChannels;
        private final int outChannels;
        private final Module norm1;
        private final Module conv1;
        private final Module norm2;
        private final Module conv2;
        private final Module timeEmbProj;
        private final Module convShort;

        public ResnetBlock2D(int inChannels, int outChannels, int timeEmbDim) {
            super("ResnetBlock2D");
            this.inChannels = inChannels;
            this.outChannels = outChannels;

            this.norm1 = register_module("norm1", new GroupNormImpl(32, inChannels));
            this.conv1 = register_module("conv1", new Conv2dImpl(
                conv2dNoBias(inChannels, outChannels, 3)));
            this.timeEmbProj = register_module("time_emb_proj",
                new LinearImpl(new LinearOptions(timeEmbDim, outChannels)));
            this.norm2 = register_module("norm2", new GroupNormImpl(32, outChannels));
            this.conv2 = register_module("conv2", new Conv2dImpl(
                conv2dNoBias(outChannels, outChannels, 3)));

            if (inChannels != outChannels) {
                this.convShort = register_module("conv_shortcut",
                    new Conv2dImpl(conv2dNoBias(inChannels, outChannels, 1)));
            } else {
                this.convShort = null;
            }
        }

        @Override
        public Tensor forward(Tensor x, Tensor timeEmb) {
            Tensor h = conv1.forward(gelu(norm1.forward(x)));
            // Project time embedding to outChannels first, then broadcast-add to feature map.
            Tensor timeProj = ((LinearImpl) timeEmbProj).forward(timeEmb);
            h = h.add(timeProj.reshape(new long[]{1, -1, 1, 1}));
            h = conv2.forward(gelu(norm2.forward(h)));
            return h.add(convShort != null ? convShort.forward(x) : x);
        }
    }

    public static class SpatialSelfAttention extends Module {
        private final int channels;
        private final int heads;
        private final Module norm;
        private final Module toQ;
        private final Module toK;
        private final Module toV;
        private final Module projOut;

        public SpatialSelfAttention(int channels, int heads) {
            super("SpatialSelfAttention");
            this.channels = channels;
            this.heads = heads;

            this.norm = register_module("norm", new GroupNormImpl(32, channels));
            this.toQ = register_module("to_q", new LinearImpl(new LinearOptions(channels, channels).bias(false)));
            this.toK = register_module("to_k", new LinearImpl(new LinearOptions(channels, channels).bias(false)));
            this.toV = register_module("to_v", new LinearImpl(new LinearOptions(channels, channels).bias(false)));
            this.projOut = register_module("to_out", new LinearImpl(new LinearOptions(channels, channels).bias(false)));
        }

        @Override
        public Tensor forward(Tensor x) {
            long b = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
            long n = h * w;
            Tensor inNorm = norm.forward(x);
            // Reshape the 4D feature map into the [batch, seq, channels]
            // layout expected by the linear projections BEFORE invoking
            // them, otherwise the Linear layer treats `inNorm` as
            // [1*c*h, w] which has the wrong last dim.
            Tensor flat = inNorm.reshape(new long[]{b, c, n}).transpose(1, 2);  // [b, n, c]
            Tensor q = ((LinearImpl)toQ).forward(flat).reshape(new long[]{b, n, c});
            Tensor k = ((LinearImpl)toK).forward(flat).reshape(new long[]{b, n, c});
            Tensor v = ((LinearImpl)toV).forward(flat).reshape(new long[]{b, n, c});

            int hDim = (int) (c / heads);
            q = q.reshape(new long[]{b, n, heads, hDim}).transpose(1, 2);
            k = k.reshape(new long[]{b, n, heads, hDim}).transpose(1, 2);
            v = v.reshape(new long[]{b, n, heads, hDim}).transpose(1, 2);

            Tensor scale = tensor((float) (1.0 / Math.sqrt(hDim)));
            Tensor attn = softmax(matmul(q, k.transpose(-2, -1)).mul(scale), -1);
            Tensor out = matmul(attn, v).transpose(1, 2).reshape(new long[]{b, n, c}).to(torch.ScalarType.Float);
            return x.add(((LinearImpl)projOut).forward(out).reshape(new long[]{b, c, h, w}));
        }
    }

    public static class CrossAttentionModule extends Module {
        private final int dim;
        private final int heads;
        private final int dimHead;
        private final Module toQ;
        private final Module toK;
        private final Module toV;
        private final Module projOut;

        public CrossAttentionModule(int dim, int heads, int crossDim) {
            super("CrossAttention");
            this.dim = dim;
            this.heads = heads;
            this.dimHead = dim / heads;

            this.toQ = register_module("to_q", new LinearImpl(new LinearOptions(dim, dim).bias(false)));
            this.toK = register_module("to_k", new LinearImpl(new LinearOptions(crossDim, dim).bias(false)));
            this.toV = register_module("to_v", new LinearImpl(new LinearOptions(crossDim, dim).bias(false)));
            this.projOut = register_module("to_out", new LinearImpl(new LinearOptions(dim, dim).bias(false)));
        }

        @Override
        public Tensor forward(Tensor x, Tensor context) {
            long b = x.size(0), n = x.size(1);
            Tensor q = ((LinearImpl)toQ).forward(x).reshape(new long[]{b, n, heads, dimHead}).transpose(1, 2);
            Tensor k = ((LinearImpl)toK).forward(context).reshape(new long[]{b, context.size(1), heads, dimHead}).transpose(1, 2);
            Tensor v = ((LinearImpl)toV).forward(context).reshape(new long[]{b, context.size(1), heads, dimHead}).transpose(1, 2);

            // Ensure float dtype so the matmul keeps consistent types
            // (Linear weights are float, but tensor() with a double literal
            // produces a double tensor).
            Tensor scale = tensor((float) (1.0 / Math.sqrt(dimHead)));
            Tensor attn = softmax(matmul(q, k.transpose(-2, -1)).mul(scale), -1);
            Tensor contextOut = matmul(attn, v).transpose(1, 2).reshape(new long[]{b, n, dim}).to(torch.ScalarType.Float);
            return ((LinearImpl)projOut).forward(contextOut);
        }
    }

    public static class DownBlock2D extends Module {
        private final Module res1;
        private final Module res2;
        private final Module downsampler;

        public DownBlock2D(int inCh, int outCh, int timeEmbDim) {
            super("DownBlock2D");
            this.res1 = register_module("res1", new ResnetBlock2D(inCh, outCh, timeEmbDim));
            this.res2 = register_module("res2", new ResnetBlock2D(outCh, outCh, timeEmbDim));
            // Downsample by stride=2 convolution.
            this.downsampler = register_module("down",
                new Conv2dImpl(stridedConv(outCh, outCh, 3, 2)));
        }

        public List<Tensor> forwardDown(Tensor h, Tensor timeEmb, Tensor enc) {
            Tensor r1 = res1.forward(h, timeEmb);
            Tensor r2 = res2.forward(r1, timeEmb);
            // Return a single skip connection (post-resnet2) plus the
            // downsampled tensor that is fed to the next block.  The outer
            // forward pass concatenates this skip with the upsampled output
            // before each up-block.
            Tensor d  = downsampler.forward(r2);
            return List.of(r2, d);
        }
    }

    public static class CrossAttnDownBlock2D extends Module {
        private final Module res1;
        private final Module res2;
        private final Module attn1;
        private final Module attn2;
        private final Module downsampler;

        public CrossAttnDownBlock2D(int inCh, int outCh, int timeEmbDim, int crossDim, int heads) {
            super("CrossAttnDownBlock2D");
            this.res1 = register_module("res1", new ResnetBlock2D(inCh, outCh, timeEmbDim));
            this.res2 = register_module("res2", new ResnetBlock2D(outCh, outCh, timeEmbDim));
            this.attn1 = register_module("attn1", new CrossAttentionModule(outCh, heads, crossDim));
            this.attn2 = register_module("attn2", new CrossAttentionModule(outCh, heads, crossDim));
            this.downsampler = register_module("down",
                new Conv2dImpl(stridedConv(outCh, outCh, 3, 2)));
        }

        public List<Tensor> forwardCross(Tensor h, Tensor timeEmb, Tensor enc) {
            h = res1.forward(h, timeEmb);
            long b = h.size(0), c = h.size(1), hh = h.size(2), ww = h.size(3);
            Tensor h2d = h.reshape(new long[]{b, c, hh * ww}).transpose(1, 2);
            h2d = attn1.forward(h2d, enc).to(torch.ScalarType.Float);
            h = h2d.transpose(1, 2).reshape(new long[]{b, c, hh, ww});

            h = res2.forward(h, timeEmb);
            b = h.size(0); c = h.size(1); hh = h.size(2); ww = h.size(3);
            h2d = h.reshape(new long[]{b, c, hh * ww}).transpose(1, 2);
            h2d = attn2.forward(h2d, enc).to(torch.ScalarType.Float);
            h = h2d.transpose(1, 2).reshape(new long[]{b, c, hh, ww});

            return List.of(h, downsampler.forward(h));
        }
    }

    public static class UpBlock2D extends Module {
        private final Module res1;
        private final Module res2;
        private final Module attn1;
        private final Module attn2;

        public UpBlock2D(int inCh, int outCh, int timeEmbDim, int crossDim, int heads) {
            super("UpBlock2D");
            // inCh is the channel count after concatenating the skip connection
            // (previous block channels + skip channels). The first resnet
            // projects down to outCh; subsequent modules operate on outCh.
            this.res1 = register_module("res1", new ResnetBlock2D(inCh, outCh, timeEmbDim));
            this.res2 = register_module("res2", new ResnetBlock2D(outCh, outCh, timeEmbDim));
            this.attn1 = register_module("attn1", new CrossAttentionModule(outCh, heads, crossDim));
            this.attn2 = register_module("attn2", new CrossAttentionModule(outCh, heads, crossDim));
        }

        public Tensor forward(Tensor h, Tensor timeEmb, Tensor enc) {
            // The outer UNet forward handles upsampling and skip concatenation.
            // This block processes the concatenated tensor [h, skip] through
            // the residual + cross-attention stack.
            h = res1.forward(h, timeEmb);
            long b = h.size(0), c = h.size(1), hh = h.size(2), ww = h.size(3);
            Tensor h2d = h.reshape(new long[]{b, c, hh * ww}).transpose(1, 2);
            h2d = attn1.forward(h2d, enc);
            h = h2d.transpose(1, 2).reshape(new long[]{b, c, hh, ww});

            h = res2.forward(h, timeEmb);
            b = h.size(0); c = h.size(1); hh = h.size(2); ww = h.size(3);
            h2d = h.reshape(new long[]{b, c, hh * ww}).transpose(1, 2);
            h2d = attn2.forward(h2d, enc);
            h = h2d.transpose(1, 2).reshape(new long[]{b, c, hh, ww});

            return h;
        }
    }

    public static class UNetMidBlock2DCrossAttn extends Module {
        private final Module resBlock1;
        private final Module attn;
        private final Module resBlock2;

        public UNetMidBlock2DCrossAttn(int inCh, int timeEmbDim, int crossDim, int heads) {
            super("UNetMidBlock2DCrossAttn");
            this.resBlock1 = register_module("res1", new ResnetBlock2D(inCh, inCh, timeEmbDim));
            this.attn = register_module("attn", new SpatialSelfAttention(inCh, heads));
            this.resBlock2 = register_module("res2", new ResnetBlock2D(inCh, inCh, timeEmbDim));
        }

        @Override
        public Tensor forward(Tensor h, Tensor timeEmb, Tensor enc) {
            h = resBlock1.forward(h, timeEmb);
            h = attn.forward(h);
            h = resBlock2.forward(h, timeEmb);
            return h;
        }
    }

    // ── Constructor ─────────────────────────────────────────────────

    public UNet2DConditionModel(DiffusionUnetConfig config) {
        super("UNet2DConditionModel");
        this.config = Objects.requireNonNull(config);

        int[] chs = config.blockOutChannels();
        int base = chs[0];
        this.timeEmbedDim = base * 4;

        // Time embedding
        this.timeEmbedding = register_module("time_mlp", new Module("time_mlp"));
        this.timeFc1 = timeEmbedding.register_module("0", new LinearImpl(new LinearOptions(base, timeEmbedDim)));
        this.timeAct = timeEmbedding.register_module("1", new SiLUImpl());
        this.timeFc2 = timeEmbedding.register_module("2", new LinearImpl(new LinearOptions(timeEmbedDim, timeEmbedDim)));

        // Input conv
        this.convIn = register_module("conv_in",
            new Conv2dImpl(conv2d(config.inChannels(), base, 3)));

        // Down blocks
        int inCh = base;
        int heads = base / config.attentionHeadDim();
        List<Tensor> skipConnections = new ArrayList<>();

        for (int level = 0; level < chs.length; level++) {
            int outCh = chs[level];
            Module block;
            if (level == 0) {
                block = register_module("down_blocks_" + level,
                    new DownBlock2D(inCh, outCh, timeEmbedDim));
            } else {
                block = register_module("down_blocks_" + level,
                    new CrossAttnDownBlock2D(inCh, outCh, timeEmbedDim,
                        config.crossAttentionDim()[0], heads));
            }
            downBlocks.add(block);
            inCh = outCh;
            heads = outCh / config.attentionHeadDim();
        }

        // Mid block
        int midCh = chs[chs.length - 1];
        this.midBlock = register_module("mid_block",
            new UNetMidBlock2DCrossAttn(midCh, timeEmbedDim,
                config.crossAttentionDim()[0], heads));

        // Up blocks
        // upBlocks[i] mirrors downBlocks[chs.length - 1 - i].  The first up
        // level takes the mid-block output (midCh channels) and the deepest
        // skip from downBlocks[chs.length - 1].  Subsequent up levels consume
        // the previous up-block output plus the next shallower skip connection.
        int currentCh = midCh;
        for (int level = 0; level < chs.length; level++) {
            int skipCh = chs[chs.length - 1 - level];
            int outCh = (level == chs.length - 1) ? chs[0] : chs[chs.length - 2 - level];
            int blockInCh = currentCh + skipCh;
            Module block = register_module("up_blocks_" + level,
                new UpBlock2D(blockInCh, outCh, timeEmbedDim,
                    config.crossAttentionDim()[0], heads));
            upBlocks.add(block);
            currentCh = outCh;
            heads = outCh / config.attentionHeadDim();
        }

        // Output conv
        this.convOut = register_module("conv_out",
            new Conv2dImpl(conv2d(base, config.outChannels(), 3)));
    }

    // ── Forward ───────────────────────────────────────────────────

    public Tensor forward(Tensor sample, Tensor timestep, Tensor encoderHiddenStates) {
        // Time embedding
        long b = timestep.size(0);
        Tensor tEmb = timestep.reshape(new long[]{b}).to(torch.ScalarType.Float);
        // Use base channel count (320) for sinusoidal positional encoding,
        // which matches the input dim of timeFc1 (Linear(base, timeEmbedDim)).
        int base = this.timeEmbedDim / 4;
        Tensor t = new SinusoidalPosEmb(base).forward(tEmb);
        // Ensure float32 for linear ops (sinusoidal output may have been upcast)
        t = t.to(torch.ScalarType.Float);

        // Actually use the registered time_mlp
        t = ((LinearImpl) timeFc1).forward(t);
        t = ((SiLUImpl) timeAct).forward(t);
        t = ((LinearImpl) timeFc2).forward(t);

        // Cast encoder hidden states to float for the cross-attention paths.
        Tensor encStates = encoderHiddenStates.to(torch.ScalarType.Float);

        // Input conv
        Tensor h = ((Conv2dImpl) convIn).forward(sample.to(torch.ScalarType.Float));

        // Down blocks
        List<Tensor> hs = new ArrayList<>();
        hs.add(h);
        for (Module block : downBlocks) {
            List<Tensor> outs;
            if (block instanceof DownBlock2D) {
                outs = ((DownBlock2D) block).forwardDown(h, t, encStates);
            } else {
                outs = ((CrossAttnDownBlock2D) block).forwardCross(h, t, encStates);
            }
            // outs = [skip, downsampled]; skip is pushed as a skip connection,
            // downsampled becomes the next h.
            hs.add(outs.get(0));
            h = outs.get(1);
        }

        // Mid block
        h = ((UNetMidBlock2DCrossAttn) midBlock).forward(h, t, encStates);

        // Up blocks - consume one skip connection per up level.
        for (int i = 0; i < upBlocks.size(); i++) {
            Tensor skip = hs.get(hs.size() - 1 - i);
            if (h.size(2) != skip.size(2)) {
                h = upsample2x(h);
            }
            h = cat(new TensorVector(new Tensor[]{h, skip}), 1);
            h = ((UpBlock2D) upBlocks.get(i)).forward(h, t, encStates);
        }

        // Output
        h = gelu(h);
        return ((Conv2dImpl) convOut).forward(h);
    }

private static Tensor upsample2x(Tensor x) {
            // Reliable nearest-neighbor 2x upsample using torch.pixel_shuffle on a constructed tensor.
            // Strategy: replicate each pixel 2x2 by reshaping: [B, C, H, W] -> [B, C, 2, 2, H, W] -> [B, C, H, W, 2, 2] -> [B, C, H*2, W*2]
            long b = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
            // Ensure float for consistent dtype
            x = x.to(torch.ScalarType.Float);
            // Manual nearest-neighbor: expand spatial dims by 2 with repeat
            Tensor up = x.repeat_interleave(2, new LongOptional(2),new LongOptional());  // [b, c, h*2, w]
            up = up.repeat_interleave(2, new LongOptional(3) ,new LongOptional());       // [b, c, h*2, w*2]
            return up;
        }

    public DiffusionUnetConfig config() { return config; }
}
