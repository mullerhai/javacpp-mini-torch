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
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.BatchNorm2dImpl;
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

    // ── Config ─────────────────────────────────────────────────────

    public static class DiffusionUnetConfig {
        private int inChannels = 4;
        private int outChannels = 4;
        private int[] blockOutChannels = new int[]{320, 640, 1280};
        private int[] numLayers = new int[]{1, 1, 1};
        private int attentionHeadDim = 8;
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
            Tensor half = tensor(dim / 2, x.device());
            Tensor freqs = exp(
                neg(full(new long[]{dim / 2}, Math.log(10000.0), x.dtype(), x.device()))
                    .div(tensor(dim / 2, x.dtype()).fill(1.0))
                    .mul(tensor(dim / 2, x.dtype()).arange().add(0.5).neg().div(tensor(dim / 2, x.dtype()))));
            freqs = freqs.reshape(new long[]{1, dim / 2});
            Tensor t = x.reshape(new long[]{b, 1});
            return cat(new Tensor[]{t.mul(freqs), t.mul(freqs).neg().exp()}, -1);
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

            this.norm1 = register_module("norm1", new GroupNormImpl(inChannels, inChannels, 32));
            this.conv1 = register_module("conv1", new Conv2dImpl(
                new Conv2dOptions(inChannels, outChannels, 3).padding(1).bias(false)));
            this.timeEmbProj = register_module("time_emb_proj",
                new LinearImpl(new LinearOptions(timeEmbDim, outChannels)));
            this.norm2 = register_module("norm2", new GroupNormImpl(outChannels, outChannels, 32));
            this.conv2 = register_module("conv2", new Conv2dImpl(
                new Conv2dOptions(outChannels, outChannels, 3).padding(1).bias(false)));

            if (inChannels != outChannels) {
                this.convShort = register_module("conv_shortcut",
                    new Conv2dImpl(new Conv2dOptions(inChannels, outChannels, 1).bias(false)));
            } else {
                this.convShort = null;
            }
        }

        @Override
        public Tensor forward(Tensor x, Tensor timeEmb) {
            Tensor h = conv1.forward(gelu(norm1.forward(x)));
            h = h.add(timeEmb.reshape(new long[]{1, -1, 1, 1}));
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

            this.norm = register_module("norm", new GroupNormImpl(channels, channels, 32));
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
            Tensor q = ((LinearImpl)toQ).forward(inNorm).reshape(new long[]{b, n, c});
            Tensor k = ((LinearImpl)toK).forward(inNorm).reshape(new long[]{b, n, c});
            Tensor v = ((LinearImpl)toV).forward(inNorm).reshape(new long[]{b, n, c});

            int hDim = c / heads;
            q = q.reshape(new long[]{b, n, heads, hDim}).transpose(1, 2);
            k = k.reshape(new long[]{b, n, heads, hDim}).transpose(1, 2);
            v = v.reshape(new long[]{b, n, heads, hDim}).transpose(1, 2);

            Tensor scale = tensor(1.0 / Math.sqrt(hDim));
            Tensor attn = softmax(matmul(q, k.transpose(-2, -1)).mul(scale), -1);
            Tensor out = matmul(attn, v).transpose(1, 2).reshape(new long[]{b, n, c});
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

            Tensor scale = tensor(1.0 / Math.sqrt(dimHead));
            Tensor attn = softmax(matmul(q, k.transpose(-2, -1)).mul(scale), -1);
            return ((LinearImpl)projOut).forward(matmul(attn, v).transpose(1, 2).reshape(new long[]{b, n, dim}));
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
            this.downsampler = register_module("down", new Conv2dImpl(
                new Conv2dOptions(outCh, outCh, 3).padding(1)));
        }

        public List<Tensor> forward(Tensor h, Tensor timeEmb, Tensor enc) {
            return List.of(res1.forward(h, timeEmb), res2.forward(h, timeEmb),
                downsampler.forward(h));
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
            this.downsampler = register_module("down", new Conv2dImpl(
                new Conv2dOptions(outCh, outCh, 3).padding(1)));
        }

        public List<Tensor> forward(Tensor h, Tensor timeEmb, Tensor enc) {
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

            return List.of(h, downsampler.forward(h));
        }
    }

    public static class UpBlock2D extends Module {
        private final Module res1;
        private final Module res2;
        private final Module attn1;
        private final Module attn2;
        private final Module upsampler;

        public UpBlock2D(int inCh, int outCh, int timeEmbDim, int crossDim, int heads) {
            super("UpBlock2D");
            this.res1 = register_module("res1", new ResnetBlock2D(inCh, outCh, timeEmbDim));
            this.res2 = register_module("res2", new ResnetBlock2D(outCh, outCh, timeEmbDim));
            this.attn1 = register_module("attn1", new CrossAttentionModule(outCh, heads, crossDim));
            this.attn2 = register_module("attn2", new CrossAttentionModule(outCh, heads, crossDim));
            this.upsampler = register_module("up", new Conv2dImpl(
                new Conv2dOptions(inCh, inCh, 3).padding(1)));
        }

        public Tensor forward(Tensor h, Tensor timeEmb, Tensor enc) {
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

            return upsampler.forward(upsample2x(h));
        }

        private Tensor upsample2x(Tensor x) {
            long b = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
            Tensor up = torch.zeros(b, c, h * 2, w * 2, x.dtype(), x.device());
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    up.slice(2, i * h, (i + 1) * h).slice(3, j * w, (j + 1) * w)
                       .copy_(x.slice(2, i * h, i * h + 1).slice(3, j * w, j * w + 1));
                }
            }
            return up;
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
        timeEmbedding.register_module("0", new LinearImpl(new LinearOptions(base, timeEmbedDim)));
        timeEmbedding.register_module("1", new SiLUImpl());
        timeEmbedding.register_module("2", new LinearImpl(new LinearOptions(timeEmbedDim, timeEmbedDim)));

        // Input conv
        this.convIn = register_module("conv_in",
            new Conv2dImpl(new Conv2dOptions(config.inChannels(), base, 3).padding(1)));

        // Down blocks
        int inCh = base;
        int heads = base / config.attentionHeadDim();
        List<Tensor> skipConnections = new ArrayList<>();

        for (int level = 0; level < chs.length; level++) {
            int outCh = chs[level];
            Module block;
            if (level == 0) {
                block = register_module("down_blocks." + level,
                    new DownBlock2D(inCh, outCh, timeEmbedDim));
            } else {
                block = register_module("down_blocks." + level,
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
        heads = midCh / config.attentionHeadDim();
        for (int level = 0; level < chs.length; level++) {
            int outCh = level == chs.length - 1 ? chs[0] : chs[chs.length - 2 - level];
            Module block = register_module("up_blocks." + level,
                new UpBlock2D(midCh, outCh, timeEmbedDim,
                    config.crossAttentionDim()[0], heads));
            upBlocks.add(block);
            heads = outCh / config.attentionHeadDim();
        }

        // Output conv
        this.convOut = register_module("conv_out",
            new Conv2dImpl(new Conv2dOptions(base, config.outChannels(), 3).padding(1)));
    }

    // ── Forward ───────────────────────────────────────────────────

    public Tensor forward(Tensor sample, Tensor timestep, Tensor encoderHiddenStates) {
        // Time embedding
        long b = timestep.size(0);
        Tensor tEmb = timestep.reshape(new long[]{b}).to(torch.ScalarType.Float);
        Tensor t = new SinusoidalPosEmb(config.attentionHeadDim()).forward(tEmb);

        // Actually use the registered time_mlp
        t = ((LinearImpl) timeEmbedding.get_submodule("0")).forward(t);
        t = ((SiLUImpl) timeEmbedding.get_submodule("1")).forward(t);
        t = ((LinearImpl) timeEmbedding.get_submodule("2")).forward(t);

        // Input conv
        Tensor h = ((Conv2dImpl) convIn).forward(sample);

        // Down blocks
        List<Tensor> hs = new ArrayList<>();
        hs.add(h);
        for (Module block : downBlocks) {
            List<Tensor> outs;
            if (block instanceof DownBlock2D) {
                outs = ((DownBlock2D) block).forward(h, t, encoderHiddenStates);
            } else {
                outs = ((CrossAttnDownBlock2D) block).forward(h, t, encoderHiddenStates);
            }
            h = outs.get(outs.size() - 1);
            for (int i = 0; i < outs.size() - 1; i++) hs.add(outs.get(i));
        }

        // Mid block
        h = ((UNetMidBlock2DCrossAttn) midBlock).forward(h, t, encoderHiddenStates);

        // Up blocks
        for (int i = 0; i < upBlocks.size(); i++) {
            Tensor skip = hs.get(hs.size() - 1 - i * 2);
            if (h.size(2) != skip.size(2)) {
                h = upsample2x(h);
            }
            h = cat(new Tensor[]{h, skip}, 1);
            h = ((UpBlock2D) upBlocks.get(i)).forward(h, t, encoderHiddenStates);
        }

        // Output
        h = gelu(h);
        return ((Conv2dImpl) convOut).forward(h);
    }

    private static Tensor upsample2x(Tensor x) {
        long b = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3);
        Tensor up = torch.zeros(b, c, h * 2, w * 2, x.dtype(), x.device());
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                up.slice(2, i * h, (i + 1) * h).slice(3, j * w, (j + 1) * w)
                   .copy_(x.slice(2, i * h, i * h + 1).slice(3, j * w, j * w + 1));
            }
        }
        return up;
    }

    public DiffusionUnetConfig config() { return config; }
}
