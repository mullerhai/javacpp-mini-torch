/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
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
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.nn.modules.ConvTranspose2dImpl;
import org.bytedeco.pytorch.nn.modules.GroupNormImpl;
import org.bytedeco.pytorch.nn.modules.SiLUImpl;
import org.bytedeco.pytorch.enumtype.Conv2dPadding;
import org.bytedeco.pytorch.nn.options.Conv2dOptions;
import org.bytedeco.pytorch.nn.options.ConvTranspose2dOptions;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * AutoencoderKL — the Variational Autoencoder used in Stable Diffusion to encode
 * images into latent space and decode latents back to pixel space.
 *
 * <p>Architecture:
 * <pre>
 *  Encoder: [3, H, W] → [128, 256, 512, 512] → [4, H/8, W/8]
 *  Decoder: [4, H/8, W/8] → [512, 256, 128, 3] → [3, H, W]
 *  Latent factor: 8 (512×512 → 64×64)
 * </pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AutoencoderKL extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    // ── Config ─────────────────────────────────────────────────────

    public static class VAEConfig {
        private int inChannels = 3;
        private int latentChannels = 4;
        private int[] channelOut = new int[]{128, 256, 512, 512};
        private double latentFactor = 8.0;
        private double beta = 1.0;

        public int inChannels() { return inChannels; }
        public void inChannels(int v) { this.inChannels = v; }
        public int latentChannels() { return latentChannels; }
        public void latentChannels(int v) { this.latentChannels = v; }
        public int[] channelOut() { return channelOut; }
        public void channelOut(int[] v) { this.channelOut = v; }
        public double latentFactor() { return latentFactor; }
        public void latentFactor(double v) { this.latentFactor = v; }
        public double beta() { return beta; }
        public void beta(double v) { this.beta = v; }
    }

    // ── Helper Blocks ─────────────────────────────────────────────

    public static class ResBlock extends Module {
        private final Module norm1;
        private final Module act;
        private final Module conv1;
        private final Module norm2;
        private final Module conv2;
        private final Module convShort;

        public ResBlock(int channels) {
            super("ResBlock");
            this.norm1 = register_module("norm1", new GroupNormImpl(channels, channels));
            this.act = register_module("act", new SiLUImpl());
            this.conv1 = register_module("conv1", conv2d(channels, channels, 3, 1));
            this.norm2 = register_module("norm2", new GroupNormImpl(channels, channels));
            this.conv2 = register_module("conv2", conv2d(channels, channels, 3, 1));
            this.convShort = register_module("conv_short", conv2d(channels, channels, 1, 0));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor h = conv1.forward(act.forward(norm1.forward(x)));
            h = conv2.forward(act.forward(norm2.forward(h)));
            return h.add(convShort.forward(x));
        }
    }

    public static class DownBlock extends Module {
        private final Module res1;
        private final Module res2;
        private final Module down;

        public DownBlock(int inCh, int outCh) {
            super("DownBlock");
            this.res1 = register_module("res1", new ResBlock(inCh));
            this.res2 = register_module("res2", new ResBlock(outCh));
            this.down = register_module("down", conv2d(outCh, outCh, 3, 1));
        }

        @Override
        public Tensor forward(Tensor x) {
            return down.forward(res2.forward(res1.forward(x)));
        }
    }

    public static class UpBlock extends Module {
        private final Module res1;
        private final Module res2;
        private final Module up;

        public UpBlock(int inCh, int outCh) {
            super("UpBlock");
            this.res1 = register_module("res1", new ResBlock(inCh));
            this.res2 = register_module("res2", new ResBlock(outCh));
            this.up = register_module("up", convTranspose2d(inCh, outCh, 4, 2, 1));
        }

        @Override
        public Tensor forward(Tensor x) {
            return res2.forward(res1.forward(up.forward(x)));
        }
    }

    // Conv2d helper: padding is std::variant with 2D LongPointer — set after construction.
    static Conv2dImpl conv2d(int inCh, int outCh, int kernel, int stride) {
        long k2 = kernel;
        long s2 = stride;
        long p2 = kernel / 2;
        Conv2dOptions opt = new Conv2dOptions(inCh, outCh,
            new org.bytedeco.javacpp.LongPointer(new long[]{k2, k2}));
        opt.stride(new org.bytedeco.javacpp.LongPointer(new long[]{s2, s2}));
        opt.padding().put(new org.bytedeco.javacpp.LongPointer(new long[]{p2, p2}));
        opt.bias(false);
        return new Conv2dImpl(opt);
    }

    static ConvTranspose2dImpl convTranspose2d(int inCh, int outCh, int kernel, int stride, int pad) {
        long k2 = kernel;
        long s2 = stride;
        long p2 = pad;
        return new ConvTranspose2dImpl(
            new ConvTranspose2dOptions(inCh, outCh, new org.bytedeco.javacpp.LongPointer(new long[]{k2, k2}))
                .stride(new org.bytedeco.javacpp.LongPointer(new long[]{s2, s2}))
                .padding(new org.bytedeco.javacpp.LongPointer(new long[]{p2, p2})));
    }

    // ── Encoder ───────────────────────────────────────────────────

    public static class Encoder extends Module {
        private final Module convIn;
        private final List<Module> downBlocks = new ArrayList<>();
        private final Module midRes1;
        private final Module midNorm;
        private final Module midConv;

        public Encoder(int inChannels, int latentChannels, int[] ch) {
            super("Encoder");
            this.convIn = register_module("conv_in", conv2d(inChannels, ch[0], 3, 1));

            for (int i = 0; i < ch.length - 1; i++) {
                downBlocks.add(register_module("down_blocks_" + i,
                    new DownBlock(ch[i], ch[i + 1])));
            }

            this.midRes1 = register_module("mid_res1", new ResBlock(ch[ch.length - 1]));
            this.midNorm = register_module("mid_norm", new GroupNormImpl(ch[ch.length - 1], ch[ch.length - 1]));
            register_module("mid_act", new SiLUImpl());
            this.midConv = register_module("mid_conv", conv2d(ch[ch.length - 1], ch[ch.length - 1], 3, 1));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor h = ((Conv2dImpl) convIn).forward(x);
            for (Module block : downBlocks) {
                h = block.forward(h);
            }
            h = ((ResBlock) midRes1).forward(h);
            h = ((GroupNormImpl) midNorm).forward(h);
            h = gelu(h);
            h = ((Conv2dImpl) midConv).forward(h);
            return h;
        }
    }

    // ── Decoder ───────────────────────────────────────────────────

    public static class Decoder extends Module {
        private final Module convIn;
        private final List<Module> upBlocks = new ArrayList<>();
        private final Module midRes1;
        private final Module midNorm;
        private final Module midConv;
        private final Module convOut;

        public Decoder(int inChannels, int latentChannels, int[] ch) {
            super("Decoder");
            this.convIn = register_module("conv_in", conv2d(latentChannels, ch[0], 3, 1));

            this.midRes1 = register_module("mid_res1", new ResBlock(ch[0]));
            this.midNorm = register_module("mid_norm", new GroupNormImpl(ch[0], ch[0]));
            register_module("mid_act", new SiLUImpl());
            this.midConv = register_module("mid_conv", conv2d(ch[0], ch[0], 3, 1));

            for (int i = 0; i < ch.length - 1; i++) {
                upBlocks.add(register_module("up_blocks_" + i,
                    new UpBlock(ch[i], ch[i + 1])));
            }

            this.convOut = register_module("conv_out", conv2d(ch[ch.length - 1], inChannels, 3, 1));
        }

        @Override
        public Tensor forward(Tensor z) {
            Tensor h = ((Conv2dImpl) convIn).forward(z);
            h = ((ResBlock) midRes1).forward(h);
            h = ((GroupNormImpl) midNorm).forward(h);
            h = gelu(h);
            h = ((Conv2dImpl) midConv).forward(h);
            for (Module block : upBlocks) {
                h = block.forward(h);
            }
            h = ((Conv2dImpl) convOut).forward(gelu(h));
            return h;
        }
    }

    // ── Fields ────────────────────────────────────────────────────

    private final VAEConfig config;
    private final Encoder encoder;
    private final Decoder decoder;
    private final Module quantConv;
    private final Module logSigConv;
    private final Module postQuantConv;

    // ── Constructor ────────────────────────────────────────────────

    public AutoencoderKL(VAEConfig config) {
        super("AutoencoderKL");
        this.config = Objects.requireNonNull(config);

        int[] ch = config.channelOut();
        int latentCh = config.latentChannels();

        this.encoder = register_module("encoder",
            new Encoder(config.inChannels(), latentCh, ch));
        this.decoder = register_module("decoder",
            new Decoder(config.inChannels(), latentCh, ch));

        this.quantConv = register_module("quant_conv", conv2d(ch[ch.length - 1], latentCh, 1, 0));
        this.logSigConv = register_module("log_sig_conv", conv2d(ch[ch.length - 1], latentCh, 1, 0));
        this.postQuantConv = register_module("post_quant_conv", conv2d(latentCh, ch[0], 1, 0));
    }

    // ── Encode / Decode ───────────────────────────────────────────

    /** Encode image tensor to latent distribution parameters. */
    public Tensor encode(Tensor imageTensor) {
        return ((Conv2dImpl) quantConv).forward(encoder.forward(imageTensor));
    }

    /** Encode and sample from the latent distribution (training). */
    public Tensor encodeAndSample(Tensor imageTensor) {
        Tensor h = encoder.forward(imageTensor);
        Tensor mean = ((Conv2dImpl) quantConv).forward(h);
        Tensor logvar = ((Conv2dImpl) logSigConv).forward(h);
        return mean.add(torch.randn_like(mean).mul(exp(logvar.mul(new Scalar(0.5)))));
    }

    /** Decode latent tensor to image. */
    public Tensor decode(Tensor z) {
        z = ((Conv2dImpl) postQuantConv).forward(z);
        return decoder.forward(z);
    }

    /** Full encode → sample → decode (training forward). */
    public Tensor forward(Tensor imageTensor) {
        return decode(encodeAndSample(imageTensor));
    }

    /** KL divergence loss. */
    public Tensor klLoss(Tensor imageTensor) {
        Tensor h = encoder.forward(imageTensor);
        Tensor mean = ((Conv2dImpl) quantConv).forward(h);
        Tensor logvar = ((Conv2dImpl) logSigConv).forward(h);
        return logvar.add(new Scalar(1.0)).sub(mean.pow(new Scalar(2.0))).sub(exp(logvar)).neg().sub(new Scalar(1.0)).mul(new Scalar(0.5)).sum().div(new Scalar(mean.size(0)));
    }

    public VAEConfig config() { return config; }
}
