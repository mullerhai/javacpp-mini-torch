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
 *     http://www.gnu.org/licenses/
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.diffusion;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.util.Objects;

/**
 * Diffusion schedulers: manage noise addition and removal during inference.
 *
 * <p>Supported schedules:
 * <ul>
 *   <li>{@link DDPMScheduler} — Denoising Diffusion Probabilistic Models</li>
 *   <li>{@link DDIMScheduler} — Denoising Diffusion Implicit Models</li>
 *   <li>{@link EulerDiscreteScheduler} — Euler method for ODE integration</li>
 * </ul>
 *
 * <p>Usage pattern:
 * <pre>{@code
 * Scheduler scheduler = new EulerDiscreteScheduler();
 * scheduler.setTimesteps(50);
 *
 * for (int i = 0; i < 50; i++) {
 *     long t = scheduler.timestepAt(i);
 *     Tensor noisePred = unet.forward(latent, tensor(t), textCond);
 *     latent = scheduler.step(noisePred, t, latent);
 * }
 * }</pre>
 */
public abstract class Scheduler {

    // ── Config ───────────────────────────────────────────────────

    public static class SchedulerConfig {
        public int numTrainTimesteps = 1000;
        public double betaStart = 0.00085;
        public double betaEnd = 0.012;
        public String betaSchedule = "scaled_linear";
    }

    // ── Fields ──────────────────────────────────────────────────

    protected int numTrainTimesteps;
    protected int numInferenceTimesteps;
    protected double[] alphasCumprod;
    protected double[] betas;
    protected long[] inferenceTimesteps;

    public long timestepAt(int i) { return inferenceTimesteps[i]; }
    public int numInferenceTimesteps() { return numInferenceTimesteps; }

    /** Set inference timesteps (e.g. 50 steps out of 1000). */
    public abstract void setTimesteps(int numSteps);

    /**
     * Denoising step: given model prediction and current latent, produce next latent.
     *
     * @param modelOutput predicted noise
     * @param timestep    current timestep index
     * @param sample      current latent sample
     * @return next latent sample
     */
    public abstract Tensor step(Tensor modelOutput, long timestep, Tensor sample);

    /** Add noise to a clean image at a given timestep. */
    public abstract Tensor addNoise(Tensor originalSamples, Tensor noise, Tensor timesteps);

    protected double getAlphaCumprod(int t) {
        if (t < 0 || t >= alphasCumprod.length) return 1.0;
        return alphasCumprod[t];
    }

    static long[] computeBetasInt(SchedulerConfig config) {
        int n = config.numTrainTimesteps;
        long[] betasLong = new long[n];
        if ("linear".equals(config.betaSchedule)) {
            for (int i = 0; i < n; i++) {
                betasLong[i] = (long) ((config.betaStart
                    + (config.betaEnd - config.betaStart) * i / (n - 1)) * 1000);
            }
        } else {
            for (int i = 0; i < n; i++) {
                double t = i / (double) n;
                double beta = config.betaStart + (config.betaEnd - config.betaStart) * t;
                betasLong[i] = (long) (beta * 1000);
            }
        }
        return betasLong;
    }

    static long[] extractLongs(Tensor t) {
        long[] result = new long[(int) t.numel()];
        for (int i = 0; i < result.length; i++) {
            result[i] = t.reshape(-1).select(0, i).item().toLong();
        }
        return result;
    }

    // ── DDPM Scheduler ─────────────────────────────────────────────

    /** Classic DDPM scheduler with linear beta schedule. */
    public static class DDPMScheduler extends Scheduler {
        private final SchedulerConfig config;

        public DDPMScheduler() { this(new SchedulerConfig()); }
        public DDPMScheduler(SchedulerConfig config) {
            this.config = Objects.requireNonNull(config);
            this.numTrainTimesteps = config.numTrainTimesteps;

            long[] betasLong = computeBetasInt(config);
            this.betas = new double[betasLong.length];
            for (int i = 0; i < betasLong.length; i++) {
                this.betas[i] = betasLong[i] / 1000.0;
            }

            this.alphasCumprod = new double[numTrainTimesteps];
            alphasCumprod[0] = 1.0 - betas[0];
            for (int i = 1; i < numTrainTimesteps; i++) {
                alphasCumprod[i] = alphasCumprod[i - 1] * (1.0 - betas[i]);
            }
        }

        @Override
        public void setTimesteps(int numSteps) {
            this.numInferenceTimesteps = numSteps;
            this.inferenceTimesteps = new long[numSteps];
            long stepSize = numTrainTimesteps / numSteps;
            for (int i = 0; i < numSteps; i++) {
                inferenceTimesteps[i] = Math.max(0, numTrainTimesteps - 1 - i * stepSize);
            }
        }

        @Override
        public Tensor step(Tensor modelOutput, long timestep, Tensor sample) {
            int t = (int) timestep;
            double ac = getAlphaCumprod(t);
            double acPrev = getAlphaCumprod(Math.max(0, t - 1));
            double beta = betas[t];

            double sqrtAc = Math.sqrt(ac);
            double sqrtOneMinusAc = Math.sqrt(1.0 - ac);

            // Predicted x_0
            Tensor noiseTimesSqrtOneMinusAc = modelOutput.mul(new Scalar(sqrtOneMinusAc));
            Tensor predX0 = sample.sub(noiseTimesSqrtOneMinusAc).div(new Scalar(sqrtAc));

            // Prev sample
            double predCoeff = Math.sqrt(acPrev) * beta / (1.0 - ac);
            double sampleCoeff = Math.sqrt(ac) * (1.0 - acPrev) / (1.0 - ac);
            double noiseCoeff = Math.sqrt(1.0 - acPrev) * Math.sqrt(beta) / (1.0 - ac);

            Tensor t1 = predX0.mul(new Scalar(predCoeff));
            Tensor t2 = sample.mul(new Scalar(sampleCoeff));
            Tensor t3 = torch.randn_like(sample).mul(new Scalar(noiseCoeff));
            return t1.add(t2).add(t3);
        }

        @Override
        public Tensor addNoise(Tensor originalSamples, Tensor noise, Tensor timesteps) {
            long[] ts = extractLongs(timesteps);
            int B = (int) originalSamples.shape()[0];
            Tensor result = torch.zeros_like(originalSamples);
            for (int i = 0; i < B; i++) {
                long t = ts[i];
                double ac = getAlphaCumprod((int) t);
                double sqrtAc = Math.sqrt(ac);
                double sqrtOneMinusAc = Math.sqrt(1.0 - ac);
                Tensor orig = originalSamples.narrow(0, i, 1);
                Tensor n = noise.narrow(0, i, 1);
                result.narrow(0, i, 1).copy_(orig.mul(new Scalar(sqrtAc)).add(n.mul(new Scalar(sqrtOneMinusAc))));
            }
            return result;
        }
    }

    // ── DDIM Scheduler ─────────────────────────────────────────────

    /** DDIM scheduler — enables fast sampling (10-50 steps). */
    public static class DDIMScheduler extends Scheduler {
        private final SchedulerConfig config;
        private double eta = 1.0;

        public DDIMScheduler() { this(new SchedulerConfig()); }
        public DDIMScheduler(SchedulerConfig config) {
            this.config = Objects.requireNonNull(config);
            this.numTrainTimesteps = config.numTrainTimesteps;

            long[] betasLong = computeBetasInt(config);
            this.betas = new double[betasLong.length];
            for (int i = 0; i < betasLong.length; i++) {
                this.betas[i] = betasLong[i] / 1000.0;
            }

            this.alphasCumprod = new double[numTrainTimesteps];
            alphasCumprod[0] = 1.0 - betas[0];
            for (int i = 1; i < numTrainTimesteps; i++) {
                alphasCumprod[i] = alphasCumprod[i - 1] * (1.0 - betas[i]);
            }
        }

        public void setEta(double eta) { this.eta = eta; }

        @Override
        public void setTimesteps(int numSteps) {
            this.numInferenceTimesteps = numSteps;
            this.inferenceTimesteps = new long[numSteps];
            long stepSize = numTrainTimesteps / numSteps;
            for (int i = 0; i < numSteps; i++) {
                inferenceTimesteps[i] = Math.max(0, numTrainTimesteps - 1 - i * stepSize);
            }
        }

        @Override
        public Tensor step(Tensor modelOutput, long timestep, Tensor sample) {
            int t = (int) timestep;
            double ac = getAlphaCumprod(t);
            double acPrev = getAlphaCumprod(Math.max(0, t - 1));

            double sqrtAc = Math.sqrt(ac);
            double sqrtAcPrev = Math.sqrt(acPrev);
            double sqrtOneMinusAc = Math.sqrt(1.0 - ac);
            double sqrtOneMinusAcPrev = Math.sqrt(1.0 - acPrev);

            // Predicted x_0
            Tensor noiseTimesSqrtOneMinusAc = modelOutput.mul(new Scalar(sqrtOneMinusAc));
            Tensor predX0 = sample.sub(noiseTimesSqrtOneMinusAc).div(new Scalar(sqrtAc));

            // Direction pointing to x_t
            Tensor predDir = modelOutput.mul(new Scalar(sqrtOneMinusAcPrev));

            // Variance
            double variance = (1.0 - acPrev) / (1.0 - ac) * (1.0 - ac / acPrev);
            double sqrtVariance = Math.sqrt(Math.max(0, variance));

            Tensor prevSample = predX0.mul(new Scalar(sqrtAcPrev)).add(predDir);

            if (eta > 0 && sqrtVariance > 0) {
                prevSample = prevSample.add(torch.randn_like(sample).mul(new Scalar(sqrtVariance)));
            }

            return prevSample;
        }

        @Override
        public Tensor addNoise(Tensor originalSamples, Tensor noise, Tensor timesteps) {
            long[] ts = extractLongs(timesteps);
            int B = (int) originalSamples.shape()[0];
            Tensor result = torch.zeros_like(originalSamples);
            for (int i = 0; i < B; i++) {
                long t = ts[i];
                double ac = getAlphaCumprod((int) t);
                Tensor orig = originalSamples.narrow(0, i, 1);
                Tensor n = noise.narrow(0, i, 1);
                result.narrow(0, i, 1).copy_(
                    orig.mul(new Scalar(Math.sqrt(ac)))
                        .add(n.mul(new Scalar(Math.sqrt(1.0 - ac)))));
            }
            return result;
        }
    }

    // ── Euler Discrete Scheduler ────────────────────────────────────

    /** Euler ODE solver — versatile, best quality-speed tradeoff. */
    public static class EulerDiscreteScheduler extends Scheduler {
        private final SchedulerConfig config;
        private double[] sigmas;

        public EulerDiscreteScheduler() { this(new SchedulerConfig()); }
        public EulerDiscreteScheduler(SchedulerConfig config) {
            this.config = Objects.requireNonNull(config);
            this.numTrainTimesteps = config.numTrainTimesteps;

            long[] betasLong = computeBetasInt(config);
            this.betas = new double[betasLong.length];
            for (int i = 0; i < betasLong.length; i++) {
                this.betas[i] = betasLong[i] / 1000.0;
            }

            this.alphasCumprod = new double[numTrainTimesteps];
            alphasCumprod[0] = 1.0 - betas[0];
            for (int i = 1; i < numTrainTimesteps; i++) {
                alphasCumprod[i] = alphasCumprod[i - 1] * (1.0 - betas[i]);
            }

            this.sigmas = new double[numTrainTimesteps];
            for (int i = 0; i < numTrainTimesteps; i++) {
                sigmas[i] = Math.sqrt(1.0 - alphasCumprod[i]);
            }
        }

        @Override
        public void setTimesteps(int numSteps) {
            this.numInferenceTimesteps = numSteps;
            this.inferenceTimesteps = new long[numSteps];
            this.sigmas = new double[numSteps];

            double sigmaMin = sigmas[sigmas.length - 1];
            double sigmaMax = sigmas[0];

            for (int i = 0; i < numSteps; i++) {
                double t = i / (double) (numSteps - 1);
                double sigma = sigmaMin * Math.pow(sigmaMax / sigmaMin, t);
                this.sigmas[i] = sigma;
                this.inferenceTimesteps[i] = findClosestSigmaIndex(sigma);
            }
        }

        private int findClosestSigmaIndex(double sigma) {
            int closest = 0;
            double minDiff = Double.MAX_VALUE;
            for (int i = 0; i < this.sigmas.length; i++) {
                double diff = Math.abs(this.sigmas[i] - sigma);
                if (diff < minDiff) { minDiff = diff; closest = i; }
            }
            return closest;
        }

        @Override
        public Tensor step(Tensor modelOutput, long timestep, Tensor sample) {
            int tIdx = (int) timestep;
            double sigma = this.sigmas[Math.min(tIdx, this.sigmas.length - 1)];
            return sample.sub(modelOutput.mul(new Scalar(sigma)));
        }

        @Override
        public Tensor addNoise(Tensor originalSamples, Tensor noise, Tensor timesteps) {
            return originalSamples.add(noise.mul(new Scalar(0.5)));
        }
    }
}
