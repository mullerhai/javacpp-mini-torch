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
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.dataframe.media;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.audio.utils.AudioTensors;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Specialized converter for Tensor to Audio conversions with advanced features.
 *
 * <p>Features:
 * <ul>
 *   <li>Waveform to AudioData conversion</li>
 *   <li>Multiple audio format support</li>
 *   <li>Resampling and channel conversion</li>
 *   <li>Audio feature extraction (spectrogram, mel-spectrogram)</li>
 *   <li>Batch processing support</li>
 * </ul>
 */
public final class TensorToAudioConverter {

    private TensorToAudioConverter() {}

    // ── Core Conversion Methods ───────────────────────────────────────────

    /**
     * Convert waveform tensor to AudioData.
     *
     * @param tensor Shape [T], [C,T], or [B,C,T]
     * @param sampleRate Sample rate in Hz
     * @return AudioData object
     */
    public static AudioData toAudioData(Tensor tensor, int sampleRate) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return AudioTensors.toAudioData(tensor, sampleRate);
    }

    /**
     * Convert AudioData to waveform tensor [C,T].
     *
     * @param audio Source AudioData
     * @return Tensor shape [C,T] float32
     */
    public static Tensor toTensor(AudioData audio) {
        Objects.requireNonNull(audio, "audio cannot be null");
        return AudioTensors.toTensor(audio);
    }

    /**
     * Convert waveform tensor to float array.
     *
     * @param tensor Waveform tensor [T] or [C,T]
     * @return Interleaved float samples
     */
    public static float[] toFloatArray(Tensor tensor) {
        Objects.requireNonNull(tensor, "tensor cannot be null");
        return AudioTensors.fromTensor(tensor);
    }

    /**
     * Create waveform tensor from interleaved float samples.
     *
     * @param samples Interleaved float samples
     * @param channels Number of channels
     * @return Tensor [C,T]
     */
    public static Tensor fromFloatArray(float[] samples, int channels) {
        Objects.requireNonNull(samples, "samples cannot be null");
        return AudioTensors.toTensor(samples, channels);
    }

    // ── Audio Processing Methods ─────────────────────────────────────────

    /**
     * Resample waveform to target sample rate.
     *
     * @param tensor Waveform tensor
     * @param sourceRate Source sample rate
     * @param targetRate Target sample rate
     * @return Resampled waveform tensor
     */
    public static Tensor resample(Tensor tensor, int sourceRate, int targetRate) {
        if (sourceRate == targetRate) {
            return tensor;
        }

        AudioData audio = toAudioData(tensor, sourceRate);
        AudioData resampled = MediaBridge.resample(audio, targetRate);
        return toTensor(resampled);
    }

    /**
     * Convert multi-channel audio to mono.
     *
     * @param tensor Multi-channel waveform tensor [C,T]
     * @return Mono waveform tensor [T]
     */
    public static Tensor toMono(Tensor tensor) {
        Tensor cpu = tensor.contiguous().cpu().to(ScalarType.Float);
        long[] shape = cpu.sizes().stream().mapToLong(Long::longValue).toArray();

        if (shape.length == 1) {
            return cpu;
        }

        int channels = (int) shape[0];
        int time = (int) shape[1];
        float[] data = AudioTensors.toFloatArray(cpu);

        float[] mono = new float[time];
        for (int t = 0; t < time; t++) {
            float sum = 0f;
            for (int c = 0; c < channels; c++) {
                sum += data[c * time + t];
            }
            mono[t] = sum / channels;
        }
        return org.bytedeco.pytorch.global.torch.tensor(mono);
    }

    /**
     * Mix multiple audio tensors to mono.
     *
     * @param tensors List of waveform tensors
     * @return Mixed mono waveform
     */
    public static Tensor mixToMono(List<Tensor> tensors) {
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("Empty tensor list");
        }
        if (tensors.size() == 1) {
            return toMono(tensors.get(0));
        }

        int maxTime = 0;
        for (Tensor t : tensors) {
            long[] shape = t.sizes().stream().mapToLong(Long::longValue).toArray();
            int time = shape.length == 1 ? (int) shape[0] : (int) shape[1];
            maxTime = Math.max(maxTime, time);
        }

        float[] mixed = new float[maxTime];
        for (Tensor tensor : tensors) {
            float[] data = toFloatArray(toMono(tensor));
            for (int i = 0; i < data.length && i < mixed.length; i++) {
                mixed[i] += data[i];
            }
        }

        // Normalize
        float maxAbs = 0f;
        for (float v : mixed) {
            maxAbs = Math.max(maxAbs, Math.abs(v));
        }
        if (maxAbs > 1.0f) {
            for (int i = 0; i < mixed.length; i++) {
                mixed[i] /= maxAbs;
            }
        }

        return org.bytedeco.pytorch.global.torch.tensor(mixed);
    }

    /**
     * Apply fade in/out to waveform.
     *
     * @param tensor Waveform tensor
     * @param fadeInSamples Fade in duration in samples
     * @param fadeOutSamples Fade out duration in samples
     * @return Tensor with fade applied
     */
    public static Tensor applyFade(Tensor tensor, int fadeInSamples, int fadeOutSamples) {
        Tensor cpu = tensor.contiguous().cpu().to(ScalarType.Float);
        long[] shape = cpu.sizes().stream().mapToLong(Long::longValue).toArray();
        int time = shape.length == 1 ? (int) shape[0] : (int) shape[shape.length - 1];
        float[] data = AudioTensors.toFloatArray(cpu);

        // Fade in
        for (int i = 0; i < fadeInSamples && i < data.length; i++) {
            float gain = (float) i / fadeInSamples;
            data[i] *= gain;
        }

        // Fade out
        int fadeOutStart = Math.max(0, data.length - fadeOutSamples);
        for (int i = fadeOutStart; i < data.length; i++) {
            float gain = (float) (data.length - i) / fadeOutSamples;
            data[i] *= gain;
        }

        if (shape.length == 1) {
            return org.bytedeco.pytorch.global.torch.tensor(data);
        } else {
            int channels = (int) shape[0];
            return AudioTensors.toTensor(data, channels);
        }
    }

    /**
     * Normalize audio to target peak level.
     *
     * @param tensor Waveform tensor
     * @param targetPeak Target peak level (0.0 to 1.0)
     * @return Normalized tensor
     */
    public static Tensor normalize(Tensor tensor, float targetPeak) {
        Tensor cpu = tensor.contiguous().cpu().to(ScalarType.Float);
        float[] data = AudioTensors.toFloatArray(cpu);

        float maxAbs = 0f;
        for (float v : data) {
            maxAbs = Math.max(maxAbs, Math.abs(v));
        }

        if (maxAbs > 1e-6) {
            float gain = targetPeak / maxAbs;
            for (int i = 0; i < data.length; i++) {
                data[i] *= gain;
            }
        }

        long[] shape = cpu.sizes().stream().mapToLong(Long::longValue).toArray();
        if (shape.length == 1) {
            return org.bytedeco.pytorch.global.torch.tensor(data);
        } else {
            int channels = (int) shape[0];
            return AudioTensors.toTensor(data, channels);
        }
    }

    // ── Feature Extraction ────────────────────────────────────────────────

    /**
     * Compute simple spectrogram from waveform.
     *
     * @param tensor Waveform tensor
     * @param nfft FFT size
     * @param hopLength Hop length
     * @return Spectrogram tensor [F,T]
     */
    public static Tensor spectrogram(Tensor tensor, int nfft, int hopLength) {
        float[] waveform = toFloatArray(tensor);
        int numSamples = waveform.length;

        // Simple STFT implementation
        int numFrames = (numSamples - nfft) / hopLength + 1;
        int numFreqBins = nfft / 2 + 1;

        float[][] spectrogram = new float[numFrames][numFreqBins];

        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopLength;
            float[] frameData = new float[nfft];
            System.arraycopy(waveform, start, frameData, 0, nfft);

            // Apply Hann window
            for (int i = 0; i < nfft; i++) {
                float window = (float) (0.5 * (1 - Math.cos(2 * Math.PI * i / (nfft - 1))));
                frameData[i] *= window;
            }

            // Simple DFT (for demonstration - in production use FFT library)
            for (int k = 0; k < numFreqBins; k++) {
                float real = 0, imag = 0;
                for (int n = 0; n < nfft; n++) {
                    float angle = (float) (-2 * Math.PI * k * n / nfft);
                    real += frameData[n] * Math.cos(angle);
                    imag += frameData[n] * Math.sin(angle);
                }
                spectrogram[frame][k] = (float) Math.sqrt(real * real + imag * imag);
            }
        }

        // Convert to tensor
        float[] flat = new float[numFrames * numFreqBins];
        for (int i = 0; i < numFrames; i++) {
            System.arraycopy(spectrogram[i], 0, flat, i * numFreqBins, numFreqBins);
        }
        return org.bytedeco.pytorch.global.torch.tensor(flat).reshape(numFrames, numFreqBins);
    }

    /**
     * Compute RMS energy of waveform.
     *
     * @param tensor Waveform tensor
     * @param frameLength Frame length for analysis
     * @param hopLength Hop length
     * @return RMS energy tensor [T]
     */
    public static Tensor rmsEnergy(Tensor tensor, int frameLength, int hopLength) {
        float[] waveform = toFloatArray(tensor);
        int numFrames = (waveform.length - frameLength) / hopLength + 1;
        float[] rms = new float[numFrames];

        for (int frame = 0; frame < numFrames; frame++) {
            int start = frame * hopLength;
            float sum = 0;
            for (int i = 0; i < frameLength; i++) {
                float sample = waveform[start + i];
                sum += sample * sample;
            }
            rms[frame] = (float) Math.sqrt(sum / frameLength);
        }

        return org.bytedeco.pytorch.global.torch.tensor(rms);
    }

    // ── Format Validation ─────────────────────────────────────────────────

    /**
     * Validate tensor shape for audio conversion.
     */
    public static boolean isValidAudioTensor(Tensor tensor) {
        if (tensor == null) return false;
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        return shape.length == 1 || shape.length == 2 || shape.length == 3;
    }

    /**
     * Get audio duration from tensor and sample rate.
     */
    public static double getDuration(Tensor tensor, int sampleRate) {
        long[] shape = MultimodalTensorConverter.getShape(tensor);
        int timeSamples = shape.length == 1 ? (int) shape[0] :
                         (shape[0] <= 16 ? (int) shape[1] : (int) shape[0]);
        return (double) timeSamples / sampleRate;
    }

    /**
     * Get number of samples from tensor.
     */
    public static int getSampleCount(Tensor tensor) {
        float[] data = toFloatArray(tensor);
        return data.length;
    }

    /**
     * Get channel count from tensor.
     */
    public static int getChannelCount(Tensor tensor) {
        return AudioTensors.inferChannels(tensor);
    }

    // ── Utility Methods ───────────────────────────────────────────────────

    /**
     * Generate test audio signal.
     *
     * @param durationSec Duration in seconds
     * @param sampleRate Sample rate
     * @param frequency Signal frequency in Hz
     * @return Waveform tensor [T]
     */
    public static Tensor generateSineWave(double durationSec, int sampleRate, float frequency) {
        int numSamples = (int) (durationSec * sampleRate);
        float[] samples = new float[numSamples];
        for (int i = 0; i < numSamples; i++) {
            samples[i] = (float) Math.sin(2 * Math.PI * frequency * i / sampleRate);
        }
        return org.bytedeco.pytorch.global.torch.tensor(samples);
    }

    /**
     * Generate white noise.
     *
     * @param durationSec Duration in seconds
     * @param sampleRate Sample rate
     * @param amplitude Peak amplitude
     * @return Waveform tensor [T]
     */
    public static Tensor generateNoise(double durationSec, int sampleRate, float amplitude) {
        int numSamples = (int) (durationSec * sampleRate);
        float[] samples = new float[numSamples];
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < numSamples; i++) {
            samples[i] = (float) (rand.nextGaussian() * amplitude);
        }
        return org.bytedeco.pytorch.global.torch.tensor(samples);
    }
}
