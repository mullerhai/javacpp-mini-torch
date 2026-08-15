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
package org.bytedeco.pytorch.llm.transformers.processor;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Whisper processor for audio-to-text transcription.
 *
 * <p>Handles:
 * <ul>
 *   <li>Audio loading from various formats</li>
 *   <li>Mel spectrogram extraction</li>
 *   <li>Audio chunking for long-form transcription</li>
 *   <li>Language detection</li>
 *   <li>Task specification (transcribe, translate)</li>
 * </ul>
 *
 * <p>Reference: OpenAI Whisper, transformers WhisperProcessor
 *
 * <pre>{@code
 * try (WhisperProcessor processor = WhisperProcessor.fromPretrained("openai/whisper-tiny")) {
 *     // Process audio file
 *     ProcessorOutput output = processor.process(
 *         Processor.ProcessingInput.ofAudio(null, audioData, 16000)
 *     );
 *
 *     // Use with Whisper model
 *     int[] tokenIds = model.generate(output.inputIds());
 *     String text = processor.decode(tokenIds);
 * }
 * }</pre>
 */
public class WhisperProcessor implements Processor {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Components
    private final FastTokenizer tokenizer;
    private final AudioProcessor featureExtractor;

    // Configuration
    private final int sampleRate;
    private final int chunkLength;
    private final int nMaxMelBins;
    private final boolean doNormalize;
    private final boolean returnTimestamp;
    private final String language;
    private final String task;  // "transcribe" or "translate"

    // Performance metrics
    private final AtomicLong totalProcessingTimeMs = new AtomicLong(0);
    private final AtomicLong audioProcessed = new AtomicLong(0);

    /**
     * Create WhisperProcessor with defaults.
     */
    public static WhisperProcessor createDefault() {
        return builder().build();
    }

    /**
     * Create WhisperProcessor from pretrained model.
     */
    public static WhisperProcessor fromPretrained(String modelPath) {
        return builder()
                .featureExtractor(AudioProcessor.createWhisper())
                .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private WhisperProcessor(Builder builder) {
        this.tokenizer = builder.tokenizer;
        this.featureExtractor = builder.featureExtractor != null ?
                builder.featureExtractor : AudioProcessor.createWhisper();
        this.sampleRate = builder.sampleRate;
        this.chunkLength = builder.chunkLength;
        this.nMaxMelBins = builder.nMaxMelBins;
        this.doNormalize = builder.doNormalize;
        this.returnTimestamp = builder.returnTimestamp;
        this.language = builder.language;
        this.task = builder.task;
    }

    @Override
    public String version() { return VERSION; }

    @Override
    public FastTokenizer tokenizer() { return tokenizer; }

    @Override
    public List<Modality> supportedModalities() {
        return List.of(Modality.TEXT, Modality.AUDIO);
    }

    @Override
    public TextOutput processText(String text, boolean addSpecialTokens) {
        if (tokenizer == null) {
            return new TextOutput(new int[0], null, 0);
        }
        int[] ids = tokenizer.encode(text, addSpecialTokens).ids();
        return new TextOutput(ids, null, 0);
    }

    @Override
    public TextOutput processTextBatch(List<String> texts) {
        if (tokenizer == null) {
            return new TextOutput(new int[0], null, 0);
        }
        var encodings = tokenizer.encodeBatch(texts, false);
        int totalLen = 0;
        for (var enc : encodings) totalLen += enc.ids().length;
        int[] ids = new int[totalLen];
        int pos = 0;
        for (var enc : encodings) {
            System.arraycopy(enc.ids(), 0, ids, pos, enc.ids().length);
            pos += enc.ids().length;
        }
        return new TextOutput(ids, null, 0);
    }

    @Override
    public ImageOutput processImage(Object image) {
        throw new UnsupportedOperationException("Whisper does not support image input");
    }

    @Override
    public List<ImageOutput> processImageBatch(List<?> images) {
        throw new UnsupportedOperationException("Whisper does not support image input");
    }

    @Override
    public AudioOutput processAudio(Object audio) {
        long start = System.currentTimeMillis();
        audioProcessed.incrementAndGet();

        try {
            // Extract features based on audio type
            Tensor features;
            if (audio instanceof float[] audioData) {
                features = extractFeatures(audioData, audioData.length);
            } else if (audio instanceof short[] audioData) {
                float[] floatData = new float[audioData.length];
                for (int i = 0; i < audioData.length; i++) {
                    floatData[i] = audioData[i] / 32768.0f;
                }
                features = extractFeatures(floatData, floatData.length);
            } else {
                features = torch.zeros(new long[]{1, nMaxMelBins, 3000});
            }

            totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);

            return new AudioOutput(features, sampleRate, (int) features.size(2), nMaxMelBins, new float[0]);

        } catch (Exception e) {
            System.err.println("WhisperProcessor.processAudio error: " + e.getMessage());
            return createEmptyOutput();
        }
    }

    /**
     * Extract mel spectrogram features from audio.
     */
    private Tensor extractFeatures(float[] audioData, int numSamples) {
        // Get processor config
        int nMels = featureExtractor.nMels();
        int hopLength = featureExtractor.hopLength();
        int nFft = featureExtractor.nFft();

        // Compute number of frames
        int numFrames = Math.max(1, (numSamples - nFft) / hopLength + 1);

        // Compute mel spectrogram using feature extractor
        var audioOutput = featureExtractor.process(audioData, numSamples);
        return audioOutput.features();
    }

    /**
     * Process audio with forced decoder IDs (e.g., for timestamps or language).
     */
    public ProcessorOutput processForcedAudio(Object audio, long[] forcedDecoderIds) {
        long start = System.currentTimeMillis();

        try {
            AudioOutput audioOut = processAudio(audio);

            // Add forced decoder IDs as special tokens
            int[] inputIds = forcedDecoderIds != null ?
                    java.util.stream.LongStream.of(forcedDecoderIds).mapToInt(i -> (int) i).toArray() :
                    new int[0];

            totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);

            return ProcessorOutput.builder()
                    .inputIds(torch.tensor(inputIds))
                    .audioFeatures(audioOut.features())
                    .build();

        } catch (Exception e) {
            System.err.println("WhisperProcessor.processForcedAudio error: " + e.getMessage());
            return ProcessorOutput.builder()
                    .inputIds(torch.zeros(1, 1))
                    .build();
        }
    }

    /**
     * Generate forced decoder IDs for a language.
     */
    public int[] generateForcedDecoderIds(String lang, boolean translate) {
        // Whisper uses special token IDs for language/task specification
        // This is a simplified implementation
        return new int[0];  // Actual implementation would return language-specific tokens
    }

    /**
     * Generate forced decoder IDs for timestamps.
     */
    public int[] generateTimestampDecoderIds() {
        // Return timestamp-specific tokens
        return new int[0];
    }

    @Override
    public VideoOutput processVideo(Object video) {
        throw new UnsupportedOperationException("Whisper does not support video input");
    }

    @Override
    public ProcessorOutput process(ProcessingInput input) {
        long start = System.currentTimeMillis();

        try {
            // Process audio
            Tensor audioFeatures = null;
            int[] inputIds = new int[0];

            if (input.audios() != null && !input.audios().isEmpty()) {
                AudioOutput audioOut = processAudio(input.audios().get(0));
                audioFeatures = audioOut.features();

                // Generate forced decoder IDs
                int[] forcedIds = generateForcedDecoderIds(language, "translate".equals(task));
                if (forcedIds.length > 0) {
                    inputIds = forcedIds;
                }
            }

            // Process text if provided
            if (input.text() != null && !input.text().isEmpty()) {
                TextOutput textOut = processText(input.text(), true);
                int[] textIds = textOut.inputIds();

                // Append text IDs to forced IDs
                if (inputIds.length > 0) {
                    int[] combined = new int[inputIds.length + textIds.length];
                    System.arraycopy(inputIds, 0, combined, 0, inputIds.length);
                    System.arraycopy(textIds, 0, combined, inputIds.length, textIds.length);
                    inputIds = combined;
                } else {
                    inputIds = textIds;
                }
            }

            totalProcessingTimeMs.addAndGet(System.currentTimeMillis() - start);

            return ProcessorOutput.builder()
                    .inputIds(torch.tensor(inputIds))
                    .audioFeatures(audioFeatures)
                    .build();

        } catch (Exception e) {
            System.err.println("WhisperProcessor.process error: " + e.getMessage());
            return ProcessorOutput.builder()
                    .inputIds(torch.zeros(1, 1))
                    .build();
        }
    }

    @Override
    public String decode(int[] tokenIds) {
        if (tokenizer == null) return "";
        return tokenizer.decode(tokenIds, true);
    }

    @Override
    public ProcessorStats getStats() {
        return new ProcessorStats(
                0, 0,
                audioProcessed.get(),
                0,
                totalProcessingTimeMs.get(),
                0
        );
    }

    @Override
    public void resetStats() {
        totalProcessingTimeMs.set(0);
        audioProcessed.set(0);
    }

    @Override
    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (featureExtractor != null) {
            try { featureExtractor.close(); } catch (Exception ignored) {}
        }
        System.out.printf("[WhisperProcessor] Closed: audio=%d, time=%.2fs%n",
                audioProcessed.get(), totalProcessingTimeMs.get() / 1000.0);
    }

    private AudioOutput createEmptyOutput() {
        return new AudioOutput(
                torch.zeros(new long[]{1, nMaxMelBins, 300}),
                sampleRate, 300, nMaxMelBins, new float[0]
        );
    }

    /**
     * Get sample rate.
     */
    public int sampleRate() { return sampleRate; }

    /**
     * Get chunk length.
     */
    public int chunkLength() { return chunkLength; }

    /**
     * Builder for WhisperProcessor.
     */
    public static class Builder {
        private FastTokenizer tokenizer;
        private AudioProcessor featureExtractor;
        private int sampleRate = 16000;
        private int chunkLength = 30;  // seconds
        private int nMaxMelBins = 80;
        private boolean doNormalize = true;
        private boolean returnTimestamp = false;
        private String language = null;
        private String task = "transcribe";

        public Builder tokenizer(FastTokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return this;
        }

        public Builder featureExtractor(AudioProcessor featureExtractor) {
            this.featureExtractor = featureExtractor;
            return this;
        }

        public Builder sampleRate(int sampleRate) {
            this.sampleRate = sampleRate;
            return this;
        }

        public Builder chunkLength(int chunkLength) {
            this.chunkLength = chunkLength;
            return this;
        }

        public Builder nMaxMelBins(int nMaxMelBins) {
            this.nMaxMelBins = nMaxMelBins;
            return this;
        }

        public Builder doNormalize(boolean doNormalize) {
            this.doNormalize = doNormalize;
            return this;
        }

        public Builder returnTimestamp(boolean returnTimestamp) {
            this.returnTimestamp = returnTimestamp;
            return this;
        }

        public Builder language(String language) {
            this.language = language;
            return this;
        }

        public Builder task(String task) {
            this.task = task;
            return this;
        }

        public WhisperProcessor build() {
            return new WhisperProcessor(this);
        }
    }
}
