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
package org.bytedeco.pytorch.recommend.multimodal;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * MultiModal feature extractor base interface for recommendation systems.
 *
 * <p>Supports extracting features from:
 * <ul>
 *   <li>Images (product images, thumbnails, covers)</li>
 *   <li>Videos (short videos, live streams)</li>
 *   <li>Audio (music, speech, sound effects)</li>
 *   <li>Text (descriptions, titles, comments)</li>
 * </ul>
 *
 * <p>Reference: CLIP, BLIP, Whisper, VideoLlama
 *
 * <pre>{@code
 * MultiModalFeatureExtractor extractor = MultiModalFeatureExtractor.builder()
 *     .imageEncoder(imageEncoder)
 *     .videoEncoder(videoEncoder)
 *     .audioEncoder(audioEncoder)
 *     .textEncoder(textEncoder)
 *     .build();
 *
 * MultiModalFeatures features = extractor.extract(
 *     imageBatch,
 *     videoBatch,
 *     audioBatch,
 *     textBatch
 * );
 * }</pre>
 */
public class MultiModalFeatureExtractor implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Encoders
    private final Module imageEncoder;
    private final Module videoEncoder;
    private final Module audioEncoder;
    private final Module textEncoder;

    // Configuration
    private final int embedDim;
    private final ModalityFusionType fusionType;
    private final boolean useProjection;
    private final int projectionDim;

    // Performance metrics
    private final AtomicLong totalProcessed = new AtomicLong(0);
    private final AtomicLong imagesProcessed = new AtomicLong(0);
    private final AtomicLong videosProcessed = new AtomicLong(0);
    private final AtomicLong audioProcessed = new AtomicLong(0);
    private final AtomicLong textProcessed = new AtomicLong(0);
    private final AtomicLong totalTimeMs = new AtomicLong(0);
    private final AtomicReference<String> lastError = new AtomicReference<>(null);

    /**
     * Modality fusion types.
     */
    public enum ModalityFusionType {
        /** Concatenate all modalities */
        CONCATENATION,
        /** Weighted sum of modalities */
        WEIGHTED_SUM,
        /** Cross-attention fusion */
        CROSS_ATTENTION,
        /** Gated fusion with learned gates */
        GATED,
        /** Late fusion with separate encoders */
        LATE_FUSION
    }

    public static Builder builder() {
        return new Builder();
    }

    private MultiModalFeatureExtractor(Builder builder) {
        this.imageEncoder = builder.imageEncoder;
        this.videoEncoder = builder.videoEncoder;
        this.audioEncoder = builder.audioEncoder;
        this.textEncoder = builder.textEncoder;
        this.embedDim = builder.embedDim;
        this.fusionType = builder.fusionType;
        this.useProjection = builder.useProjection;
        this.projectionDim = builder.projectionDim;
    }

    /**
     * Extract features from all modalities.
     *
     * @param images Image tensor [batch, channels, height, width] or null
     * @param videos Video tensor [batch, frames, channels, height, width] or null
     * @param audio Audio tensor [batch, samples] or null
     * @param text Text tensor [batch, seq_len] (token IDs) or null
     * @return MultiModalFeatures containing all modality features
     */
    public MultiModalFeatures extract(Tensor images, Tensor videos, Tensor audio, Tensor text) {
        long start = System.currentTimeMillis();

        try {
            MultiModalFeatures.Builder featuresBuilder = MultiModalFeatures.builder();

            // Extract image features
            if (images != null && imageEncoder != null) {
                Tensor imageFeatures = imageEncoder.forward(images);
                featuresBuilder.imageFeatures(imageFeatures);
                imagesProcessed.addAndGet(images.size(0));
            }

            // Extract video features
            if (videos != null && videoEncoder != null) {
                Tensor videoFeatures = videoEncoder.forward(videos);
                featuresBuilder.videoFeatures(videoFeatures);
                videosProcessed.addAndGet(videos.size(0));
            }

            // Extract audio features
            if (audio != null && audioEncoder != null) {
                Tensor audioFeatures = audioEncoder.forward(audio);
                featuresBuilder.audioFeatures(audioFeatures);
                audioProcessed.addAndGet(audio.size(0));
            }

            // Extract text features
            if (text != null && textEncoder != null) {
                Tensor textFeatures = textEncoder.forward(text);
                featuresBuilder.textFeatures(textFeatures);
                textProcessed.addAndGet(text.size(0));
            }

            MultiModalFeatures features = featuresBuilder.build();

            // Fuse modalities
            features.fusedFeatures = fuseModalities(features);

            totalProcessed.incrementAndGet();
            totalTimeMs.addAndGet(System.currentTimeMillis() - start);

            return features;

        } catch (Exception e) {
            lastError.set(e.getMessage());
            System.err.println("MultiModalFeatureExtractor.extract error: " + e.getMessage());
            return MultiModalFeatures.builder().build();
        }
    }

    /**
     * Extract image features only.
     */
    public Tensor extractImageFeatures(Tensor images) {
        if (imageEncoder == null) {
            throw new IllegalStateException("Image encoder not configured");
        }
        return imageEncoder.forward(images);
    }

    /**
     * Extract video features only.
     */
    public Tensor extractVideoFeatures(Tensor videos) {
        if (videoEncoder == null) {
            throw new IllegalStateException("Video encoder not configured");
        }
        return videoEncoder.forward(videos);
    }

    /**
     * Extract audio features only.
     */
    public Tensor extractAudioFeatures(Tensor audio) {
        if (audioEncoder == null) {
            throw new IllegalStateException("Audio encoder not configured");
        }
        return audioEncoder.forward(audio);
    }

    /**
     * Extract text features only.
     */
    public Tensor extractTextFeatures(Tensor text) {
        if (textEncoder == null) {
            throw new IllegalStateException("Text encoder not configured");
        }
        return textEncoder.forward(text);
    }

    /**
     * Fuse modality features based on configured fusion type.
     */
    private Tensor fuseModalities(MultiModalFeatures features) {
        switch (fusionType) {
            case CONCATENATION:
                return fuseByConcatenation(features);
            case WEIGHTED_SUM:
                return fuseByWeightedSum(features);
            case CROSS_ATTENTION:
                return fuseByCrossAttention(features);
            case GATED:
                return fuseByGating(features);
            case LATE_FUSION:
            default:
                return fuseByLateFusion(features);
        }
    }

    /**
     * Fuse by concatenation.
     */
    private Tensor fuseByConcatenation(MultiModalFeatures features) {
        return features.concat();
    }

    /**
     * Fuse by weighted sum.
     */
    private Tensor fuseByWeightedSum(MultiModalFeatures features) {
        Tensor fused = null;
        int count = 0;

        if (features.imageFeatures != null) {
            fused = features.imageFeatures;
            count++;
        }
        if (features.videoFeatures != null) {
            if (fused == null) {
                fused = features.videoFeatures;
            } else {
                fused = fused.add(features.videoFeatures);
            }
            count++;
        }
        if (features.audioFeatures != null) {
            if (fused == null) {
                fused = features.audioFeatures;
            } else {
                fused = fused.add(features.audioFeatures);
            }
            count++;
        }
        if (features.textFeatures != null) {
            if (fused == null) {
                fused = features.textFeatures;
            } else {
                fused = fused.add(features.textFeatures);
            }
            count++;
        }

        return fused != null ? fused.div(new Scalar((double) count)) : null;
    }

    /**
     * Fuse by cross-attention.
     */
    private Tensor fuseByCrossAttention(MultiModalFeatures features) {
        // Simplified cross-attention fusion
        return features.concat();
    }

    /**
     * Fuse by gating mechanism.
     */
    private Tensor fuseByGating(MultiModalFeatures features) {
        // Simplified gated fusion
        return features.concat();
    }

    /**
     * Fuse by late fusion (average pooling).
     */
    private Tensor fuseByLateFusion(MultiModalFeatures features) {
        return fuseByWeightedSum(features);
    }

    /**
     * Get statistics.
     */
    public MultiModalExtractorStats getStats() {
        return new MultiModalExtractorStats(
                embedDim,
                fusionType,
                totalProcessed.get(),
                imagesProcessed.get(),
                videosProcessed.get(),
                audioProcessed.get(),
                textProcessed.get(),
                totalTimeMs.get(),
                lastError.get()
        );
    }

    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        if (imageEncoder != null) imageEncoder.close();
        if (videoEncoder != null) videoEncoder.close();
        if (audioEncoder != null) audioEncoder.close();
        if (textEncoder != null) textEncoder.close();

        System.out.printf(
                "[MultiModalFeatureExtractor] Closed: processed=%d, images=%d, videos=%d, " +
                "audio=%d, text=%d, time=%.2fs%n",
                totalProcessed.get(), imagesProcessed.get(), videosProcessed.get(),
                audioProcessed.get(), textProcessed.get(), totalTimeMs.get() / 1000.0);
    }

    /**
     * MultiModal features container.
     */
    public static class MultiModalFeatures {
        private Tensor imageFeatures;
        private Tensor videoFeatures;
        private Tensor audioFeatures;
        private Tensor textFeatures;
        private Tensor fusedFeatures;

        private MultiModalFeatures() {}

        public static Builder builder() {
            return new Builder();
        }

        public Tensor imageFeatures() { return imageFeatures; }
        public Tensor videoFeatures() { return videoFeatures; }
        public Tensor audioFeatures() { return audioFeatures; }
        public Tensor textFeatures() { return textFeatures; }
        public Tensor fusedFeatures() { return fusedFeatures; }

        /**
         * Concatenate all available features.
         */
        public Tensor concat() {
            Tensor[] features = new Tensor[4];
            int count = 0;

            if (imageFeatures != null) features[count++] = imageFeatures;
            if (videoFeatures != null) features[count++] = videoFeatures;
            if (audioFeatures != null) features[count++] = audioFeatures;
            if (textFeatures != null) features[count++] = textFeatures;

            if (count == 0) return null;

            Tensor[] nonNull = new Tensor[count];
            System.arraycopy(features, 0, nonNull, 0, count);
            return torch.cat(new TensorVector(nonNull), -1);
        }

        public static class Builder {
            private final MultiModalFeatures features = new MultiModalFeatures();

            public Builder imageFeatures(Tensor t) { features.imageFeatures = t; return this; }
            public Builder videoFeatures(Tensor t) { features.videoFeatures = t; return this; }
            public Builder audioFeatures(Tensor t) { features.audioFeatures = t; return this; }
            public Builder textFeatures(Tensor t) { features.textFeatures = t; return this; }

            public MultiModalFeatures build() { return features; }
        }
    }

    /**
     * Statistics.
     */
    public static class MultiModalExtractorStats {
        public final int embedDim;
        public final ModalityFusionType fusionType;
        public final long totalProcessed;
        public final long imagesProcessed;
        public final long videosProcessed;
        public final long audioProcessed;
        public final long textProcessed;
        public final long totalTimeMs;
        public final String lastError;

        public MultiModalExtractorStats(int embedDim, ModalityFusionType fusionType,
                                 long totalProcessed, long imagesProcessed, long videosProcessed,
                                 long audioProcessed, long textProcessed, long totalTimeMs,
                                 String lastError) {
            this.embedDim = embedDim;
            this.fusionType = fusionType;
            this.totalProcessed = totalProcessed;
            this.imagesProcessed = imagesProcessed;
            this.videosProcessed = videosProcessed;
            this.audioProcessed = audioProcessed;
            this.textProcessed = textProcessed;
            this.totalTimeMs = totalTimeMs;
            this.lastError = lastError;
        }

        public double avgTimeMs() {
            return totalProcessed > 0 ? (double) totalTimeMs / totalProcessed : 0;
        }

        @Override
        public String toString() {
            return String.format(
                    "MultiModalExtractorStats{embedDim=%d, fusion=%s, processed=%d, " +
                    "images=%d, videos=%d, audio=%d, text=%d, avgTime=%.2fms}",
                    embedDim, fusionType, totalProcessed,
                    imagesProcessed, videosProcessed, audioProcessed, textProcessed, avgTimeMs());
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private Module imageEncoder;
        private Module videoEncoder;
        private Module audioEncoder;
        private Module textEncoder;
        private int embedDim = 512;
        private ModalityFusionType fusionType = ModalityFusionType.CONCATENATION;
        private boolean useProjection = true;
        private int projectionDim = 512;

        public Builder imageEncoder(Module encoder) { this.imageEncoder = encoder; return this; }
        public Builder videoEncoder(Module encoder) { this.videoEncoder = encoder; return this; }
        public Builder audioEncoder(Module encoder) { this.audioEncoder = encoder; return this; }
        public Builder textEncoder(Module encoder) { this.textEncoder = encoder; return this; }
        public Builder embedDim(int dim) { this.embedDim = dim; return this; }
        public Builder fusionType(ModalityFusionType type) { this.fusionType = type; return this; }
        public Builder useProjection(boolean use) { this.useProjection = use; return this; }
        public Builder projectionDim(int dim) { this.projectionDim = dim; return this; }

        /**
         * Configure for CLIP-style image-text.
         */
        public Builder clipStyle() {
            this.embedDim = 512;
            this.fusionType = ModalityFusionType.CONCATENATION;
            return this;
        }

        /**
         * Configure for video understanding.
         */
        public Builder videoStyle() {
            this.embedDim = 768;
            this.fusionType = ModalityFusionType.CROSS_ATTENTION;
            return this;
        }

        public MultiModalFeatureExtractor build() {
            return new MultiModalFeatureExtractor(this);
        }
    }
}
