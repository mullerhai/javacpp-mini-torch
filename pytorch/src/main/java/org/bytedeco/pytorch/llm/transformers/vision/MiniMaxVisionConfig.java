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
package org.bytedeco.pytorch.llm.transformers.vision;

import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Configuration for MiniMax VL vision-language models.
 *
 * <p>Reference: MiniMax-VL
 */
public class MiniMaxVisionConfig {

    public static final String VERSION = "2.0";

    // Vision configuration
    private final int hiddenSize;
    private final int intermediateSize;
    private final int numHiddenLayers;
    private final int numAttentionHeads;
    private final int numKeyValueHeads;
    private final int visionEmbedDim;
    private final int patchSize;
    private final int imageSize;
    private final int maxImageSize;
    private final int maxSeqLength;
    private final double layerNormEps;
    private final String hiddenAct;
    private final boolean useGatedActivation;

    // Image processing
    private final float[] imageMean;
    private final float[] imageStd;
    private final int spatialMergeSize;
    private final boolean dynamicResolution;

    // Video processing
    private final int maxFrames;
    private final float frameRate;
    private final int temporalMergeSize;

    // Special tokens
    private final int imageTokenId;
    private final int videoTokenId;
    private final int systemTokenId;

    private MiniMaxVisionConfig(Builder builder) {
        this.hiddenSize = builder.hiddenSize;
        this.intermediateSize = builder.intermediateSize;
        this.numHiddenLayers = builder.numHiddenLayers;
        this.numAttentionHeads = builder.numAttentionHeads;
        this.numKeyValueHeads = builder.numKeyValueHeads;
        this.visionEmbedDim = builder.visionEmbedDim;
        this.patchSize = builder.patchSize;
        this.imageSize = builder.imageSize;
        this.maxImageSize = builder.maxImageSize;
        this.maxSeqLength = builder.maxSeqLength;
        this.layerNormEps = builder.layerNormEps;
        this.hiddenAct = builder.hiddenAct;
        this.useGatedActivation = builder.useGatedActivation;
        this.imageMean = builder.imageMean != null ? builder.imageMean.clone() : null;
        this.imageStd = builder.imageStd != null ? builder.imageStd.clone() : null;
        this.spatialMergeSize = builder.spatialMergeSize;
        this.dynamicResolution = builder.dynamicResolution;
        this.maxFrames = builder.maxFrames;
        this.frameRate = builder.frameRate;
        this.temporalMergeSize = builder.temporalMergeSize;
        this.imageTokenId = builder.imageTokenId;
        this.videoTokenId = builder.videoTokenId;
        this.systemTokenId = builder.systemTokenId;
    }

    /**
     * Create config from JSON file.
     */
    public static MiniMaxVisionConfig fromFile(Path path) throws IOException {
        String json = Files.readString(path, StandardCharsets.UTF_8);
        return fromJson(json);
    }

    /**
     * Create config from JSON string.
     */
    @SuppressWarnings("unchecked")
    public static MiniMaxVisionConfig fromJson(String json) throws IOException {
        Map<String, Object> m = Json.decodeObject(json);
        Builder builder = builder();

        // Vision config may be nested
        Map<String, Object> visionConfig = null;
        Object vc = m.get("vision_config");
        if (vc instanceof Map) {
            visionConfig = (Map<String, Object>) vc;
        }

        Map<String, Object> config = visionConfig != null ? visionConfig : m;

        builder.hiddenSize(getInt(config, "hidden_size", 2048));
        builder.intermediateSize(getInt(config, "intermediate_size", 8192));
        builder.numHiddenLayers(getInt(config, "num_hidden_layers", 24));
        builder.numAttentionHeads(getInt(config, "num_attention_heads", 16));
        builder.numKeyValueHeads(getInt(config, "num_key_value_heads", 16));
        builder.visionEmbedDim(getInt(config, "vision_embed_dim", 2048));
        builder.patchSize(getInt(config, "patch_size", 14));
        builder.imageSize(getInt(config, "image_size", 384));
        builder.maxImageSize(getInt(config, "max_image_size", 384));
        builder.maxSeqLength(getInt(config, "max_seq_length", 1048576));
        builder.layerNormEps(getDouble(config, "layer_norm_eps", 1e-5));
        builder.hiddenAct(getString(config, "hidden_act", "silu"));
        builder.useGatedActivation(getBool(config, "use_gated_activation", true));

        // Image processing
        builder.spatialMergeSize(getInt(config, "spatial_merge_size", 2));
        builder.dynamicResolution(getBool(config, "dynamic_resolution", true));

        // Video
        builder.maxFrames(getInt(config, "max_frames", 64));
        builder.frameRate(getFloat(config, "frame_rate", 1.0f));
        builder.temporalMergeSize(getInt(config, "temporal_merge_size", 2));

        // Special tokens
        builder.imageTokenId(getInt(config, "image_token_id", 151652));
        builder.videoTokenId(getInt(config, "video_token_id", 151653));
        builder.systemTokenId(getInt(config, "system_token_id", 151654));

        return builder.build();
    }

    // Getters
    public int hiddenSize() { return hiddenSize; }
    public int intermediateSize() { return intermediateSize; }
    public int numHiddenLayers() { return numHiddenLayers; }
    public int numAttentionHeads() { return numAttentionHeads; }
    public int numKeyValueHeads() { return numKeyValueHeads; }
    public int visionEmbedDim() { return visionEmbedDim; }
    public int patchSize() { return patchSize; }
    public int imageSize() { return imageSize; }
    public int maxImageSize() { return maxImageSize; }
    public int maxSeqLength() { return maxSeqLength; }
    public double layerNormEps() { return layerNormEps; }
    public String hiddenAct() { return hiddenAct; }
    public boolean useGatedActivation() { return useGatedActivation; }
    public float[] imageMean() { return imageMean; }
    public float[] imageStd() { return imageStd; }
    public int spatialMergeSize() { return spatialMergeSize; }
    public boolean dynamicResolution() { return dynamicResolution; }
    public int maxFrames() { return maxFrames; }
    public float frameRate() { return frameRate; }
    public int temporalMergeSize() { return temporalMergeSize; }
    public int imageTokenId() { return imageTokenId; }
    public int videoTokenId() { return videoTokenId; }
    public int systemTokenId() { return systemTokenId; }

    /**
     * Calculate number of image tokens for given dimensions.
     */
    public int getNumImageTokens(int height, int width) {
        int h = (int) Math.ceil(height / (float) spatialMergeSize);
        int w = (int) Math.ceil(width / (float) spatialMergeSize);
        return h * w;
    }

    /**
     * Calculate video tokens for given frames.
     */
    public int getNumVideoTokens(int frames, int height, int width) {
        int temporal = (int) Math.ceil(frames / (float) temporalMergeSize);
        int h = (int) Math.ceil(height / (float) spatialMergeSize);
        int w = (int) Math.ceil(width / (float) spatialMergeSize);
        return temporal * h * w;
    }

    @Override
    public String toString() {
        return String.format(
                "MiniMaxVisionConfig{hiddenSize=%d, layers=%d, embedDim=%d, " +
                "patchSize=%d, maxSeqLen=%d}",
                hiddenSize, numHiddenLayers, visionEmbedDim, patchSize, maxSeqLength);
    }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private int hiddenSize = 2048;
        private int intermediateSize = 8192;
        private int numHiddenLayers = 24;
        private int numAttentionHeads = 16;
        private int numKeyValueHeads = 16;
        private int visionEmbedDim = 2048;
        private int patchSize = 14;
        private int imageSize = 384;
        private int maxImageSize = 384;
        private int maxSeqLength = 1048576;
        private double layerNormEps = 1e-5;
        private String hiddenAct = "silu";
        private boolean useGatedActivation = true;
        private float[] imageMean = new float[]{0.485f, 0.456f, 0.406f};
        private float[] imageStd = new float[]{0.229f, 0.224f, 0.225f};
        private int spatialMergeSize = 2;
        private boolean dynamicResolution = true;
        private int maxFrames = 64;
        private float frameRate = 1.0f;
        private int temporalMergeSize = 2;
        private int imageTokenId = 151652;
        private int videoTokenId = 151653;
        private int systemTokenId = 151654;

        public Builder hiddenSize(int v) { this.hiddenSize = v; return this; }
        public Builder intermediateSize(int v) { this.intermediateSize = v; return this; }
        public Builder numHiddenLayers(int v) { this.numHiddenLayers = v; return this; }
        public Builder numAttentionHeads(int v) { this.numAttentionHeads = v; return this; }
        public Builder numKeyValueHeads(int v) { this.numKeyValueHeads = v; return this; }
        public Builder visionEmbedDim(int v) { this.visionEmbedDim = v; return this; }
        public Builder patchSize(int v) { this.patchSize = v; return this; }
        public Builder imageSize(int v) { this.imageSize = v; return this; }
        public Builder maxImageSize(int v) { this.maxImageSize = v; return this; }
        public Builder maxSeqLength(int v) { this.maxSeqLength = v; return this; }
        public Builder layerNormEps(double v) { this.layerNormEps = v; return this; }
        public Builder hiddenAct(String v) { this.hiddenAct = v; return this; }
        public Builder useGatedActivation(boolean v) { this.useGatedActivation = v; return this; }
        public Builder imageMean(float[] v) { this.imageMean = v; return this; }
        public Builder imageStd(float[] v) { this.imageStd = v; return this; }
        public Builder spatialMergeSize(int v) { this.spatialMergeSize = v; return this; }
        public Builder dynamicResolution(boolean v) { this.dynamicResolution = v; return this; }
        public Builder maxFrames(int v) { this.maxFrames = v; return this; }
        public Builder frameRate(float v) { this.frameRate = v; return this; }
        public Builder temporalMergeSize(int v) { this.temporalMergeSize = v; return this; }
        public Builder imageTokenId(int v) { this.imageTokenId = v; return this; }
        public Builder videoTokenId(int v) { this.videoTokenId = v; return this; }
        public Builder systemTokenId(int v) { this.systemTokenId = v; return this; }

        public MiniMaxVisionConfig build() { return new MiniMaxVisionConfig(this); }
    }

    // Helper methods
    private static int getInt(Map<String, Object> m, String key, int def) {
        Object v = m.get(key);
        if (v instanceof Number) return ((Number) v).intValue();
        return def;
    }

    private static double getDouble(Map<String, Object> m, String key, double def) {
        Object v = m.get(key);
        if (v instanceof Number) return ((Number) v).doubleValue();
        return def;
    }

    private static String getString(Map<String, Object> m, String key, String def) {
        Object v = m.get(key);
        return v != null ? v.toString() : def;
    }

    private static boolean getBool(Map<String, Object> m, String key, boolean def) {
        Object v = m.get(key);
        if (v instanceof Boolean) return (Boolean) v;
        if (v instanceof Number) return ((Number) v).doubleValue() != 0;
        return def;
    }

    private static float getFloat(Map<String, Object> m, String key, float def) {
        Object v = m.get(key);
        if (v instanceof Number) return ((Number) v).floatValue();
        return def;
    }
}
