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
 * Configuration for Qwen2-VL and Qwen3-VL vision-language models.
 *
 * <p>Reference: Qwen2-VL, Qwen2.5-VL, Qwen3-VL
 */
public class QwenVisionConfig {

    public static final String VERSION = "2.0";

    // Vision configuration
    private final int hiddenSize;
    private final int intermediateSize;
    private final int numHiddenLayers;
    private final int numAttentionHeads;
    private final int numKeyValueHeads;
    private final int imageGridSize;
    private final int spatialMergeSize;
    private final int spatialMergeUnit;
    private final int patchSize;
    private final int temporalPatchSize;
    private final int visionEmbedDim;
    private final double rmsNormEps;
    private final String hiddenAct;
    private final boolean attentionBias;
    private final boolean attentionDropout;
    private final boolean hiddenDropout;

    // Image processing
    private final int imageSize;
    private final int maxImageSize;
    private final int minImageSize;
    private final float[] imageMean;
    private final float[] imageStd;
    private final boolean useImagenipt;

    // Video processing
    private final int maxFrames;
    private final float frameRate;

    // Special tokens
    private final int imageBoundTokenId;
    private final int imageStartTokenId;
    private final int imageEndTokenId;
    private final int videoStartTokenId;
    private final int videoEndTokenId;

    private QwenVisionConfig(Builder builder) {
        this.hiddenSize = builder.hiddenSize;
        this.intermediateSize = builder.intermediateSize;
        this.numHiddenLayers = builder.numHiddenLayers;
        this.numAttentionHeads = builder.numAttentionHeads;
        this.numKeyValueHeads = builder.numKeyValueHeads;
        this.imageGridSize = builder.imageGridSize;
        this.spatialMergeSize = builder.spatialMergeSize;
        this.spatialMergeUnit = builder.spatialMergeUnit;
        this.patchSize = builder.patchSize;
        this.temporalPatchSize = builder.temporalPatchSize;
        this.visionEmbedDim = builder.visionEmbedDim;
        this.rmsNormEps = builder.rmsNormEps;
        this.hiddenAct = builder.hiddenAct;
        this.attentionBias = builder.attentionBias;
        this.attentionDropout = builder.attentionDropout;
        this.hiddenDropout = builder.hiddenDropout;
        this.imageSize = builder.imageSize;
        this.maxImageSize = builder.maxImageSize;
        this.minImageSize = builder.minImageSize;
        this.imageMean = builder.imageMean != null ? builder.imageMean.clone() : null;
        this.imageStd = builder.imageStd != null ? builder.imageStd.clone() : null;
        this.useImagenipt = builder.useImagenipt;
        this.maxFrames = builder.maxFrames;
        this.frameRate = builder.frameRate;
        this.imageBoundTokenId = builder.imageBoundTokenId;
        this.imageStartTokenId = builder.imageStartTokenId;
        this.imageEndTokenId = builder.imageEndTokenId;
        this.videoStartTokenId = builder.videoStartTokenId;
        this.videoEndTokenId = builder.videoEndTokenId;
    }

    /**
     * Create config from JSON file (preprocessor_config.json or config.json).
     */
    public static QwenVisionConfig fromFile(Path path) throws IOException {
        String json = Files.readString(path, StandardCharsets.UTF_8);
        return fromJson(json);
    }

    /**
     * Create config from JSON string.
     */
    @SuppressWarnings("unchecked")
    public static QwenVisionConfig fromJson(String json) throws IOException {
        Map<String, Object> m = Json.decodeObject(json);
        Builder builder = builder();

        // Vision config may be nested
        Map<String, Object> visionConfig = null;
        Object vc = m.get("vision_config");
        if (vc instanceof Map) {
            visionConfig = (Map<String, Object>) vc;
        } else if (m.containsKey("visual")) {
            vc = m.get("visual");
            if (vc instanceof Map) {
                visionConfig = (Map<String, Object>) vc;
            }
        }

        Map<String, Object> config = visionConfig != null ? visionConfig : m;

        // Parse vision parameters
        builder.hiddenSize(getInt(config, "hidden_size", 1280));
        builder.intermediateSize(getInt(config, "intermediate_size", 5120));
        builder.numHiddenLayers(getInt(config, "num_hidden_layers", 32));
        builder.numAttentionHeads(getInt(config, "num_attention_heads", 16));
        builder.numKeyValueHeads(getInt(config, "num_key_value_heads", 16));
        builder.imageGridSize(getInt(config, "image_grid_size", 2));
        builder.spatialMergeSize(getInt(config, "spatial_merge_size", 2));
        builder.spatialMergeUnit(getInt(config, "spatial_merge_unit", 2));
        builder.patchSize(getInt(config, "patch_size", 14));
        builder.temporalPatchSize(getInt(config, "temporal_patch_size", 2));
        builder.visionEmbedDim(getInt(config, "vision_embed_dim", 1280));
        builder.rmsNormEps(getDouble(config, "rms_norm_eps", 1e-6));
        builder.hiddenAct(getString(config, "hidden_act", "silu"));
        builder.attentionBias(getBool(config, "attention_bias", false));
        builder.attentionDropout(getBool(config, "attention_dropout", false));
        builder.hiddenDropout(getBool(config, "hidden_dropout", false));

        // Image processing
        builder.imageSize(getInt(config, "image_size", 224));
        builder.maxImageSize(getInt(config, "max_image_size", 1280));
        builder.minImageSize(getInt(config, "min_image_size", 28));

        // Special tokens
        builder.imageBoundTokenId(getInt(config, "image_bound_token_id", 151652));
        builder.imageStartTokenId(getInt(config, "image_start_token_id", 151644));
        builder.imageEndTokenId(getInt(config, "image_end_token_id", 151645));
        builder.videoStartTokenId(getInt(config, "video_start_token_id", 151646));
        builder.videoEndTokenId(getInt(config, "video_end_token_id", 151647));

        // Video
        builder.maxFrames(getInt(config, "max_frames", 32));
        builder.frameRate(getFloat(config, "frame_rate", 2.0f));

        return builder.build();
    }

    // Getters
    public int hiddenSize() { return hiddenSize; }
    public int intermediateSize() { return intermediateSize; }
    public int numHiddenLayers() { return numHiddenLayers; }
    public int numAttentionHeads() { return numAttentionHeads; }
    public int numKeyValueHeads() { return numKeyValueHeads; }
    public int imageGridSize() { return imageGridSize; }
    public int spatialMergeSize() { return spatialMergeSize; }
    public int spatialMergeUnit() { return spatialMergeUnit; }
    public int patchSize() { return patchSize; }
    public int temporalPatchSize() { return temporalPatchSize; }
    public int visionEmbedDim() { return visionEmbedDim; }
    public double rmsNormEps() { return rmsNormEps; }
    public String hiddenAct() { return hiddenAct; }
    public boolean attentionBias() { return attentionBias; }
    public boolean attentionDropout() { return attentionDropout; }
    public boolean hiddenDropout() { return hiddenDropout; }
    public int imageSize() { return imageSize; }
    public int maxImageSize() { return maxImageSize; }
    public int minImageSize() { return minImageSize; }
    public float[] imageMean() { return imageMean; }
    public float[] imageStd() { return imageStd; }
    public boolean useImagenipt() { return useImagenipt; }
    public int maxFrames() { return maxFrames; }
    public float frameRate() { return frameRate; }
    public int imageBoundTokenId() { return imageBoundTokenId; }
    public int imageStartTokenId() { return imageStartTokenId; }
    public int imageEndTokenId() { return imageEndTokenId; }
    public int videoStartTokenId() { return videoStartTokenId; }
    public int videoEndTokenId() { return videoEndTokenId; }

    /**
     * Calculate number of image tokens for given dimensions.
     */
    public int getNumImageTokens(int height, int width) {
        int h = (int) Math.ceil(height / (float) spatialMergeSize);
        int w = (int) Math.ceil(width / (float) spatialMergeSize);
        return h * w;
    }

    /**
     * Calculate image grid THW for given dimensions.
     */
    public int[] getImageGridTHW(int height, int width) {
        int t = 1;
        int h = (int) Math.ceil(height / (float) spatialMergeSize);
        int w = (int) Math.ceil(width / (float) spatialMergeSize);
        return new int[]{t, h, w};
    }

    /**
     * Convert to Map for serialization.
     */
    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("hidden_size", hiddenSize);
        m.put("intermediate_size", intermediateSize);
        m.put("num_hidden_layers", numHiddenLayers);
        m.put("num_attention_heads", numAttentionHeads);
        m.put("num_key_value_heads", numKeyValueHeads);
        m.put("image_grid_size", imageGridSize);
        m.put("spatial_merge_size", spatialMergeSize);
        m.put("spatial_merge_unit", spatialMergeUnit);
        m.put("patch_size", patchSize);
        m.put("temporal_patch_size", temporalPatchSize);
        m.put("vision_embed_dim", visionEmbedDim);
        m.put("rms_norm_eps", rmsNormEps);
        m.put("hidden_act", hiddenAct);
        m.put("attention_bias", attentionBias);
        m.put("attention_dropout", attentionDropout);
        m.put("hidden_dropout", hiddenDropout);
        m.put("image_size", imageSize);
        m.put("max_image_size", maxImageSize);
        m.put("min_image_size", minImageSize);
        m.put("max_frames", maxFrames);
        m.put("frame_rate", frameRate);
        m.put("image_bound_token_id", imageBoundTokenId);
        m.put("image_start_token_id", imageStartTokenId);
        m.put("image_end_token_id", imageEndTokenId);
        m.put("video_start_token_id", videoStartTokenId);
        m.put("video_end_token_id", videoEndTokenId);
        return m;
    }

    @Override
    public String toString() {
        return String.format(
                "QwenVisionConfig{hideenSize=%d, layers=%d, heads=%d, embedDim=%d, " +
                "patchSize=%d, spatialMerge=%d}",
                hiddenSize, numHiddenLayers, numAttentionHeads, visionEmbedDim,
                patchSize, spatialMergeSize);
    }

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private int hiddenSize = 1280;
        private int intermediateSize = 5120;
        private int numHiddenLayers = 32;
        private int numAttentionHeads = 16;
        private int numKeyValueHeads = 16;
        private int imageGridSize = 2;
        private int spatialMergeSize = 2;
        private int spatialMergeUnit = 2;
        private int patchSize = 14;
        private int temporalPatchSize = 2;
        private int visionEmbedDim = 1280;
        private double rmsNormEps = 1e-6;
        private String hiddenAct = "silu";
        private boolean attentionBias = false;
        private boolean attentionDropout = false;
        private boolean hiddenDropout = false;
        private int imageSize = 224;
        private int maxImageSize = 1280;
        private int minImageSize = 28;
        private float[] imageMean = new float[]{0.48145466f, 0.4578275f, 0.40821073f};
        private float[] imageStd = new float[]{0.26862954f, 0.26130258f, 0.27577711f};
        private boolean useImagenipt = false;
        private int maxFrames = 32;
        private float frameRate = 2.0f;
        private int imageBoundTokenId = 151652;
        private int imageStartTokenId = 151644;
        private int imageEndTokenId = 151645;
        private int videoStartTokenId = 151646;
        private int videoEndTokenId = 151647;

        public Builder hiddenSize(int v) { this.hiddenSize = v; return this; }
        public Builder intermediateSize(int v) { this.intermediateSize = v; return this; }
        public Builder numHiddenLayers(int v) { this.numHiddenLayers = v; return this; }
        public Builder numAttentionHeads(int v) { this.numAttentionHeads = v; return this; }
        public Builder numKeyValueHeads(int v) { this.numKeyValueHeads = v; return this; }
        public Builder imageGridSize(int v) { this.imageGridSize = v; return this; }
        public Builder spatialMergeSize(int v) { this.spatialMergeSize = v; return this; }
        public Builder spatialMergeUnit(int v) { this.spatialMergeUnit = v; return this; }
        public Builder patchSize(int v) { this.patchSize = v; return this; }
        public Builder temporalPatchSize(int v) { this.temporalPatchSize = v; return this; }
        public Builder visionEmbedDim(int v) { this.visionEmbedDim = v; return this; }
        public Builder rmsNormEps(double v) { this.rmsNormEps = v; return this; }
        public Builder hiddenAct(String v) { this.hiddenAct = v; return this; }
        public Builder attentionBias(boolean v) { this.attentionBias = v; return this; }
        public Builder attentionDropout(boolean v) { this.attentionDropout = v; return this; }
        public Builder hiddenDropout(boolean v) { this.hiddenDropout = v; return this; }
        public Builder imageSize(int v) { this.imageSize = v; return this; }
        public Builder maxImageSize(int v) { this.maxImageSize = v; return this; }
        public Builder minImageSize(int v) { this.minImageSize = v; return this; }
        public Builder imageMean(float[] v) { this.imageMean = v; return this; }
        public Builder imageStd(float[] v) { this.imageStd = v; return this; }
        public Builder useImagenipt(boolean v) { this.useImagenipt = v; return this; }
        public Builder maxFrames(int v) { this.maxFrames = v; return this; }
        public Builder frameRate(float v) { this.frameRate = v; return this; }
        public Builder imageBoundTokenId(int v) { this.imageBoundTokenId = v; return this; }
        public Builder imageStartTokenId(int v) { this.imageStartTokenId = v; return this; }
        public Builder imageEndTokenId(int v) { this.imageEndTokenId = v; return this; }
        public Builder videoStartTokenId(int v) { this.videoStartTokenId = v; return this; }
        public Builder videoEndTokenId(int v) { this.videoEndTokenId = v; return this; }

        public QwenVisionConfig build() { return new QwenVisionConfig(this); }
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
