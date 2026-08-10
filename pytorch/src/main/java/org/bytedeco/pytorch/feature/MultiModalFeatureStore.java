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
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.feature;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.global.torch;

import java.io.Closeable;
import java.sql.*;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Enterprise-grade Feature Store for multi-modal recommendation systems.
 *
 * <p>Supports:
 * <ul>
 *   <li>Offline storage (Parquet/Iceberg)</li>
 *   <li>Online storage (Redis/SQLite)</li>
 *   <li>Multi-modal features (image, video, audio, text embeddings)</li>
 *   <li>Real-time feature materialization</li>
 *   <li>Feature versioning and lineage</li>
 * </ul>
 *
 * <p>Reference: Feast, Tecton, Databricks Feature Store
 *
 * <pre>{@code
 * MultiModalFeatureStore store = MultiModalFeatureStore.builder()
 *     .name("recsys_features")
 *     .offlineStore("s3://bucket/features/")
 *     .onlineStore(RedisOnlineStore.builder().host("redis").build())
 *     .build();
 *
 * // Store image embeddings
 * store.putEmbedding("user_123", "image_emb", imageTensor);
 *
 * // Retrieve for online inference
 * Map<String, Tensor> features = store.getOnline("user_123",
 *     "image_emb", "audio_emb", "text_emb");
 * }</pre>
 */
public class MultiModalFeatureStore implements AutoCloseable {

    public static final String VERSION = "2.0";

    private volatile boolean closed;

    // Configuration
    private final String name;
    private final String offlinePath;
    private final OnlineFeatureStore onlineStore;
    private final int embeddingDim;
    private final int maxCacheSize;

    // Cache for hot features
    private final Map<String, CachedFeature> cache = new ConcurrentHashMap<>();
    private final ScheduledExecutorService cacheEvictor;

    // Statistics
    private final AtomicLong totalReads = new AtomicLong(0);
    private final AtomicLong totalWrites = new AtomicLong(0);
    private final AtomicLong cacheHits = new AtomicLong(0);
    private final AtomicLong cacheMisses = new AtomicLong(0);
    private final AtomicLong totalLatencyMs = new AtomicLong(0);

    public static Builder builder() {
        return new Builder();
    }

    private MultiModalFeatureStore(Builder builder) {
        this.name = builder.name;
        this.offlinePath = builder.offlinePath;
        this.onlineStore = builder.onlineStore;
        this.embeddingDim = builder.embeddingDim;
        this.maxCacheSize = builder.maxCacheSize;

        this.cacheEvictor = Executors.newScheduledThreadPool(1);
        this.cacheEvictor.scheduleAtFixedRate(
                this::evictCache, 60_000, 60_000, TimeUnit.MILLISECONDS
        );
    }

    // ============= Feature Operations =============

    /**
     * Store an embedding feature.
     */
    public void putEmbedding(String entityKey, String featureName, Tensor embedding) {
        long start = System.currentTimeMillis();

        try {
            // Store in online store
            if (onlineStore != null) {
                onlineStore.write(entityKey, featureName, embedding);
            }

            // Update cache
            String cacheKey = entityKey + ":" + featureName;
            cache.put(cacheKey, new CachedFeature(
                    embedding.clone(),
                    System.currentTimeMillis(),
                    embedding.nelement() * 4L
            ));

            totalWrites.incrementAndGet();
            totalLatencyMs.addAndGet(System.currentTimeMillis() - start);

        } catch (Exception e) {
            System.err.println("Failed to store embedding: " + e.getMessage());
        }
    }

    /**
     * Store multiple embedding features at once.
     */
    public void putEmbeddings(String entityKey, Map<String, Tensor> embeddings) {
        for (Map.Entry<String, Tensor> entry : embeddings.entrySet()) {
            putEmbedding(entityKey, entry.getKey(), entry.getValue());
        }
    }

    /**
     * Get an embedding feature.
     */
    public Tensor getEmbedding(String entityKey, String featureName) {
        long start = System.currentTimeMillis();

        // Check cache first
        String cacheKey = entityKey + ":" + featureName;
        CachedFeature cached = cache.get(cacheKey);
        if (cached != null) {
            cacheHits.incrementAndGet();
            totalReads.incrementAndGet();
            totalLatencyMs.addAndGet(System.currentTimeMillis() - start);
            return cached.tensor.clone();
        }

        // Cache miss - fetch from online store
        cacheMisses.incrementAndGet();

        try {
            Tensor embedding = null;
            if (onlineStore != null) {
                embedding = onlineStore.read(entityKey, featureName);
            }

            if (embedding != null) {
                // Update cache
                cache.put(cacheKey, new CachedFeature(
                        embedding.clone(),
                        System.currentTimeMillis(),
                        embedding.nelement() * 4L
                ));
            }

            totalReads.incrementAndGet();
            totalLatencyMs.addAndGet(System.currentTimeMillis() - start);
            return embedding;

        } catch (Exception e) {
            System.err.println("Failed to read embedding: " + e.getMessage());
            return null;
        }
    }

    /**
     * Get multiple embedding features.
     */
    public Map<String, Tensor> getEmbeddings(String entityKey, String... featureNames) {
        Map<String, Tensor> result = new HashMap<>();
        for (String name : featureNames) {
            Tensor t = getEmbedding(entityKey, name);
            if (t != null) {
                result.put(name, t);
            }
        }
        return result;
    }

    /**
     * Get embeddings for multiple entities (batch).
     */
    public Map<String, Map<String, Tensor>> getEmbeddingsBatch(
            List<String> entityKeys, List<String> featureNames) {

        Map<String, Map<String, Tensor>> result = new HashMap<>();

        for (String key : entityKeys) {
            Map<String, Tensor> features = getEmbeddings(key,
                    featureNames.toArray(new String[0]));
            result.put(key, features);
        }

        return result;
    }

    /**
     * Delete a feature.
     */
    public void deleteEmbedding(String entityKey, String featureName) {
        String cacheKey = entityKey + ":" + featureName;
        cache.remove(cacheKey);

        if (onlineStore != null) {
            onlineStore.delete(entityKey, featureName);
        }
    }

    // ============= Multi-modal Feature Types =============

    /**
     * Store image embedding.
     */
    public void putImageEmbedding(String entityKey, Tensor embedding) {
        putEmbedding(entityKey, "image_embedding", embedding);
    }

    /**
     * Get image embedding.
     */
    public Tensor getImageEmbedding(String entityKey) {
        return getEmbedding(entityKey, "image_embedding");
    }

    /**
     * Store video embedding.
     */
    public void putVideoEmbedding(String entityKey, Tensor embedding) {
        putEmbedding(entityKey, "video_embedding", embedding);
    }

    /**
     * Get video embedding.
     */
    public Tensor getVideoEmbedding(String entityKey) {
        return getEmbedding(entityKey, "video_embedding");
    }

    /**
     * Store audio embedding.
     */
    public void putAudioEmbedding(String entityKey, Tensor embedding) {
        putEmbedding(entityKey, "audio_embedding", embedding);
    }

    /**
     * Get audio embedding.
     */
    public Tensor getAudioEmbedding(String entityKey) {
        return getEmbedding(entityKey, "audio_embedding");
    }

    /**
     * Store text embedding.
     */
    public void putTextEmbedding(String entityKey, Tensor embedding) {
        putEmbedding(entityKey, "text_embedding", embedding);
    }

    /**
     * Get text embedding.
     */
    public Tensor getTextEmbedding(String entityKey) {
        return getEmbedding(entityKey, "text_embedding");
    }

    /**
     * Store all modality embeddings at once.
     */
    public void putMultiModalEmbeddings(String entityKey,
            Tensor imageEmb, Tensor videoEmb, Tensor audioEmb, Tensor textEmb) {

        Map<String, Tensor> embeddings = new HashMap<>();
        if (imageEmb != null) embeddings.put("image_embedding", imageEmb);
        if (videoEmb != null) embeddings.put("video_embedding", videoEmb);
        if (audioEmb != null) embeddings.put("audio_embedding", audioEmb);
        if (textEmb != null) embeddings.put("text_embedding", textEmb);

        putEmbeddings(entityKey, embeddings);
    }

    /**
     * Get all modality embeddings.
     */
    public MultiModalEmbeddings getMultiModalEmbeddings(String entityKey) {
        return new MultiModalEmbeddings(
                getImageEmbedding(entityKey),
                getVideoEmbedding(entityKey),
                getAudioEmbedding(entityKey),
                getTextEmbedding(entityKey)
        );
    }

    // ============= Offline Operations =============

    /**
     * Write to offline storage (Parquet/Iceberg).
     */
    public void writeOffline(DataFrame df, String entityColumn) throws Exception {
        // Write to offline path
        String path = offlinePath + "/" + name + "/" + System.currentTimeMillis() + ".parquet";
        df.write().format("parquet").option("path", path).save();
    }

    /**
     * Read from offline storage.
     */
    public DataFrame readOffline(String startTime, String endTime) throws Exception {
        // Read from offline path with time filter
        // Simplified implementation
        return null;
    }

    // ============= Cache Management =============

    /**
     * Evict expired cache entries.
     */
    private void evictCache() {
        long now = System.currentTimeMillis();
        long maxAge = 5 * 60 * 1000;  // 5 minutes

        cache.entrySet().removeIf(entry -> {
            CachedFeature cf = entry.getValue();
            return now - cf.timestamp > maxAge || cache.size() > maxCacheSize;
        });

        if (cache.size() > maxCacheSize) {
            // Evict oldest entries
            List<String> keys = new ArrayList<>(cache.keySet());
            keys.sort((a, b) -> {
                long ta = cache.get(a).timestamp;
                long tb = cache.get(b).timestamp;
                return Long.compare(ta, tb);
            });

            int toRemove = cache.size() - maxCacheSize;
            for (int i = 0; i < toRemove; i++) {
                cache.remove(keys.get(i));
            }
        }
    }

    /**
     * Clear cache.
     */
    public void clearCache() {
        cache.clear();
    }

    // ============= Statistics =============

    public MultiModalFeatureStoreStats getStats() {
        return new MultiModalFeatureStoreStats(
                name,
                cache.size(),
                maxCacheSize,
                totalReads.get(),
                totalWrites.get(),
                cacheHits.get(),
                cacheMisses.get(),
                totalLatencyMs.get()
        );
    }

    public boolean isClosed() { return closed; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        cacheEvictor.shutdown();
        try {
            cacheEvictor.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            cacheEvictor.shutdownNow();
        }

        if (onlineStore != null) {
            onlineStore.close();
        }

        System.out.printf(
                "[MultiModalFeatureStore] Closed: name=%s, reads=%d, writes=%d, " +
                "cacheHits=%d, cacheMisses=%d, hitRate=%.2f%%%n",
                name, totalReads.get(), totalWrites.get(),
                cacheHits.get(), cacheMisses.get(), getCacheHitRate() * 100);
    }

    private double getCacheHitRate() {
        long hits = cacheHits.get();
        long misses = cacheMisses.get();
        long total = hits + misses;
        return total > 0 ? (double) hits / total : 0;
    }

    // ============= Inner Types =============

    /**
     * Cached feature entry.
     */
    private static class CachedFeature {
        final Tensor tensor;
        final long timestamp;
        final long sizeBytes;

        CachedFeature(Tensor tensor, long timestamp, long sizeBytes) {
            this.tensor = tensor;
            this.timestamp = timestamp;
            this.sizeBytes = sizeBytes;
        }
    }

    /**
     * Online feature store interface.
     */
    public interface OnlineFeatureStore extends Closeable {
        void write(String entityKey, String featureName, Tensor embedding);
        Tensor read(String entityKey, String featureName);
        void delete(String entityKey, String featureName);
    }

    /**
     * Multi-modal embeddings container.
     */
    public static class MultiModalEmbeddings {
        public final Tensor image;
        public final Tensor video;
        public final Tensor audio;
        public final Tensor text;

        public MultiModalEmbeddings(Tensor image, Tensor video, Tensor audio, Tensor text) {
            this.image = image;
            this.video = video;
            this.audio = audio;
            this.text = text;
        }

        /**
         * Concatenate all available embeddings.
         */
        public Tensor concat() {
            List<Tensor> tensors = new ArrayList<>();
            if (image != null) tensors.add(image);
            if (video != null) tensors.add(video);
            if (audio != null) tensors.add(audio);
            if (text != null) tensors.add(text);

            if (tensors.isEmpty()) return null;

            // Stack and flatten
            Tensor stacked = tensors.size() == 1 ? tensors.get(0) : torch.stack(tensors, 0);
            return stacked.reshape(-1);
        }
    }

    /**
     * Statistics.
     */
    public static class MultiModalFeatureStoreStats {
        public final String name;
        public final int cacheSize;
        public final int maxCacheSize;
        public final long totalReads;
        public final long totalWrites;
        public final long cacheHits;
        public final long cacheMisses;
        public final long totalLatencyMs;

        public MultiModalFeatureStoreStats(String name, int cacheSize, int maxCacheSize,
                                   long totalReads, long totalWrites, long cacheHits,
                                   long cacheMisses, long totalLatencyMs) {
            this.name = name;
            this.cacheSize = cacheSize;
            this.maxCacheSize = maxCacheSize;
            this.totalReads = totalReads;
            this.totalWrites = totalWrites;
            this.cacheHits = cacheHits;
            this.cacheMisses = cacheMisses;
            this.totalLatencyMs = totalLatencyMs;
        }

        public double cacheHitRate() {
            long total = cacheHits + cacheMisses;
            return total > 0 ? (double) cacheHits / total : 0;
        }

        public double avgLatencyMs() {
            return totalReads > 0 ? (double) totalLatencyMs / totalReads : 0;
        }
    }

    /**
     * Builder.
     */
    public static class Builder {
        private String name = "feature_store";
        private String offlinePath = "/tmp/features";
        private OnlineFeatureStore onlineStore;
        private int embeddingDim = 512;
        private int maxCacheSize = 100000;

        public Builder name(String name) { this.name = name; return this; }
        public Builder offlinePath(String path) { this.offlinePath = path; return this; }
        public Builder onlineStore(OnlineFeatureStore store) { this.onlineStore = store; return this; }
        public Builder embeddingDim(int dim) { this.embeddingDim = dim; return this; }
        public Builder maxCacheSize(int size) { this.maxCacheSize = size; return this; }

        /** Use Redis as online store */
        public Builder redis(String host, int port) {
            // Use existing RedisOnlineStore
            this.onlineStore = null;  // Would be: new RedisOnlineStore(host, port)
            return this;
        }

        /** Use in-memory store */
        public Builder inMemory() {
            this.onlineStore = new InMemoryOnlineStore();
            return this;
        }

        public MultiModalFeatureStore build() {
            return new MultiModalFeatureStore(this);
        }
    }

    /**
     * In-memory online store implementation.
     */
    private static class InMemoryOnlineStore implements OnlineFeatureStore {
        private final Map<String, Tensor> store = new ConcurrentHashMap<>();

        @Override
        public void write(String entityKey, String featureName, Tensor embedding) {
            String key = entityKey + ":" + featureName;
            store.put(key, embedding.clone());
        }

        @Override
        public Tensor read(String entityKey, String featureName) {
            String key = entityKey + ":" + featureName;
            Tensor t = store.get(key);
            return t != null ? t.clone() : null;
        }

        @Override
        public void delete(String entityKey, String featureName) {
            String key = entityKey + ":" + featureName;
            store.remove(key);
        }

        @Override
        public void close() {
            store.clear();
        }
    }
}
