/*
 * Enhanced Candidate — enterprise-grade recommendation candidate.
 *
 * Key enhancements:
 *   1. Rich scoring with multiple dimensions
 *   2. Multi-modal features
 *   3. Embedding storage
 *   4. Exposure and impression tracking
 *   5. Business metadata
 *   6. Quality signals
 *   7. Cold-start features
 *   8. Serialization support
 *
 * Production patterns (Meta, YouTube, Alibaba, ByteDance):
 *   - Multi-task scores (CTR, CVR, dwell time)
 *   - Embeddings for similarity
 *   - Rich metadata for filtering/reranking
 *   - Impression history for dampening
 *   - Quality signals from upstream systems
 */
package org.bytedeco.pytorch.deploy.serving.pipeline;

import java.io.Serializable;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Enterprise-grade recommendation candidate with comprehensive metadata.
 */
public final class CandidateV2 implements Serializable {

    private static final long serialVersionUID = 1L;

    // ---- Core identity ----
    private final String itemId;
    private final String itemType;
    private final long createdAtMs;

    // ---- Scores (immutable primary score, mutable for intermediate stages) ----
    private double score;
    private final Map<String, Double> subScores;

    // ---- Recall sources ----
    private final List<String> recallChannels;
    private final Map<String, Double> channelScores;

    // ---- Embeddings ----
    private final Map<String, double[]> embeddings;
    private final Map<String, float[]> embeddingsFloat;

    // ---- Metadata ----
    private final Map<String, String> tags;
    private final Map<String, Double> numericFeatures;
    private final Map<String, String> attributes;

    // ---- Business signals ----
    private final Map<String, Long> timestamps;
    private final Map<String, Integer> counters;
    private final Map<String, List<String>> lists;

    // ---- Runtime state (mutable) ----
    private int rank;
    private double exposureBias;
    private double qualityScore;
    private String qualityLevel;  // HIGH, MEDIUM, LOW

    // ---- Quality signals ----
    private double ctrEstimate;
    private double cvrEstimate;
    private double price;
    private double popularity;
    private int freshnessHours;

    public CandidateV2(String itemId) {
        this(itemId, 0.0, "unknown");
    }

    public CandidateV2(String itemId, double score) {
        this(itemId, score, "unknown");
    }

    public CandidateV2(String itemId, double score, String itemType) {
        this.itemId = Objects.requireNonNull(itemId, "itemId");
        this.score = score;
        this.itemType = itemType != null ? itemType : "unknown";
        this.createdAtMs = System.currentTimeMillis();
        this.subScores = new ConcurrentHashMap<>();
        this.recallChannels = new CopyOnWriteArrayList<>();
        this.channelScores = new ConcurrentHashMap<>();
        this.embeddings = new ConcurrentHashMap<>();
        this.embeddingsFloat = new ConcurrentHashMap<>();
        this.tags = new ConcurrentHashMap<>();
        this.numericFeatures = new ConcurrentHashMap<>();
        this.attributes = new ConcurrentHashMap<>();
        this.timestamps = new ConcurrentHashMap<>();
        this.counters = new ConcurrentHashMap<>();
        this.lists = new ConcurrentHashMap<>();
        this.rank = -1;
    }

    // ---- Core getters ----

    public String itemId() { return itemId; }
    public String itemType() { return itemType; }
    public double score() { return score; }
    public int rank() { return rank; }
    public long createdAtMs() { return createdAtMs; }

    public double exposureBias() { return exposureBias; }
    public double qualityScore() { return qualityScore; }
    public String qualityLevel() { return qualityLevel; }

    public double ctrEstimate() { return ctrEstimate; }
    public double cvrEstimate() { return cvrEstimate; }
    public double price() { return price; }
    public double popularity() { return popularity; }
    public int freshnessHours() { return freshnessHours; }

    // ---- Score manipulation ----

    public CandidateV2 score(double score) {
        this.score = score;
        return this;
    }

    public CandidateV2 adjustScore(double delta) {
        this.score += delta;
        return this;
    }

    public CandidateV2 multiplyScore(double factor) {
        this.score *= factor;
        return this;
    }

    // ---- Rank ----

    public CandidateV2 rank(int rank) {
        this.rank = rank;
        return this;
    }

    // ---- Sub-scores ----

    public CandidateV2 putScore(String name, double value) {
        subScores.put(name, value);
        return this;
    }

    public double getScore(String name) {
        return subScores.getOrDefault(name, 0.0);
    }

    public double getScore(String name, double defaultValue) {
        return subScores.getOrDefault(name, defaultValue);
    }

    public Map<String, Double> scores() {
        return Collections.unmodifiableMap(subScores);
    }

    // ---- Recall channels ----

    public CandidateV2 addRecallChannel(String channel) {
        if (channel != null && !recallChannels.contains(channel)) {
            recallChannels.add(channel);
        }
        return this;
    }

    public CandidateV2 addRecallChannel(String channel, double channelScore) {
        addRecallChannel(channel);
        channelScores.put(channel, channelScore);
        return this;
    }

    public List<String> recallChannels() {
        return Collections.unmodifiableList(recallChannels);
    }

    public double recallChannelScore(String channel) {
        return channelScores.getOrDefault(channel, 0.0);
    }

    public boolean fromChannel(String channel) {
        return recallChannels.contains(channel);
    }

    public int numRecallChannels() {
        return recallChannels.size();
    }

    // ---- Embeddings ----

    public CandidateV2 embedding(String name, double[] embedding) {
        embeddings.put(name, embedding);
        return this;
    }

    public CandidateV2 embedding(String name, float[] embedding) {
        embeddingsFloat.put(name, embedding);
        return this;
    }

    public double[] embedding(String name) {
        return embeddings.get(name);
    }

    public float[] embeddingFloat(String name) {
        return embeddingsFloat.get(name);
    }

    public boolean hasEmbedding(String name) {
        return embeddings.containsKey(name) || embeddingsFloat.containsKey(name);
    }

    public Set<String> embeddingNames() {
        Set<String> names = new HashSet<>(embeddings.keySet());
        names.addAll(embeddingsFloat.keySet());
        return names;
    }

    public double embeddingSimilarity(String embeddingName, double[] queryEmbedding) {
        double[] embedding = embeddings.get(embeddingName);
        if (embedding == null) return 0.0;
        return cosineSimilarity(embedding, queryEmbedding);
    }

    private static double cosineSimilarity(double[] a, double[] b) {
        if (a.length != b.length) return 0.0;
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        double denom = Math.sqrt(normA) * Math.sqrt(normB);
        return denom > 0 ? dot / denom : 0.0;
    }

    // ---- Tags ----

    public CandidateV2 tag(String key, String value) {
        tags.put(key, value);
        return this;
    }

    public String tag(String key) {
        return tags.get(key);
    }

    public String setTag(String key, String defaultValue) {
        return tags.getOrDefault(key, defaultValue);
    }

    public Map<String, String> tags() {
        return Collections.unmodifiableMap(tags);
    }

    public boolean hasTag(String key) {
        return tags.containsKey(key);
    }

    // ---- Numeric features ----

    public CandidateV2 feature(String key, double value) {
        numericFeatures.put(key, value);
        return this;
    }

    public double feature(String key) {
        return numericFeatures.getOrDefault(key, 0.0);
    }

    public double setFeature(String key, double defaultValue) {
        return numericFeatures.getOrDefault(key, defaultValue);
    }

    public Map<String, Double> features() {
        return Collections.unmodifiableMap(numericFeatures);
    }

    // ---- Attributes ----

    public CandidateV2 attribute(String key, String value) {
        attributes.put(key, value);
        return this;
    }

    public String attribute(String key) {
        return attributes.get(key);
    }

    public Map<String, String> attributes() {
        return Collections.unmodifiableMap(attributes);
    }

    // ---- Timestamps ----

    public CandidateV2 timestamp(String key, long timestampMs) {
        timestamps.put(key, timestampMs);
        return this;
    }

    public long timestamp(String key) {
        return timestamps.getOrDefault(key, createdAtMs);
    }

    public long ageMs() {
        return System.currentTimeMillis() - createdAtMs;
    }

    public int ageHours() {
        return (int) (ageMs() / 3600000);
    }

    public boolean isFresh(int maxAgeHours) {
        return ageHours() <= maxAgeHours;
    }

    // ---- Counters ----

    public CandidateV2 incrementCounter(String key) {
        counters.merge(key, 1, Integer::sum);
        return this;
    }

    public CandidateV2 incrementCounter(String key, int delta) {
        counters.merge(key, delta, Integer::sum);
        return this;
    }

    public int counter(String key) {
        return counters.getOrDefault(key, 0);
    }

    public Map<String, Integer> counters() {
        return Collections.unmodifiableMap(counters);
    }

    // ---- Lists ----

    public CandidateV2 addToList(String key, String value) {
        lists.computeIfAbsent(key, k -> new CopyOnWriteArrayList<>()).add(value);
        return this;
    }

    public List<String> list(String key) {
        return lists.getOrDefault(key, List.of());
    }

    // ---- Quality signals ----

    public CandidateV2 ctrEstimate(double ctr) {
        this.ctrEstimate = ctr;
        return this;
    }

    public CandidateV2 cvrEstimate(double cvr) {
        this.cvrEstimate = cvr;
        return this;
    }

    public CandidateV2 price(double price) {
        this.price = price;
        return this;
    }

    public CandidateV2 popularity(double popularity) {
        this.popularity = popularity;
        return this;
    }

    public CandidateV2 freshnessHours(int hours) {
        this.freshnessHours = hours;
        return this;
    }

    public CandidateV2 qualityScore(double score, String level) {
        this.qualityScore = score;
        this.qualityLevel = level;
        return this;
    }

    public CandidateV2 exposureBias(double bias) {
        this.exposureBias = bias;
        return this;
    }

    // ---- Computed properties ----

    public double eCpm() {
        return ctrEstimate * cvrEstimate * price * 1000;
    }

    public double adjustedScore(double exposureWeight) {
        double dampen = 1.0 / (1.0 + exposureWeight * Math.log1p(counter("impressions")));
        return score * dampen;
    }

    public double blendedScore(double ctrWeight, double cvrWeight, double qualityWeight) {
        return score
                + ctrWeight * ctrEstimate
                + cvrWeight * cvrEstimate
                + qualityWeight * qualityScore;
    }

    // ---- Copy ----

    public CandidateV2 copy() {
        CandidateV2 c = new CandidateV2(itemId, score, itemType);
        c.rank = rank;
        c.exposureBias = exposureBias;
        c.qualityScore = qualityScore;
        c.qualityLevel = qualityLevel;
        c.ctrEstimate = ctrEstimate;
        c.cvrEstimate = cvrEstimate;
        c.price = price;
        c.popularity = popularity;
        c.freshnessHours = freshnessHours;

        c.subScores.putAll(subScores);
        c.recallChannels.addAll(recallChannels);
        c.channelScores.putAll(channelScores);
        c.embeddings.putAll(embeddings);
        c.embeddingsFloat.putAll(embeddingsFloat);
        c.tags.putAll(tags);
        c.numericFeatures.putAll(numericFeatures);
        c.attributes.putAll(attributes);
        c.timestamps.putAll(timestamps);
        c.counters.putAll(counters);
        for (Map.Entry<String, List<String>> e : lists.entrySet()) {
            c.lists.put(e.getKey(), new CopyOnWriteArrayList<>(e.getValue()));
        }

        return c;
    }

    // ---- Serialization ----

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("item_id", itemId);
        map.put("item_type", itemType);
        map.put("score", score);
        map.put("rank", rank);
        map.put("scores", scores());
        map.put("recall_channels", recallChannels());
        map.put("tags", tags());
        map.put("features", features());
        map.put("created_at_ms", createdAtMs);
        return map;
    }

    @Override
    public String toString() {
        return String.format("CandidateV2{id=%s score=%.4f rank=%d channels=%s type=%s}",
                itemId, score, rank, recallChannels, itemType);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof CandidateV2)) return false;
        return itemId.equals(((CandidateV2) o).itemId);
    }

    @Override
    public int hashCode() {
        return itemId.hashCode();
    }

    // ---- Builder ----

    public static Builder builder(String itemId) {
        return new Builder(itemId);
    }

    public static final class Builder {
        private final String itemId;
        private double score = 0.0;
        private String itemType = "unknown";

        private Builder(String itemId) {
            this.itemId = itemId;
        }

        public Builder score(double score) { this.score = score; return this; }
        public Builder itemType(String type) { this.itemType = type; return this; }

        public CandidateV2 build() {
            return new CandidateV2(itemId, score, itemType);
        }
    }
}
