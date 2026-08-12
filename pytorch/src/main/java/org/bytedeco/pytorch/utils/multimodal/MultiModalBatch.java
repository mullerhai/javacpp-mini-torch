/*
 * Multi-modal batch container.
 *
 * <p>Holds a batch of heterogeneous modality data (image, audio, video, text).
 * Used by {@link MultiModalDataLoader} and downstream consumers.
 */
package org.bytedeco.pytorch.utils.multimodal;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One batch of multi-modal data.
 */
public final class MultiModalBatch {
    private final Map<String, Object> data;
    private final long batchSize;

    public MultiModalBatch(Map<String, Object> data, long batchSize) {
        this.data = data == null
                ? Collections.emptyMap()
                : Collections.unmodifiableMap(new LinkedHashMap<>(data));
        this.batchSize = batchSize;
    }

    public Map<String, Object> data() { return data; }
    public long batchSize() { return batchSize; }

    /** Get a typed batch entry by modality key (e.g. "image", "audio", "text"). */
    @SuppressWarnings("unchecked")
    public <T> T get(String key) {
        return (T) data.get(key);
    }

    /** Empty batch placeholder. */
    public static MultiModalBatch empty() {
        return new MultiModalBatch(Collections.emptyMap(), 0);
    }
}
