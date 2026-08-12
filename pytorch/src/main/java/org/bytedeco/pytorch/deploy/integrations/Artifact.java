/*
 * Artifact descriptor (model file, dataset, plot, etc.).
 */
package org.bytedeco.pytorch.deploy.integrations;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Artifact descriptor (model file, dataset, plot, etc.).
 */
public final class Artifact {
    public final String name;
    public final String uri;
    public final ArtifactKind kind;
    public final long sizeBytes;
    public final String contentType;
    public final Map<String, String> metadata;

    public Artifact(String name, String uri, ArtifactKind kind) {
        this(name, uri, kind, -1L, "", Collections.emptyMap());
    }

    public Artifact(String name, String uri, ArtifactKind kind, long sizeBytes,
                     String contentType, Map<String, String> metadata) {
        this.name = Objects.requireNonNull(name, "name");
        this.uri = Objects.requireNonNull(uri, "uri");
        this.kind = kind != null ? kind : ArtifactKind.FILE;
        this.sizeBytes = sizeBytes;
        this.contentType = contentType != null ? contentType : "";
        this.metadata = metadata == null
                ? Collections.emptyMap()
                : Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
    }

    public enum ArtifactKind {
        MODEL,
        CHECKPOINT,
        DATASET,
        FEATURE_STORE,
        PLOT,
        REPORT,
        LOG,
        FILE
    }
}
