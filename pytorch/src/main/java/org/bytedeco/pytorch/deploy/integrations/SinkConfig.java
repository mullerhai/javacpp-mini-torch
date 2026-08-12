/*
 * Sink configuration value object.
 */
package org.bytedeco.pytorch.deploy.integrations;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Sink configuration.
 */
public final class SinkConfig {
    public final String endpointUrl;
    public final String authToken;
    public final String projectNamespace;
    public final String storageRoot;
    public final boolean async;
    public final int batchSize;
    public final Map<String, String> extras;

    private SinkConfig(Builder b) {
        this.endpointUrl = b.endpointUrl;
        this.authToken = b.authToken;
        this.projectNamespace = b.projectNamespace != null ? b.projectNamespace : "default";
        this.storageRoot = b.storageRoot != null ? b.storageRoot : "/tmp";
        this.async = b.async;
        this.batchSize = b.batchSize > 0 ? b.batchSize : 32;
        this.extras = Map.copyOf(b.extras);
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String endpointUrl;
        private String authToken;
        private String projectNamespace;
        private String storageRoot;
        private boolean async = true;
        private int batchSize = 32;
        private final Map<String, String> extras = new LinkedHashMap<>();

        public Builder endpointUrl(String u) { this.endpointUrl = u; return this; }
        public Builder authToken(String t) { this.authToken = t; return this; }
        public Builder projectNamespace(String n) { this.projectNamespace = n; return this; }
        public Builder storageRoot(String r) { this.storageRoot = r; return this; }
        public Builder async(boolean a) { this.async = a; return this; }
        public Builder batchSize(int n) { this.batchSize = n; return this; }
        public Builder extra(String k, String v) { this.extras.put(k, v); return this; }

        public SinkConfig build() { return new SinkConfig(this); }
    }
}
