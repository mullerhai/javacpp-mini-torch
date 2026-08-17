package org.bytedeco.pytorch.deploy.k8s;

import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public final class DeploymentResult {

    private final String name;
    private final String strategy;
    private final Instant startTime;
    private final Instant endTime;
    private final boolean success;
    private final Duration duration;
    private final String error;
    private final Map<String, String> metadata;

    private DeploymentResult(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        this.strategy = b.strategy;
        this.startTime = b.startTime;
        this.endTime = b.endTime;
        this.success = b.success;
        this.duration = b.duration;
        this.error = b.error;
        this.metadata = Map.copyOf(b.metadata);
    }

    public static DeploymentResult success(String name, String strategy) {
        return builder().name(name).strategy(strategy).success(true).build();
    }

    public static DeploymentResult failure(String name, String strategy, Throwable t) {
        return builder()
                .name(name)
                .strategy(strategy)
                .success(false)
                .error(t.getMessage())
                .build();
    }

    public static Builder builder() { return new Builder(); }

    public String name() { return name; }
    public String strategy() { return strategy; }
    public Instant startTime() { return startTime; }
    public Instant endTime() { return endTime; }
    public boolean success() { return success; }
    public Duration duration() { return duration; }
    public String error() { return error; }
    public Map<String, String> metadata() { return metadata; }

    public String toString() {
        return String.format("DeploymentResult{name='%s', strategy=%s, success=%s, duration=%s}",
                name, strategy, success, duration);
    }

    public static final class Builder {
        private String name;
        private String strategy;
        private Instant startTime;
        private Instant endTime;
        private boolean success;
        private Duration duration;
        private String error;
        private Map<String, String> metadata = new HashMap<>();

        public Builder name(String name) { this.name = name; return this; }
        public Builder strategy(String strategy) { this.strategy = strategy; return this; }
        public Builder startTime(Instant time) { this.startTime = time; return this; }
        public Builder endTime(Instant time) { this.endTime = time; return this; }
        public Builder success(boolean success) { this.success = success; return this; }
        public Builder duration(Duration duration) { this.duration = duration; return this; }
        public Builder error(String error) { this.error = error; return this; }
        public Builder metadata(Map<String, String> metadata) { this.metadata.putAll(metadata); return this; }
        public Builder metadata(String key, String value) { this.metadata.put(key, value); return this; }

        public DeploymentResult build() {
            if (startTime == null) startTime = Instant.now();
            if (endTime == null) endTime = Instant.now();
            if (duration == null) duration = Duration.between(startTime, endTime);
            return new DeploymentResult(this);
        }
    }
}
