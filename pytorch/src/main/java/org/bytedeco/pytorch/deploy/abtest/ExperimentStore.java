/*
 * ExperimentStore — persistent store interface for experiments.
 *
 * Industry (Meta / Google / ByteDance / DoorDash):
 *   Experiments are stored in a config service (MySQL / Spanner / KV) with
 *   versioned immutable records and a change log. The runtime client SDK
 *   pulls snapshots via polling / push and applies them to in-memory cache.
 *
 * This module provides:
 *   - Storage interface (CRUD + versioning)
 *   - In-memory reference implementation (unit tests / local dev)
 *   - ChangeLog entry types (audit trail)
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Storage interface for experiments.
 *
 * <p>Implementations: {@link InMemoryExperimentStore}, or custom
 * (MySQL / Redis / config-service backed).
 */
public interface ExperimentStore {

    /** Persist an experiment (create or update by id). */
    void save(Experiment experiment, String actor);

    /** Retrieve by id; empty optional if missing. */
    Optional<Experiment> get(String experimentId);

    /** List all current experiments. */
    List<Experiment> list();

    /** List experiments in a layer. */
    List<Experiment> listByLayer(String layerId);

    /** Soft-delete experiment (mark as KILLED). */
    boolean delete(String experimentId, String actor);

    /** Return the most recent change log entries (newest first). */
    List<ChangeLog> recentChanges(int limit);

    /** Current snapshot version. */
    long version();

    /** Audit trail entry — every save / delete records one. */
    final class ChangeLog {
        public enum Action {
            CREATED,
            UPDATED,
            DELETED
        }

        public final long version;
        public final Action action;
        public final String experimentId;
        public final String actor;
        public final Instant timestamp;
        public final String detail;

        public ChangeLog(long version, Action action, String experimentId, String actor,
                         Instant timestamp, String detail) {
            this.version = version;
            this.action = action;
            this.experimentId = experimentId;
            this.actor = actor;
            this.timestamp = timestamp;
            this.detail = detail != null ? detail : "";
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT,
                    "ChangeLog{v=%d %s exp=%s actor=%s ts=%s detail=%s}",
                    version, action, experimentId, actor, timestamp, detail);
        }
    }

    /**
     * In-memory implementation suitable for tests and local development.
     *
     * <p>Production deployments should swap this for a MySQL / Redis backed
     * implementation behind the same interface.
     */
    final class InMemoryExperimentStore implements ExperimentStore {

        private final ConcurrentHashMap<String, Experiment> byId = new ConcurrentHashMap<>();
        private final List<ChangeLog> changes = Collections.synchronizedList(new ArrayList<>());
        private volatile long version = 0L;

        @Override
        public void save(Experiment experiment, String actor) {
            Objects.requireNonNull(experiment, "experiment");
            boolean created = !byId.containsKey(experiment.id());
            byId.put(experiment.id(), experiment);
            long v = ++version;
            changes.add(new ChangeLog(v,
                    created ? ChangeLog.Action.CREATED : ChangeLog.Action.UPDATED,
                    experiment.id(), actor, Instant.now(), ""));
        }

        @Override
        public Optional<Experiment> get(String experimentId) {
            return Optional.ofNullable(byId.get(experimentId));
        }

        @Override
        public List<Experiment> list() {
            return List.copyOf(byId.values());
        }

        @Override
        public List<Experiment> listByLayer(String layerId) {
            List<Experiment> out = new ArrayList<>();
            for (Experiment e : byId.values()) {
                if (e.layerId().equals(layerId)) out.add(e);
            }
            return out;
        }

        @Override
        public boolean delete(String experimentId, String actor) {
            Experiment e = byId.remove(experimentId);
            if (e == null) return false;
            long v = ++version;
            changes.add(new ChangeLog(v, ChangeLog.Action.DELETED, experimentId, actor,
                    Instant.now(), ""));
            return true;
        }

        @Override
        public List<ChangeLog> recentChanges(int limit) {
            synchronized (changes) {
                int from = Math.max(0, changes.size() - limit);
                List<ChangeLog> out = new ArrayList<>(changes.subList(from, changes.size()));
                Collections.reverse(out);
                return out;
            }
        }

        @Override
        public long version() {
            return version;
        }
    }
}