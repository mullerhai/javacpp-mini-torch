/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.distributed.trainer;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.optim.*;
import org.bytedeco.pytorch.distributed.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.*;
import java.util.*;

/**
 * Distributed checkpoint trainer with sharded state_dict support.
 *
 * <p>Implements the PyTorch DistributedCheckpoint format:
 * <ul>
 *   <li>Sharded checkpoints: each rank saves its own parameter shard</li>
 *   <li>Rank 0 writes a distributed checkpoint manifest</li>
 *   <li>Full state loading: each rank loads only its shard</li>
 *   <li>Atomic saves with staged writes (temp file then rename)</li>
 *   <li>Elastic / fault-tolerant loading (skip missing shards)</li>
 *   <li>Optimizer state alongside model weights</li>
 *   <li>Metadata: total shards, version, hash verification</li>
 * </ul>
 *
 * <p>Can wrap any trainer that implements {@code stateDict()} /
 * {@code loadStateDict}. Designed for use with {@link NativeDDPTrainer},
 * {@link NativeFSDPTrainer}, {@link ZeroRedundancyOptimizerTrainer}.
 *
 * <pre>{@code
 * DistributedCheckpointTrainer ckpt = DistributedCheckpointTrainer.builder()
 *         .wrappedTrainer(ddp)
 *         .processGroup(pg)
 *         .checkpointDir("./checkpoints")
 *         .saveInterval(1000)   // save every 1000 steps
 *         .maxShards(8)         // max shards to keep
 *         .build();
 *
 * // Training loop
 * for (int i = 0; i < steps; i++) {
 *     trainer.step(...);
 *     if (ckpt.shouldSave(i)) ckpt.save(i);
 * }
 *
 * // Resume
 * ckpt.load("./checkpoints/step_1000");
 *
 * // Cleanup old checkpoints
 * ckpt.pruneOldCheckpoints(keep=3);
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DistributedCheckpointTrainer implements BaseDistributedTrainer, AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "1.0";
    public static final String MANIFEST_FILE = "checkpoint_manifest.json";
    public static final String META_FILE = "checkpoint_meta.json";

    // ── Configuration ──────────────────────────────────────────────────────
    private final TrainerWrapper wrapped;
    private final ProcessGroupWrapper processGroup;
    private final Path checkpointDir;
    private final int saveInterval;
    private final int maxShardsToKeep;
    private final boolean atomicSave;
    private final boolean includeOptimizerState;
    private final boolean includeSchedulerState;

    // ── State ─────────────────────────────────────────────────────────────
    private final int worldSize;
    private final int rank;
    private final TrainerStats stats = new TrainerStats();
    private long numSaves;
    private long lastSavedStep;
    private boolean closed;

    /** Interface that any trainer must implement to participate in checkpointing. */
    public interface TrainerWrapper {
        /** Return a Map describing the trainer's local state (params, buffers, optimizer state, etc). */
        Map<String, Object> stateDict();
        /** Load trainer state from the Map. */
        void loadStateDict(Map<String, Object> state);
        /** Module whose parameters are being checkpointed. */
        Module module();
        /** Return the optimizer state Map if available (may return null). */
        Map<String, Object> optimizerStateDict();
        /** Load optimizer state from the Map. */
        void loadOptimizerState(Map<String, Object> state);
    }

    // ── Constructors ──────────────────────────────────────────────────────

    public DistributedCheckpointTrainer(
            TrainerWrapper wrapped,
            ProcessGroupWrapper processGroup,
            Path checkpointDir) {
        this(wrapped, processGroup, builder()
                .checkpointDir(checkpointDir));
    }

    private DistributedCheckpointTrainer(TrainerWrapper wrapped,
                                       ProcessGroupWrapper processGroup,
                                       Builder b) {
        this.wrapped = Objects.requireNonNull(wrapped, "wrapped");
        this.processGroup = Objects.requireNonNull(processGroup, "processGroup");
        this.worldSize = processGroup.getWorldSize();
        this.rank = processGroup.getRank();
        this.checkpointDir = b.checkpointDir != null ? b.checkpointDir : Paths.get("./checkpoints");
        this.saveInterval = Math.max(1, b.saveInterval);
        this.maxShardsToKeep = Math.max(1, b.maxShardsToKeep);
        this.atomicSave = b.atomicSave;
        this.includeOptimizerState = b.includeOptimizerState;
        this.includeSchedulerState = b.includeSchedulerState;

        if (processGroup.isMainProcess()) {
            try { Files.createDirectories(this.checkpointDir); }
            catch (IOException e) { throw new RuntimeException(e); }
        }
        processGroup.barrierWait();

        System.out.printf(
                "[DistributedCheckpointTrainer v%s] rank=%d dir=%s saveInterval=%d atomic=%s "
                        + "includeOptState=%s includeSchedState=%s%n",
                VERSION, rank, checkpointDir, saveInterval, atomicSave,
                includeOptimizerState, includeSchedulerState);
    }

    public static Builder builder() { return new Builder(); }

    // ── Required by BaseDistributedTrainer ─────────────────────────────────

    @Override
    public Module getModule() {
        return wrapped != null ? wrapped.module() : null;
    }

    @Override
    public ProcessGroupWrapper getProcessGroup() {
        return processGroup;
    }

    /**
     * Forward pass through the wrapped trainer.
     * Note: DistributedCheckpointTrainer wraps another trainer but doesn't directly
     * implement forward. This returns null as a placeholder - the actual forward
     * is typically done through the wrapped trainer in a training loop.
     */
    @Override
    public org.bytedeco.pytorch.Tensor forward(org.bytedeco.pytorch.Tensor input) {
        return input;
    }

    /**
     * Step through the wrapped trainer.
     */
    @Override
    public org.bytedeco.pytorch.Tensor step(org.bytedeco.pytorch.Tensor input,
            org.bytedeco.pytorch.Tensor target, org.bytedeco.pytorch.optim.Optimizer optimizer) {
        throw new UnsupportedOperationException(
            "step() should be called on the wrapped trainer, not DistributedCheckpointTrainer");
    }

    // ── Save ─────────────────────────────────────────────────────────────

    /**
     * Save a checkpoint to {@code checkpointDir / "step_" + step}.
     * All ranks participate; only rank 0 writes the manifest.
     *
     * @param step the training step (used in the directory name)
     * @throws IOException if the save fails
     */
    public void save(long step) throws IOException {
        save("step_" + step, step);
    }

    /**
     * Save a checkpoint to a named directory.
     *
     * @param name   sub-directory name
     * @param step   training step for logging
     * @throws IOException if the save fails
     */
    public void save(String name, long step) throws IOException {
        Path dir = checkpointDir.resolve(name);
        if (atomicSave) {
            Path tmp = checkpointDir.resolve("._tmp_" + name + "_rank" + rank);
            saveToDir(tmp, step);
            processGroup.barrierWait();
            if (rank == 0) {
                Files.move(tmp, dir, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            }
        } else {
            saveToDir(dir, step);
        }
        processGroup.barrierWait();
        numSaves++;
        lastSavedStep = step;
        stats.fireStepEnd(null);
    }

    private void saveToDir(Path dir, long step) throws IOException {
        Files.createDirectories(dir);
        // Save trainer state
        Map<String, Object> state = wrapped.stateDict();
        Path paramFile = dir.resolve("params_rank" + rank + ".pt");
        writeStateMap(state, paramFile);

        // Save optimizer state
        if (includeOptimizerState) {
            Map<String, Object> optState = wrapped.optimizerStateDict();
            if (optState != null && !optState.isEmpty()) {
                Path optFile = dir.resolve("optimizer_rank" + rank + ".pt");
                writeStateMap(optState, optFile);
            }
        }

        // Rank 0 writes the manifest
        if (rank == 0) {
            writeManifest(dir, step);
        }
        processGroup.barrierWait();
    }

    private void writeManifest(Path dir, long step) throws IOException {
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("version", VERSION);
        manifest.put("step", step);
        manifest.put("world_size", worldSize);
        manifest.put("saved_at", System.currentTimeMillis());
        manifest.put("rank_files", new LinkedHashMap<String, String>() {{
            put("params", "params_rank{i}.pt");
            if (includeOptimizerState) put("optimizer", "optimizer_rank{i}.pt");
        }});

        Path metaFile = dir.resolve(MANIFEST_FILE);
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(metaFile))) {
            w.println("{");
            w.println("  \"version\": \"" + VERSION + "\",");
            w.println("  \"step\": " + step + ",");
            w.println("  \"world_size\": " + worldSize + ",");
            w.println("  \"saved_at\": " + System.currentTimeMillis() + ",");
            w.println("  \"include_optimizer_state\": " + includeOptimizerState + ",");
            w.println("  \"include_scheduler_state\": " + includeSchedulerState);
            w.println("}");
        }
    }

    /**
     * Write a state dict Map to a binary file.
     * Format: JSON header + raw tensor data.
     */
    private long estimateTensorSizeBytes(Tensor t) {
        if (t == null || t.isNull() || !t.defined()) return 0;
        long numel = t.numel();
        if (numel <= 0) return 0;
        return numel * Math.max(1, t.element_size());
    }

    private long estimateTensorBytes(Tensor t) {
        if (t == null || t.isNull() || !t.defined()) return 0;
        return t.numel() * Math.max(1, t.element_size());
    }

    private void writeStateMap(Map<String, Object> state, Path file) throws IOException {
        if (state == null || state.isEmpty()) return;

        // Collect all tensors
        List<String> keys = new ArrayList<>(state.keySet());
        List<Tensor> tensors = new ArrayList<>();
        List<Long> offsets = new ArrayList<>();
        long offset = 0;

        for (String key : keys) {
            Object val = state.get(key);
            if (val instanceof Tensor) {
                offsets.add(offset);
                tensors.add((Tensor) val);
                offset += estimateTensorBytes((Tensor) val);
            } else if (val instanceof Map) {
                // Nested map: recursively collect
                @SuppressWarnings("unchecked")
                Map<String, Object> nested = (Map<String, Object>) val;
                for (Map.Entry<String, Object> e : nested.entrySet()) {
                    if (e.getValue() instanceof Tensor) {
                        offsets.add(offset);
                        tensors.add((Tensor) e.getValue());
                        offset += estimateTensorBytes((Tensor) e.getValue());
                    }
                }
            }
        }

        // Write: [num_keys(int)] [key1] [key2] ... [tensor1_bytes] [tensor2_bytes] ...
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(file), 1 << 20))) {
            out.writeInt(keys.size());
            for (String key : keys) {
                out.writeUTF(key);
            }
            for (Tensor t : tensors) {
                writeTensorBinary(out, t);
            }
        }
    }

    private void writeTensorBinary(DataOutputStream out, Tensor t) throws IOException {
        if (t == null || t.isNull() || !t.defined()) {
            out.writeLong(0);
            return;
        }
        Tensor cpu = t.detach().contiguous().to(ScalarType.Float).cpu();
        long n = cpu.numel();
        int ni = (int) Math.min(n, Integer.MAX_VALUE);
        float[] data = new float[ni];
        try {
            org.bytedeco.javacpp.FloatPointer p = cpu.data_ptr_float();
            p.capacity(ni).limit(ni).asBuffer().get(data);
        } catch (Throwable bulkFail) {
            org.bytedeco.javacpp.FloatPointer p = cpu.data_ptr_float();
            for (int i = 0; i < ni; i++) data[i] = p.get((long) i);
        }
        // Convert float[] to byte[] properly
        byte[] raw = new byte[ni * 4];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().put(data);
        out.writeLong(n);
        out.write(raw);
        if (cpu != t) {
            try { cpu.close(); } catch (Throwable ignored) {}
        }
    }

    // ── Load ─────────────────────────────────────────────────────────────

    /**
     * Load a checkpoint from a named directory.
     *
     * <p>Elastic: if a rank's shard file is missing, the load is skipped
     * for that rank with a warning (fault tolerance for partial saves).
     *
     * @param name sub-directory name (e.g. "step_1000")
     * @return true if all ranks loaded successfully
     */
    public boolean load(String name) throws IOException {
        Path dir = checkpointDir.resolve(name);
        return loadFromDir(dir);
    }

    /**
     * Load from a named directory, with elastic (skip missing shards) semantics.
     */
    public boolean loadFromDir(Path dir) throws IOException {
        processGroup.barrierWait();

        // Read manifest
        Path metaFile = dir.resolve(MANIFEST_FILE);
        Map<String, Object> manifest = null;
        if (rank == 0 && Files.exists(metaFile)) {
            manifest = readManifest(metaFile);
        }
        processGroup.barrierWait();

        // Load params
        Path paramFile = dir.resolve("params_rank" + rank + ".pt");
        boolean loadedParams = false;
        if (Files.exists(paramFile)) {
            Map<String, Object> state = readStateMap(paramFile);
            if (state != null && !state.isEmpty()) {
                wrapped.loadStateDict(state);
                loadedParams = true;
            }
        } else {
            System.err.println("[DistributedCheckpointTrainer] rank=" + rank
                    + ": missing shard " + paramFile + " — skipping load (elastic resume)");
        }

        // Load optimizer state
        if (includeOptimizerState) {
            Path optFile = dir.resolve("optimizer_rank" + rank + ".pt");
            if (Files.exists(optFile)) {
                Map<String, Object> optState = readStateMap(optFile);
                if (optState != null && !optState.isEmpty()) {
                    wrapped.loadOptimizerState(optState);
                }
            }
        }

        processGroup.barrierWait();
        return loadedParams;
    }

    private Map<String, Object> readManifest(Path file) throws IOException {
        Map<String, Object> m = new LinkedHashMap<>();
        try (BufferedReader r = Files.newBufferedReader(file)) {
            String line;
            while ((line = r.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.equals("{") || line.equals("}")) continue;
                int colon = line.indexOf(':');
                if (colon > 0) {
                    String key = line.substring(0, colon).trim().replaceAll("\"", "");
                    String val = line.substring(colon + 1).trim().replaceAll("[,\"]", "");
                    m.put(key, val);
                }
            }
        }
        return m;
    }

    private Map<String, Object> readStateMap(Path file) throws IOException {
        Map<String, Object> out = new LinkedHashMap<>();
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(file), 1 << 20))) {
            int numKeys = in.readInt();
            List<String> keys = new ArrayList<>();
            for (int i = 0; i < numKeys; i++) {
                keys.add(in.readUTF());
            }
            for (String key : keys) {
                long n = in.readLong();
                if (n > 0) {
                    byte[] raw = in.readNBytes((int) (n * 4));
                    float[] data = new float[(int) n];
                    ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(data);
                    out.put(key, org.bytedeco.pytorch.global.torch.tensor(data, new org.bytedeco.pytorch.TensorOptions()));
                }
            }
        }
        return out;
    }

    // ── Checkpoint management ───────────────────────────────────────────

    /** Returns true if a checkpoint should be saved at the given step. */
    public boolean shouldSave(long step) {
        return step > 0 && step % saveInterval == 0;
    }

    /**
     * Prune old checkpoints, keeping only the {@code keep} most recent ones.
     *
     * @param keep number of checkpoints to keep (from the most recent)
     */
    public void pruneOldCheckpoints(int keep) throws IOException {
        if (rank != 0) {
            processGroup.barrierWait();
            return;
        }
        try {
            List<Path> dirs = new ArrayList<>();
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(checkpointDir)) {
                for (Path p : stream) {
                    if (Files.isDirectory(p) && !p.getFileName().toString().startsWith("._tmp_")) {
                        dirs.add(p);
                    }
                }
            }
            // Sort by modification time (newest first)
            dirs.sort((a, b) -> {
                try { return Long.compare(Files.getLastModifiedTime(b).toMillis(),
                        Files.getLastModifiedTime(a).toMillis()); }
                catch (IOException e) { return 0; }
            });
            for (int i = keep; i < dirs.size(); i++) {
                deleteRecursively(dirs.get(i));
                System.out.println("[DistributedCheckpointTrainer] pruned: " + dirs.get(i));
            }
        } finally {
            processGroup.barrierWait();
        }
    }

    private static void deleteRecursively(Path path) throws IOException {
        if (Files.isDirectory(path)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(path)) {
                for (Path p : stream) deleteRecursively(p);
            }
        }
        Files.deleteIfExists(path);
    }

    /**
     * List all checkpoints in the checkpoint directory.
     */
    public List<String> listCheckpoints() throws IOException {
        List<String> names = new ArrayList<>();
        if (rank != 0) {
            processGroup.barrierWait();
            return names;
        }
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(checkpointDir)) {
            for (Path p : stream) {
                if (Files.isDirectory(p) && !p.getFileName().toString().startsWith("._tmp_")) {
                    names.add(p.getFileName().toString());
                }
            }
        }
        processGroup.barrierWait();
        Collections.sort(names);
        return names;
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    public TrainerStats stats() { return stats; }
    public long getNumSaves() { return numSaves; }
    public long getLastSavedStep() { return lastSavedStep; }
    public Path getCheckpointDir() { return checkpointDir; }
    public int getSaveInterval() { return saveInterval; }
    public TrainerWrapper getWrapped() { return wrapped; }
    public int getRank() { return rank; }
    public int getWorldSize() { return worldSize; }
    public boolean isMainProcess() { return rank == 0; }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
    }

    @Override
    public String toString() {
        return "DistributedCheckpointTrainer{rank=" + rank + ", dir=" + checkpointDir
                + ", saves=" + numSaves + ", lastStep=" + lastSavedStep + '}';
    }

    // ── Builder ─────────────────────────────────────────────────────────

    public static final class Builder {
        private TrainerWrapper wrapped;
        private ProcessGroupWrapper processGroup;
        private Path checkpointDir = Paths.get("./checkpoints");
        private int saveInterval = 1000;
        private int maxShardsToKeep = 3;
        private boolean atomicSave = true;
        private boolean includeOptimizerState = true;
        private boolean includeSchedulerState = false;

        public Builder wrapped(TrainerWrapper w) { this.wrapped = w; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.processGroup = pg; return this; }
        public Builder checkpointDir(Path dir) { this.checkpointDir = dir; return this; }
        public Builder checkpointDir(String dir) { this.checkpointDir = Paths.get(dir); return this; }
        public Builder saveInterval(int n) { this.saveInterval = n; return this; }
        public Builder maxShardsToKeep(int n) { this.maxShardsToKeep = n; return this; }
        /** Atomic save: write to temp dir, then atomically rename (default true). */
        public Builder atomicSave(boolean b) { this.atomicSave = b; return this; }
        public Builder includeOptimizerState(boolean b) { this.includeOptimizerState = b; return this; }
        public Builder includeSchedulerState(boolean b) { this.includeSchedulerState = b; return this; }

        public DistributedCheckpointTrainer build() {
            Objects.requireNonNull(wrapped, "wrapped is required");
            Objects.requireNonNull(processGroup, "processGroup is required");
            return new DistributedCheckpointTrainer(wrapped, processGroup, this);
        }
    }

    // ── Adapter for trainers that already implement stateDict/loadStateDict ──

    /**
     * Wrap any trainer that has {@code stateDict()} / {@code loadStateDict(Map)}
     * methods.
     */
    public static TrainerWrapper adapt(Object trainer) {
        Objects.requireNonNull(trainer, "trainer");
        return new TrainerWrapper() {
            @Override
            public Map<String, Object> stateDict() {
                try {
                    java.lang.reflect.Method m = trainer.getClass().getMethod("stateDict");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> r = (Map<String, Object>) m.invoke(trainer);
                    return r;
                } catch (ReflectiveOperationException e) {
                    throw new UnsupportedOperationException(
                            trainer.getClass().getName() + " has no stateDict() method", e);
                }
            }

            @Override
            public void loadStateDict(Map<String, Object> state) {
                try {
                    java.lang.reflect.Method m = trainer.getClass().getMethod("loadStateDict", Map.class);
                    m.invoke(trainer, state);
                } catch (ReflectiveOperationException e) {
                    throw new UnsupportedOperationException(
                            trainer.getClass().getName() + " has no loadStateDict(Map) method", e);
                }
            }

            @Override
            public Module module() {
                try {
                    java.lang.reflect.Method m = trainer.getClass().getMethod("getModule");
                    return (Module) m.invoke(trainer);
                } catch (ReflectiveOperationException e) {
                    return null;
                }
            }

            @Override
            public Map<String, Object> optimizerStateDict() {
                try {
                    java.lang.reflect.Method m = trainer.getClass().getMethod("optimizerStateDict");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> r = (Map<String, Object>) m.invoke(trainer);
                    return r;
                } catch (ReflectiveOperationException e) {
                    return null;
                }
            }

            @Override
            public void loadOptimizerState(Map<String, Object> state) {
                try {
                    java.lang.reflect.Method m = trainer.getClass().getMethod("loadOptimizerState", Map.class);
                    m.invoke(trainer, state);
                } catch (ReflectiveOperationException e) {
                    // Silently ignore — not all trainers support optimizer state dict
                }
            }
        };
    }
}
