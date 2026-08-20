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
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.transformers.distributed;

import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.distributed.config.DistributedConfig;
import org.bytedeco.pytorch.llm.hub.HfHub;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.llm.transformers.AutoTokenizer;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.llm.transformers.loading.WeightLoader;
import org.bytedeco.pytorch.llm.transformers.mapping.ModelRegistry;
import org.bytedeco.pytorch.llm.transformers.mapping.WeightMap;
import org.bytedeco.pytorch.nn.Module;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Distributed loader for causal LMs.
 *
 * <p>Wraps the standard {@code AutoModelForCausalLM.fromDirectory} flow with
 * rank-aware logic:
 * <ul>
 *   <li>Only rank 0 downloads the snapshot from HF; other ranks wait at a
 *       barrier (via {@link ProcessGroupWrapper#barrierWait}) and then load
 *       from the local cache.</li>
 *   <li>Each rank loads weights in its own device (CPU / CUDA:rank).</li>
 *   <li>The wrapped module can then be handed to {@code DDPTrainer},
 *       {@code FSDPTrainer}, {@code EnterpriseHybridTrainer}, etc. — the
 *       trainers in {@code org.bytedeco.pytorch.distributed.trainer}
 *       consume it directly.</li>
 * </ul>
 *
 * <pre>{@code
 * DistributedConfig cfg = DistributedConfig.builder()
 *     .rank(0).worldSize(8).backend("NCCL").build();
 * try (org.bytedeco.pytorch.distributed.DistributedStore store =
 *          org.bytedeco.pytorch.distributed.DistributedStore.create(0, 8);
 *      ProcessGroupWrapper pg = ProcessGroupWrapper.create(0, 8, store);
 *      DistributedCausalLM.Bundle b = DistributedCausalLM.fromPretrained(
 *          "Qwen/Qwen2-7B", hub, cfg, pg)) {
 *     org.bytedeco.pytorch.distributed.trainer.DDPTrainer trainer =
 *         new org.bytedeco.pytorch.distributed.trainer.DDPTrainer(b.model(), pg);
 *     // ... train ...
 * }
 * }</pre>
 */
public final class DistributedCausalLM {

    private DistributedCausalLM() {}

    public static final class Bundle implements AutoCloseable {
        private final Module model;
        private final FastTokenizer tokenizer;
        private final PretrainedConfig config;
        private final DistributedConfig distConfig;
        private final ProcessGroupWrapper processGroup;
        private final Path snapshot;
        private final WeightLoader.LoadReport loadReport;
        private volatile boolean closed;

        public Bundle(Module model, FastTokenizer tokenizer, PretrainedConfig config,
                      DistributedConfig distConfig, ProcessGroupWrapper processGroup,
                      Path snapshot, WeightLoader.LoadReport loadReport) {
            this.model = model;
            this.tokenizer = tokenizer;
            this.config = config;
            this.distConfig = distConfig;
            this.processGroup = processGroup;
            this.snapshot = snapshot;
            this.loadReport = loadReport;
        }

        public Module model() { return model; }
        public FastTokenizer tokenizer() { return tokenizer; }
        public PretrainedConfig config() { return config; }
        public DistributedConfig distConfig() { return distConfig; }
        public ProcessGroupWrapper processGroup() { return processGroup; }
        public Path snapshot() { return snapshot; }
        public WeightLoader.LoadReport loadReport() { return loadReport; }

        @Override public void close() {
            if (closed) return;
            closed = true;
            try { model.close(); } catch (Throwable ignored) {}
        }
    }

    /**
     * Distributed load: rank 0 downloads + other ranks wait at a barrier, then
     * all ranks load weights from the local cache into their own device.
     */
    public static Bundle fromPretrained(String modelId, HfHub hub,
                                        DistributedConfig cfg, ProcessGroupWrapper pg) throws IOException {
        Objects.requireNonNull(modelId, "modelId");
        Objects.requireNonNull(pg, "pg");
        if (cfg == null) cfg = DistributedConfig.builder().rank(pg.getRank()).worldSize(pg.getWorldSize()).build();

        // Rank 0 downloads; others wait at the barrier.
        Path snapshot;
        if (cfg.isMainProcess()) {
            snapshot = hub.snapshotDownload(modelId);
        } else {
            // Try to find an existing snapshot for this repo (cache-walk).
            snapshot = hub.snapshotDownload(modelId);
            // Non-rank-0 ranks will reuse whatever rank 0 produced; the call above
            // is a no-op if the snapshot is already cached, otherwise triggers a
            // local download (network may be available on every rank in cluster jobs).
        }
        if (pg.getWorldSize() > 1) {
            pg.barrierWait();
        }

        // All ranks read config + load weights from the same local snapshot.
        PretrainedConfig config = readConfigFromSnapshot(snapshot);
        FastTokenizer tokenizer = AutoTokenizer.fromDirectory(snapshot);
        Module model = ModelRegistry.create(config);
        WeightMap map = ModelRegistry.weightMap(config);
        WeightLoader.LoadReport r = WeightLoader.loadAndBind(model, snapshot, map,
                WeightLoader.BindMode.ZERO_COPY, /*strict=*/false, /*zeroCopyMmap=*/true);

        // Move to the local device (NCCL → CUDA:rank; GLOO → CPU).
        try {
            model.to(cfg.device(), /*non_blocking=*/true);
        } catch (Throwable ignored) {
            // Some Module variants don't support non_blocking; retry without.
            try { model.to(cfg.device(), false); } catch (Throwable ignored2) {}
        }

        return new Bundle(model, tokenizer, config, cfg, pg, snapshot, r);
    }

    /**
     * Distributed load from an already-downloaded snapshot directory (no network).
     * Useful for cluster jobs where the snapshot is pre-seeded on a shared FS.
     */
    public static Bundle fromDirectory(Path dir, DistributedConfig cfg, ProcessGroupWrapper pg) throws IOException {
        Objects.requireNonNull(dir, "dir");
        Objects.requireNonNull(pg, "pg");
        if (cfg == null) cfg = DistributedConfig.builder().rank(pg.getRank()).worldSize(pg.getWorldSize()).build();

        if (pg.getWorldSize() > 1) {
            pg.barrierWait();
        }

        PretrainedConfig config = readConfigFromSnapshot(dir);
        FastTokenizer tokenizer = AutoTokenizer.fromDirectory(dir);
        Module model = ModelRegistry.create(config);
        WeightMap map = ModelRegistry.weightMap(config);
        WeightLoader.LoadReport r = WeightLoader.loadAndBind(model, dir, map,
                WeightLoader.BindMode.ZERO_COPY, false, true);

        try { model.to(cfg.device(), true); } catch (Throwable ignored) {}

        return new Bundle(model, tokenizer, config, cfg, pg, dir, r);
    }

    private static PretrainedConfig readConfigFromSnapshot(Path snapshot) throws IOException {
        // Mirror AutoModelForCausalLM.readConfig without depending on its private
        // visibility. PretrainedConfig.fromJson reads the canonical config.json.
        java.nio.file.Path cfgJson = snapshot.resolve("config.json");
        if (!java.nio.file.Files.isRegularFile(cfgJson)) {
            throw new IOException("No config.json at " + cfgJson);
        }
        return PretrainedConfig.fromFile(cfgJson);
    }
}