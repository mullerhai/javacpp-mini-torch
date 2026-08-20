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
package org.bytedeco.pytorch.llm.transformers.hub;

import org.bytedeco.pytorch.llm.transformers.utils.Const;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Hub snapshot downloader. Wraps {@code huggingface_hub.snapshot_download}.
 *
 * <p>The default cache directory is {@code ${HF_HOME:-~/.cache/huggingface}/hub/models--owner--name/snapshots/<rev>}.
 *
 * <p>If the snapshot is already present locally, the function returns the
 * existing path without making any network calls.
 */
public final class SnapshotDownload {

    private SnapshotDownload() {}

    public static Path resolveLocal(HfApi api, String repoId, String revision) {
        Path root = cacheRoot();
        String sanitized = repoId.replace("/", "--");
        Path repoRoot = root.resolve("models--" + sanitized);
        Path snapRoot = repoRoot.resolve("snapshots");
        Path refPath = repoRoot.resolve("refs").resolve(revision);
        try {
            Files.createDirectories(snapRoot);
            Files.createDirectories(refPath.getParent());
        } catch (IOException ignored) {}
        Path chosen;
        if (Files.isRegularFile(refPath)) {
            try {
                String target = Files.readString(refPath).trim();
                chosen = snapRoot.resolve(target);
            } catch (IOException e) {
                chosen = snapRoot.resolve(revision);
            }
        } else {
            chosen = snapRoot.resolve(revision);
        }
        return chosen;
    }

    public static Path cacheRoot() {
        String hf = Const.get(Const.HF_HOME);
        if (hf == null || hf.isEmpty()) hf = System.getProperty("user.home") + "/" + Const.DEFAULT_HF_HOME;
        return Path.of(hf, "hub");
    }

    public static Path snapshot(HfApi api, String repoId, String revision, String repoType, List<String> allowPatterns) throws IOException {
        Objects.requireNonNull(api, "api");
        Objects.requireNonNull(repoId, "repoId");
        Path snapshotPath = resolveLocal(api, repoId, revision);
        if (Files.isDirectory(snapshotPath) && containsRequiredFiles(snapshotPath, allowPatterns)) {
            return snapshotPath;
        }
        // Lightweight downloader (URL fetch) — delegates to existing HfHub if present.
        try {
            Class<?> hubClass = Class.forName("org.bytedeco.pytorch.llm.hub.HfHub");
            Object hub = hubClass.getMethod("fromEnv").invoke(null);
            Object[] filterArr = allowPatterns.toArray();
            Path downloaded = (Path) hubClass.getMethod("snapshotDownload", String.class, String.class, String.class, java.util.Collection.class)
                    .invoke(hub, repoId, revision, repoType == null ? "models" : repoType, java.util.Arrays.asList(filterArr));
            return downloaded;
        } catch (ClassNotFoundException | NoSuchMethodException ignored) {
            // Fallback: caller must do its own download.
            if (Files.isDirectory(snapshotPath)) return snapshotPath;
            throw new IOException("No HfHub available and snapshot not local: " + repoId);
        } catch (Exception e) {
            throw new IOException("snapshot download failed: " + e.getMessage(), e);
        }
    }

    private static boolean containsRequiredFiles(Path dir, List<String> patterns) {
        if (patterns == null || patterns.isEmpty()) {
            try {
                if (!Files.isDirectory(dir)) return false;
                try (var stream = Files.list(dir)) {
                    return stream.findAny().isPresent();
                }
            } catch (IOException e) { return false; }
        }
        Set<String> required = new LinkedHashSet<>(patterns);
        for (String p : patterns) {
            if (!Files.isRegularFile(dir.resolve(p))) return false;
            required.remove(p);
        }
        return required.isEmpty();
    }
}