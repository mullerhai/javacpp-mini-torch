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

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Minimal repo lifecycle client. Mirrors {@code huggingface_hub.HfApi.create_repo}
 * / {@code delete_repo} / {@code list_repo_files}.
 *
 * <p>This is a stub: in a sandboxed environment no real network is performed.
 * Tests of {@link HfApi} should mock the operations through this interface.
 */
public final class RepoClient {

    private RepoClient() {}

    public static void create(HfApi api, String repoId, String repoType, boolean existOk, boolean privateRepo) throws IOException {
        Objects.requireNonNull(api, "api");
        if (!api.isRepoIdValid(repoId)) {
            throw new IOException("Invalid repo id: " + repoId);
        }
        // No-op: real upload/download flows go through HfHub / Uploader.
    }

    public static void delete(HfApi api, String repoId, String repoType) throws IOException {
        Objects.requireNonNull(api, "api");
        if (!api.isRepoIdValid(repoId)) {
            throw new IOException("Invalid repo id: " + repoId);
        }
    }

    public static List<String> listFiles(HfApi api, String repoId, String revision) throws IOException {
        Objects.requireNonNull(api, "api");
        if (!api.isRepoIdValid(repoId)) {
            throw new IOException("Invalid repo id: " + repoId);
        }
        Path p = api.getLocalSnapshotPath(repoId, revision);
        if (!java.nio.file.Files.isDirectory(p)) return List.of();
        try (var stream = java.nio.file.Files.list(p)) {
            return stream.filter(java.nio.file.Files::isRegularFile)
                    .map(java.nio.file.Path::getFileName)
                    .map(java.nio.file.Path::toString)
                    .sorted()
                    .toList();
        }
    }
}