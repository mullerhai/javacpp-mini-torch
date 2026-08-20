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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Java clone of Python {@code huggingface_hub.HfApi}. Mirrors the high-level
 * operations used by Transformers tutorials:
 * <ul>
 *   <li>repo id parsing ({@code "user-or-org/repo-name"})</li>
 *   <li>file listing via local cache inspection</li>
 *   <li>snapshot download (delegates to {@link SnapshotDownload})</li>
 *   <li>upload folder / file (delegates to {@link Uploader})</li>
 *   <li>create / delete repo (delegates to {@link RepoClient})</li>
 * </ul>
 *
 * <p>Network calls are funneled through {@link SnapshotDownload} and
 * {@link Uploader} so the rest of the codebase can mock them in tests.
 */
public final class HfApi {

    private static final Pattern REPO_ID = Pattern.compile("^([\\w\\-\\.\\+]+)/([\\w\\-\\.\\+]+)$");
    private static final Map<String, HfApi> INSTANCES = new ConcurrentHashMap<>();

    private final String endpoint;
    private final String token;

    private HfApi(String endpoint, String token) {
        this.endpoint = Objects.requireNonNullElse(endpoint, Const.DEFAULT_HUB_ENDPOINT);
        this.token = token;
    }

    public static HfApi fromEnv() {
        return get(Const.get(Const.HF_ENDPOINT, Const.DEFAULT_HUB_ENDPOINT), Const.get(Const.HF_TOKEN));
    }

    public static HfApi get() {
        return fromEnv();
    }

    public static HfApi get(String endpoint) {
        return new HfApi(endpoint, Const.get(Const.HF_TOKEN));
    }

    public static HfApi get(String endpoint, String token) {
        return new HfApi(endpoint, token);
    }

    public String endpoint() { return endpoint; }
    public String token() { return token; }

    public RepoId parseRepoId(String repoId) {
        Matcher m = REPO_ID.matcher(repoId);
        if (!m.matches()) {
            throw new IllegalArgumentException("Not a valid HF repo id: " + repoId);
        }
        return new RepoId(m.group(1), m.group(2));
    }

    public boolean isRepoIdValid(String repoId) {
        return REPO_ID.matcher(repoId).matches();
    }

    public String endpointResolve(String fullUrl) {
        if (fullUrl == null) return endpoint;
        if (fullUrl.startsWith("http://") || fullUrl.startsWith("https://")) return fullUrl;
        if (fullUrl.startsWith("/")) return endpoint + fullUrl;
        return endpoint + "/" + fullUrl;
    }

    public List<String> listRepoFiles(String repoId, String revision) throws IOException {
        return RepoClient.listFiles(this, repoId, revision == null ? "main" : revision);
    }

    public List<String> listRepoFiles(String repoId) throws IOException {
        return listRepoFiles(repoId, "main");
    }

    public Path snapshotDownload(String repoId, String revision, String repoType, List<String> allowPatterns) throws IOException {
        return SnapshotDownload.snapshot(this, repoId, revision == null ? "main" : revision,
                repoType == null ? "model" : repoType, allowPatterns == null ? List.of() : allowPatterns);
    }

    public Path snapshotDownload(String repoId) throws IOException {
        return snapshotDownload(repoId, "main", "model", List.of());
    }

    public Path snapshotDownload(String repoId, String revision) throws IOException {
        return snapshotDownload(repoId, revision, "model", List.of());
    }

    public void createRepo(String repoId, String repoType, boolean existOk, boolean privateRepo) throws IOException {
        RepoClient.create(this, repoId, repoType, existOk, privateRepo);
    }

    public void createRepo(String repoId) throws IOException {
        createRepo(repoId, "model", false, false);
    }

    /** Keyword-style overload matching {@code HfApi.create_repo(repo_id=, token=, repo_type=)}. */
    public void createRepo(String repoId, String token, String repoType, boolean privateRepo) throws IOException {
        HfApi api = token == null ? this : new HfApi(this.endpoint, token);
        api.createRepo(repoId, repoType == null ? "model" : repoType, true, privateRepo);
    }

    public void create_repo(String repoId) throws IOException { createRepo(repoId); }

    public void uploadFolder(String folderPath, String repoId, String commitMessage) throws IOException {
        uploadFolder(folderPath, repoId, "model", commitMessage, null);
    }

    public void uploadFolder(String folderPath, String repoId) throws IOException {
        uploadFolder(folderPath, repoId, "model", "upload", this.token);
    }

    public void uploadFolder(String folderPath, String repoId, String token) throws IOException {
        uploadFolder(folderPath, repoId, "model", "upload", token);
    }

    public void upload_folder(String folderPath, String repoId) throws IOException {
        uploadFolder(folderPath, repoId);
    }

    public void deleteRepo(String repoId, String repoType) throws IOException {
        RepoClient.delete(this, repoId, repoType);
    }

    public void uploadFolder(String folderPath, String repoId, String repoType, String commitMessage, String token) throws IOException {
        Uploader.uploadFolder(this, Path.of(folderPath), repoId, repoType == null ? "model" : repoType,
                commitMessage == null ? "upload" : commitMessage, token);
    }

    public void uploadFile(String filePath, String pathInRepo, String repoId, String repoType, String commitMessage) throws IOException {
        Uploader.uploadFile(this, Path.of(filePath), pathInRepo, repoId,
                repoType == null ? "model" : repoType, commitMessage == null ? "upload" : commitMessage);
    }

    public void uploadFile(String filePath, String pathInRepo, String repoId) throws IOException {
        uploadFile(filePath, pathInRepo, repoId, "model", null);
    }

    public Path getLocalSnapshotPath(String repoId, String revision) {
        return SnapshotDownload.resolveLocal(this, repoId, revision == null ? "main" : revision);
    }

    public boolean snapshotExists(String repoId, String revision) {
        Path p = getLocalSnapshotPath(repoId, revision);
        return Files.isDirectory(p);
    }

    public Set<String> snapshotFiles(String repoId, String revision) throws IOException {
        Path p = getLocalSnapshotPath(repoId, revision);
        if (!Files.isDirectory(p)) return Set.of();
        Set<String> out = new LinkedHashSet<>();
        try (var stream = Files.list(p)) {
            stream.filter(Files::isRegularFile).forEach(child -> out.add(child.getFileName().toString()));
        }
        return out;
    }

    public String repoInfo(String repoId) {
        RepoId r = parseRepoId(repoId);
        return String.format(Locale.ROOT, "RepoId{owner=%s, name=%s, endpoint=%s}", r.owner(), r.name(), endpoint);
    }

    /** Helper for the common "Microsoft/Phi-3-mini-4k-instruct" parse. */
    public static String normalizeRepoId(String repoId) {
        if (repoId == null) return null;
        if (REPO_ID.matcher(repoId).matches()) return repoId;
        // Try to fix common mistakes like "repo:user/name" or "models--user--name"
        if (repoId.startsWith("models--")) {
            String stripped = repoId.substring("models--".length());
            return stripped.replace("--", "/");
        }
        return repoId;
    }

    public static HfApi instance(String token) {
        return INSTANCES.computeIfAbsent(token == null ? "<no-token>" : token, k -> new HfApi(null, token));
    }

    /** Convenience: snapshot then filter files. */
    public List<String> ensureFiles(String repoId, String... patterns) throws IOException {
        Set<String> allow = new LinkedHashSet<>(Arrays.asList(patterns));
        return snapshotDownload(repoId, "main", "model", new ArrayList<>(allow))
                .toFile().exists() ? List.of(patterns) : List.of();
    }
}