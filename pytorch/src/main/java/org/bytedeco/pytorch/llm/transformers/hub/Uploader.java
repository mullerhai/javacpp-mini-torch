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
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Minimal uploader that pushes a folder or single file to a HF repo. Mirrors
 * {@code huggingface_hub.HfApi.upload_folder} / {@code upload_file}.
 *
 * <p>The implementation is intentionally lightweight — for real uploads,
 * plumb through {@code huggingface_hub.HfApi.upload_folder} via HTTP multipart.
 * In tests, this class is normally mocked.
 */
public final class Uploader {

    private static final Logger LOG = Logger.getLogger(Uploader.class.getName());

    private Uploader() {}

    public static void uploadFolder(HfApi api, Path folder, String repoId, String repoType, String commitMessage, String token) throws IOException {
        Objects.requireNonNull(folder, "folder");
        if (!Files.isDirectory(folder)) throw new IOException("Not a directory: " + folder);
        if (!api.isRepoIdValid(repoId)) throw new IOException("Invalid repo id: " + repoId);
        String tokenToUse = token != null ? token : Const.get(Const.HF_TOKEN);
        // In sandbox: only simulate — log each file that would be uploaded.
        try (var stream = Files.list(folder)) {
            stream.filter(Files::isRegularFile).forEach(f ->
                    LOG.log(Level.INFO, "[upload] would upload {0} to {1}/{2}/{3}",
                            new Object[] { f.getFileName(), repoType, repoId, repoType }));
        }
        LOG.log(Level.INFO, "[upload] commit message: {0}", commitMessage);
        LOG.log(Level.INFO, "[upload] token in use: {0}", tokenToUse == null ? "<none>" : "<set>");
    }

    public static void uploadFile(HfApi api, Path file, String pathInRepo, String repoId, String repoType, String commitMessage) throws IOException {
        Objects.requireNonNull(file, "file");
        if (!Files.isRegularFile(file)) throw new IOException("Not a regular file: " + file);
        if (!api.isRepoIdValid(repoId)) throw new IOException("Invalid repo id: " + repoId);
        try (InputStream is = Files.newInputStream(file)) {
            URL url = URI.create(api.endpointResolve("/" + repoType + "/" + repoId + "/resolve/main/" + pathInRepo)).toURL();
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("HEAD");
            conn.setConnectTimeout(Const.getInt(Const.HF_HUB_DOWNLOAD_TIMEOUT, 30) * 1000);
            int rc = conn.getResponseCode();
            LOG.log(Level.INFO, "[upload] HEAD {0} returned {1}", new Object[] { url, rc });
        }
    }
}