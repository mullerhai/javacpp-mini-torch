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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;

/**
 * Reads / writes the Hugging Face token to {@code ~/.huggingface/token}.
 * Mirrors {@code huggingface_hub.HfFolder}.
 */
public final class HfFolder {

    private HfFolder() {}

    public static Path tokenFile() {
        String home = System.getProperty("user.home");
        return Path.of(home, ".huggingface", "token");
    }

    public static String getToken() {
        // 1) env var wins (matches HF behaviour)
        String env = System.getenv("HF_TOKEN");
        if (env != null && !env.isEmpty()) return env;
        Path f = tokenFile();
        if (!Files.isRegularFile(f)) return null;
        try {
            String t = Files.readString(f).trim();
            return t.isEmpty() ? null : t;
        } catch (IOException e) {
            return null;
        }
    }

    public static void saveToken(String token) throws IOException {
        Objects.requireNonNull(token, "token");
        Path f = tokenFile();
        Files.createDirectories(f.getParent());
        Files.writeString(f, token);
    }

    public static void deleteToken() throws IOException {
        Path f = tokenFile();
        Files.deleteIfExists(f);
    }

    public static String whoami() {
        String token = getToken();
        if (token == null) return "<anonymous>";
        String masked = token.length() <= 6 ? token : token.substring(0, 3) + "***" + token.substring(token.length() - 3);
        return String.format(Locale.ROOT, "<authenticated: %s>", masked);
    }
}