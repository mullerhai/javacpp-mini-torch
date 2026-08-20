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
package org.bytedeco.pytorch.llm.transformers.utils;

/**
 * Environment variable constants matching Python {@code transformers.utils.Const}
 * and {@code huggingface_hub.constants}.
 *
 * <p>Each entry mirrors the canonical Python name. Defaults match
 * {@code huggingface_hub} 0.x:
 * <ul>
 *   <li>{@link #HF_HOME} = {@code ~/.cache/huggingface}</li>
 *   <li>{@link #TRANSFORMERS_CACHE} = {@code HF_HOME/hub}</li>
 * </ul>
 *
 * <p>Use {@link #get(String)}, {@link #get(String, String)} or {@link #require(String)}
 * to read the values.
 */
public final class Const {

    private Const() {}

    // Hub side (mirrors huggingface_hub.constants)
    public static final String HF_HOME = "HF_HOME";
    public static final String HF_TOKEN = "HF_TOKEN";
    public static final String HF_HUB_OFFLINE = "HF_HUB_OFFLINE";
    public static final String HF_HUB_DISABLE_PROGRESS_BARS = "HF_HUB_DISABLE_PROGRESS_BARS";
    public static final String HF_HUB_DISABLE_TELEMETRY = "HF_HUB_DISABLE_TELEMETRY";
    public static final String HF_HUB_DOWNLOAD_TIMEOUT = "HF_HUB_DOWNLOAD_TIMEOUT";
    public static final String HF_HUB_ENABLE_HF_TRANSFER = "HF_HUB_ENABLE_HF_TRANSFER";
    public static final String HF_HUB_ETAG_TIMEOUT = "HF_HUB_ETAG_TIMEOUT";
    public static final String HF_HUB_DISABLE_IMPLICIT_TOKEN = "HF_HUB_DISABLE_IMPLICIT_TOKEN";
    public static final String HF_HUB_DISABLE_REMOTE_CODE = "HF_HUB_DISABLE_REMOTE_CODE";
    public static final String HF_ENDPOINT = "HF_ENDPOINT";
    public static final String HUGGINGFACE_HUB_CACHE = "HUGGINGFACE_HUB_CACHE";

    // transformers side
    public static final String TRANSFORMERS_CACHE = "TRANSFORMERS_CACHE";
    public static final String TRANSFORMERS_OFFLINE = "TRANSFORMERS_OFFLINE";
    public static final String TRANSFORMERS_VERBOSITY = "TRANSFORMERS_VERBOSITY";
    public static final String TRANSFORMERS_NO_ADVISORY_WARNINGS = "TRANSFORMERS_NO_ADVISORY_WARNINGS";
    public static final String TOKENIZERS_PARALLELISM = "TOKENIZERS_PARALLELISM";
    public static final String HF_DATASETS_CACHE = "HF_DATASETS_CACHE";
    public static final String HF_DATASETS_OFFLINE = "HF_DATASETS_OFFLINE";
    public static final String HF_METRICS_CACHE = "HF_METRICS_CACHE";
    public static final String TF_CPP_MIN_LOG_LEVEL = "TF_CPP_MIN_LOG_LEVEL";

    // Torch side
    public static final String OMP_NUM_THREADS = "OMP_NUM_THREADS";
    public static final String MKL_NUM_THREADS = "MKL_NUM_THREADS";
    public static final String CUDA_LAUNCH_BLOCKING = "CUDA_LAUNCH_BLOCKING";
    public static final String TORCH_HOME = "TORCH_HOME";
    public static final String TORCH_USE_CUDA_DSA = "TORCH_USE_CUDA_DSA";

    // Defaults
    public static final String DEFAULT_HF_HOME = ".cache/huggingface";
    public static final String DEFAULT_HUB_ENDPOINT = "https://huggingface.co";

    /** Read env or return {@code null}. */
    public static String get(String key) {
        return System.getenv(key);
    }

    /** Read env or return default. */
    public static String get(String key, String defaultValue) {
        String v = System.getenv(key);
        return v == null || v.isEmpty() ? defaultValue : v;
    }

    /** Read env or throw IllegalStateException. */
    public static String require(String key) {
        String v = System.getenv(key);
        if (v == null || v.isEmpty()) {
            throw new IllegalStateException("Missing required environment variable: " + key);
        }
        return v;
    }

    /** Parse boolean env var ("1", "true", "yes" → true; "0", "false", "no" → false). */
    public static boolean getBool(String key, boolean defaultValue) {
        String v = System.getenv(key);
        if (v == null || v.isEmpty()) return defaultValue;
        v = v.trim().toLowerCase();
        if (v.equals("1") || v.equals("true") || v.equals("yes") || v.equals("on")) return true;
        if (v.equals("0") || v.equals("false") || v.equals("no") || v.equals("off")) return false;
        return defaultValue;
    }

    /** Parse integer env var or return default. */
    public static int getInt(String key, int defaultValue) {
        String v = System.getenv(key);
        if (v == null || v.isEmpty()) return defaultValue;
        try { return Integer.parseInt(v.trim()); } catch (NumberFormatException e) { return defaultValue; }
    }

    /** True when hub is forced offline by env vars. */
    public static boolean isOffline() {
        return getBool(HF_HUB_OFFLINE, false) || getBool(TRANSFORMERS_OFFLINE, false);
    }

    /** True when telemetry is suppressed. */
    public static boolean isTelemetryDisabled() {
        return getBool(HF_HUB_DISABLE_TELEMETRY, false);
    }
}