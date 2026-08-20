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
package org.bytedeco.pytorch.llm.transformers.trainer.utils;

import org.bytedeco.pytorch.llm.transformers.utils.Const;

/**
 * Environment-variable helpers mirroring Python's {@code transformers.trainer_utils.EvalTrainingType}.
 *
 * <p>Exposes HF cache / home paths and thread settings.
 */
public final class env {

    private env() {}

    /**
     * Return the TRANSFORMERS_CACHE directory, falling back to {@code HF_HOME/hub}.
     */
    public static String TRANSFORMERS_CACHE() {
        String c = Const.get(Const.TRANSFORMERS_CACHE);
        if (c != null && !c.isBlank()) return c;
        String hf = HF_HOME();
        return hf + "/hub";
    }

    /**
     * Return the HF_HOME directory, falling back to {@code ~/.cache/huggingface}.
     */
    public static String HF_HOME() {
        String hf = Const.get(Const.HF_HOME);
        if (hf != null && !hf.isBlank()) return hf;
        return System.getProperty("user.home") + "/.cache/huggingface";
    }

    /**
     * Return OMP_NUM_THREADS from the environment, or {@code Runtime.getRuntime().availableProcessors()}.
     */
    public static int OMP_NUM_THREADS() {
        String v = System.getenv("OMP_NUM_THREADS");
        if (v != null && !v.isBlank()) {
            try { return Integer.parseInt(v.trim()); } catch (NumberFormatException ignored) {}
        }
        return Runtime.getRuntime().availableProcessors();
    }

    /**
     * Return the default cache sub-directory for a given model name.
     */
    public static String default_cache_path(String modelName) {
        return TRANSFORMERS_CACHE() + "/models--" + modelName.replace("/", "--");
    }
}
