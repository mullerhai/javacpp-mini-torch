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
package org.bytedeco.pytorch.llm.transformers.testing_utils;

/**
 * GPU memory limit helper for TensorFlow interop tests.
 *
 * <p>Provides a decorator that sets per-GPU memory growth limits when
 * running tests alongside a PyTorch backend.
 *
 * <p>This is a stub — the real implementation would call
 * {@code tf.config.experimental.set_memory_growth}.
 */
public final class _tf_gpu_memory_limit {

    private _tf_gpu_memory_limit() {} // static utility

    /**
     * Set per-GPU memory growth limit for TF tests.
     *
     * @param limitMB memory limit in MB per GPU, or 0 for allow_growth
     */
    public static void setLimit(long limitMB) {
        // TODO: tf.config.experimental.set_memory_growth(gpu, True)
        // TODO: tf.config.experimental.set_memory_limit(gpu, limitMB)
        System.out.println("[_tf_gpu_memory_limit] setLimit=" + limitMB + "MB (stub)");
    }

    /**
     * Reset memory growth settings.
     */
    public static void reset() {
        // TODO: reset GPU memory configuration
        System.out.println("[_tf_gpu_memory_limit] reset (stub)");
    }
}
