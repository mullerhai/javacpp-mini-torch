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
 * Utility helpers for device detection in tests.
 */
public final class _device_utils {

    private _device_utils() {} // static utility

    /**
     * Check whether CUDA (GPU) is available.
     *
     * @return true if at least one CUDA device is available
     */
    public static boolean isCuda() {
        return org.bytedeco.pytorch.global.torch.cuda_is_available();
    }

    /**
     * Check whether MPS (Apple Silicon GPU) is available.
     *
     * @return true if MPS backend is available
     */
    public static boolean isMps() {
        // TODO: Replace with torch.backends.mps.is_available() when available
        return false;
    }

    /**
     * Get the default device string (cuda / mps / cpu).
     *
     * @return "cuda:N", "mps", or "cpu"
     */
    public static String getDefaultDevice() {
        if (isCuda()) return "cuda";
        if (isMps()) return "mps";
        return "cpu";
    }

    /**
     * Return the CUDA device count.
     *
     * @return number of visible CUDA devices, or 0 if none
     */
    public static int cudaDeviceCount() {
        if (!isCuda()) return 0;
        return (int) org.bytedeco.pytorch.global.torch.cuda_device_count();
    }
}
