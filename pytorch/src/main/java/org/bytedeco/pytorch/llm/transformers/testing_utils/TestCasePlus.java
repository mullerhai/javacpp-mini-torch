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
 * Base class for tests with common environment detection helpers.
 *
 * <p>Provides convenience methods for skipping tests when
 * optional hardware or libraries are not available.
 */
public class TestCasePlus {

    /** Name of the CUDAavailable environment variable checked by HF test suite. */
    protected static final String ENV_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES";

    /**
     * Skip the current test if PyTorch was built without CUDA support.
     */
    protected void requireTorch() {
        if (!org.bytedeco.pytorch.global.torch.is_available()) {
            org.junit.jupiter.api.Assumptions.assumeTrue(
                    false, "PyTorch is not available");
        }
    }

    /**
     * Skip the current test if CUDA is not available.
     */
    protected void requireCUDA() {
        if (!org.bytedeco.pytorch.global.torch.cuda_is_available()) {
            org.junit.jupiter.api.Assumptions.assumeTrue(
                    false, "CUDA is not available");
        }
    }

    /**
     * Mark a test as slow — useful for CI to run fast tests first.
     */
    protected boolean slowTest() {
        return System.getProperty("RUN_SLOW_TESTS", "false").equalsIgnoreCase("true");
    }

    /**
     * Skip if the given command returns non-zero.
     */
    protected boolean commandExists(String cmd) {
        try {
            int exit = new ProcessBuilder("sh", "-c", "command -v " + cmd).start().waitFor();
            return exit == 0;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Get the path to a test resource file.
     */
    protected java.nio.file.Path getTestFile(String name) {
        return java.nio.file.Paths.get("src", "test", "resources", name);
    }

    /**
     * Create a temporary directory for test outputs.
     */
    protected java.nio.file.Path tempDir() throws java.io.IOException {
        return java.nio.file.Files.createTempDirectory("pytorch-test");
    }
}
