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

import org.bytedeco.pytorch.nn.Module;

import java.util.Map;

/**
 * Helper for running torch.compile benchmarks in tests.
 *
 * <p>This is a stub — full implementation depends on
 * the torch::compile binding being generated.
 */
public final class _torch_compile_helper {

    private _torch_compile_helper() {} // static utility

    /**
     * Compile a module with torch.compile (inductor backend).
     *
     * @param model    the model to compile
     * @param mode     compile mode: "default", "reduce-overhead", "max-autotune"
     * @param backend  backend: "inductor" (default), "eager"
     * @return compiled module
     */
    public static Module compile(Module model, String mode, String backend) {
        // TODO: Replace with real torch.compile binding
        // return torch.compile(model, mode, backend);
        System.out.println("[_torch_compile_helper] compile(mode=" + mode
                + ", backend=" + backend + ") (stub — returning original module)");
        return model;
    }

    /**
     * Compile with default settings (inductor, default mode).
     */
    public static Module compile(Module model) {
        return compile(model, "default", "inductor");
    }
}
