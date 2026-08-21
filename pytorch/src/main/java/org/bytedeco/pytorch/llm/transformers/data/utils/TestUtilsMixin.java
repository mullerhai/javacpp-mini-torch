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
package org.bytedeco.pytorch.llm.transformers.data.utils;

/**
 * Utility mixin for writing assertions in tests.
 */
public final class TestUtilsMixin {

    private TestUtilsMixin() {} // static utility

    /**
     * Assert that two objects are equal, throwing AssertionError if not.
     *
     * @param a first object
     * @param b second object
     * @return true if equal (for convenience in one-liners)
     * @throws AssertionError if {@code a.equals(b)} is false
     */
    public static boolean assertEqual(Object a, Object b) {
        if (a == null ? b != null : !a.equals(b)) {
            throw new AssertionError(
                    "Objects not equal:\n  expected: " + a + "\n  actual:   " + b);
        }
        return true;
    }

    /**
     * Assert that two tensors have matching shapes and all-close values.
     *
     * @param a first tensor
     * @param b second tensor
     * @param rtol relative tolerance for comparison
     * @param atol absolute tolerance for comparison
     * @throws AssertionError if tensors do not match
     */
    public static void assertTensorEqual(
            org.bytedeco.pytorch.Tensor a,
            org.bytedeco.pytorch.Tensor b,
            double rtol, double atol) {
        if (a == null || b == null) {
            throw new AssertionError("One of the tensors is null: a=" + a + ", b=" + b);
        }
        if (!a.defined() || !b.defined()) {
            throw new AssertionError("One of the tensors is undefined");
        }
        // Use torch.allclose under the hood
        boolean close = org.bytedeco.pytorch.global.torch.allclose(a, b, rtol, atol, false);
        if (!close) {
            throw new AssertionError(
                    "Tensors not close (rtol=" + rtol + ", atol=" + atol + "):\n  a=" + a + "\n  b=" + b);
        }
    }

    /**
     * Assert that two tensors have matching shapes and all-close values.
     * Uses default tolerances (rtol=1e-5, atol=1e-8).
     */
    public static void assertTensorEqual(
            org.bytedeco.pytorch.Tensor a,
            org.bytedeco.pytorch.Tensor b) {
        assertTensorEqual(a, b, 1e-5, 1e-8);
    }
}
