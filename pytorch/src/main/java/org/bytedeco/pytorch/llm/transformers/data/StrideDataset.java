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
package org.bytedeco.pytorch.llm.transformers.data;

import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Sliding-window dataset — emits overlapping windows over sequence-like examples.
 *
 * <p>Useful for language modeling, time-series, or any task where
 * contiguous subsequences of a larger sequence are training targets.
 */
public class StrideDataset extends Dataset {

    private final List<Map<String, Object>> windows;
    private final int windowSize;
    private final int stride;

    /**
     * Create a sliding-window dataset from a parent dataset.
     *
     * @param parent      the underlying dataset (each example must have a "tokens" or "input_ids" field)
     * @param windowSize  length of each window
     * @param stride      step size between consecutive windows
     */
    public StrideDataset(Dataset parent, int windowSize, int stride) {
        if (windowSize <= 0) throw new IllegalArgumentException("windowSize must be positive");
        if (stride <= 0) throw new IllegalArgumentException("stride must be positive");
        this.windowSize = windowSize;
        this.stride = stride;

        List<Map<String, Object>> built = new ArrayList<>();
        for (Map<String, Object> example : parent) {
            // Try to extract a token/ids array
            Object ids = example.get("tokens");
            if (ids == null) ids = example.get("input_ids");
            if (ids instanceof long[] arr) {
                sliceLongArray(built, arr);
            } else if (ids instanceof int[] arr) {
                sliceIntArray(built, arr);
            } else if (ids instanceof short[] arr) {
                sliceShortArray(built, arr);
            } else if (ids instanceof byte[] arr) {
                sliceByteArray(built, arr);
            }
        }
        this.windows = built;
    }

    private void sliceLongArray(List<Map<String, Object>> out, long[] arr) {
        for (int start = 0; start + windowSize <= arr.length; start += stride) {
            long[] window = Arrays.copyOfRange(arr, start, start + windowSize);
            out.add(Collections.singletonMap("input_ids", window));
        }
    }

    private void sliceIntArray(List<Map<String, Object>> out, int[] arr) {
        for (int start = 0; start + windowSize <= arr.length; start += stride) {
            int[] window = Arrays.copyOfRange(arr, start, start + windowSize);
            out.add(Collections.singletonMap("input_ids", window));
        }
    }

    private void sliceShortArray(List<Map<String, Object>> out, short[] arr) {
        for (int start = 0; start + windowSize <= arr.length; start += stride) {
            short[] window = Arrays.copyOfRange(arr, start, start + windowSize);
            out.add(Collections.singletonMap("input_ids", window));
        }
    }

    private void sliceByteArray(List<Map<String, Object>> out, byte[] arr) {
        for (int start = 0; start + windowSize <= arr.length; start += stride) {
            byte[] window = Arrays.copyOfRange(arr, start, start + windowSize);
            out.add(Collections.singletonMap("input_ids", window));
        }
    }

    @Override
    public int size() { return windows.size(); }

    @Override
    public Map<String, Object> get(int index) {
        if (index < 0 || index >= windows.size()) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + windows.size());
        }
        return Collections.unmodifiableMap(windows.get(index));
    }

    @Override
    public Dataset map(Function<Map<String, Object>, Map<String, Object>> fn) {
        List<Map<String, Object>> mapped = new ArrayList<>(windows.size());
        for (Map<String, Object> w : windows) mapped.add(fn.apply(w));
        return new MapDataset(mapped);
    }

    @Override
    public Dataset filter(Predicate<Map<String, Object>> predicate) {
        List<Map<String, Object>> filtered = new ArrayList<>();
        for (Map<String, Object> w : windows) {
            if (predicate.test(w)) filtered.add(w);
        }
        return new MapDataset(filtered);
    }

    @Override
    public Dataset shuffle(long seed) {
        List<Map<String, Object>> copy = new ArrayList<>(windows);
        Collections.shuffle(copy, new Random(seed));
        return new MapDataset(copy);
    }

    @Override
    public Dataset select(int[] indices) {
        List<Map<String, Object>> selected = new ArrayList<>();
        for (int idx : indices) {
            if (idx >= 0 && idx < windows.size()) selected.add(windows.get(idx));
        }
        return new MapDataset(selected);
    }
}
