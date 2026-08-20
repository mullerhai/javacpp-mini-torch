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
package org.bytedeco.pytorch.llm.transformers.safetensors_utils;

import org.bytedeco.pytorch.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Utilities for reading and writing safetensors checkpoint files.
 *
 * <p>Safetensors format: a header (8 bytes: BigEndian u64 length) followed by
 * JSON header, followed by binary tensor data.
 */
public final class SafeTensorsUtils {

    private SafeTensorsUtils() {} // static utility

    /**
     * Detect if a file is a safetensors file by magic bytes.
     *
     * @param p path to the file
     * @return true if the file starts with the safetensors magic header
     */
    public static boolean isSafetensorsFile(Path p) {
        if (p == null || !Files.isRegularFile(p)) return false;
        try {
            byte[] header = new byte[8];
            try (var in = Files.newInputStream(p)) {
                int read = in.read(header);
                if (read < 8) return false;
            }
            // Magic: 8 bytes: [0x00, 0x73, 0x66, 0x74, 0x00, 0x00, 0x00, 0x00]
            // or end-of-Apache-arrow marker followed by { }
            // Standard safetensors magic:
            return header[0] == 0x00
                    && header[1] == 0x73  // 's'
                    && header[2] == 0x66  // 'f'
                    && header[3] == 0x74; // 't'
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Load all tensors from a single safetensors file.
     *
     * @param p path to the safetensors file
     * @return map from tensor name to tensor data
     * @throws IOException if the file cannot be read
     */
    public static Map<String, Tensor> loadSingleFile(Path p) throws IOException {
        if (!isSafetensorsFile(p)) {
            throw new IOException("Not a safetensors file: " + p);
        }
        Map<String, Tensor> result = new LinkedHashMap<>();
        // TODO: Replace with real safetensors parser (struct header + numpy/byte tensors).
        // Until then, throw an informative stub error.
        throw new UnsupportedOperationException(
                "SafeTensorsUtils.loadSingleFile is a stub. "
              + "Implement using a safetensors Java parser for: " + p);
    }

    /**
     * Load all tensors from a directory of safetensors shards.
     *
     * @param dir    directory containing shard files (model-00001-of-00003.safetensors, etc.)
     * @param type   FULL or SHARDED
     * @return map from tensor name to tensor data
     * @throws IOException on read errors
     */
    public static Map<String, Tensor> loadSharded(Path dir, StateDictType type) throws IOException {
        if (type != StateDictType.SHARDED) {
            throw new IllegalArgumentException("loadSharded requires SHARDED type");
        }
        // TODO: Implement shard discovery (sorted by filename) and load in order.
        throw new UnsupportedOperationException("SafeTensorsUtils.loadSharded is a stub");
    }
}
