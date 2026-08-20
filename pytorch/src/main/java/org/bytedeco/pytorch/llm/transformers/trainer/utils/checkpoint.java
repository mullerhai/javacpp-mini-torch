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

import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Checkpoint utilities mirroring Python's {@code transformers.trainer_utils.checkpoint_utils}.
 *
 * <p>Provides {@link #get_last_checkpoint(Path)} to locate the most recent
 * checkpoint directory within an output folder.
 */
public final class checkpoint {

    private checkpoint() {}

    /**
     * Return the path to the most recently saved checkpoint under {@code dir},
     * or {@code null} if no checkpoint is found.
     *
     * <p>Searches for directories matching {@code checkpoint-*\d*} or
     * returns the highest numbered checkpoint.
     *
     * @param dir output / checkpoint root directory
     * @return path to the last checkpoint, or null
     */
    public static Path get_last_checkpoint(Path dir) {
        if (dir == null || !Files.isDirectory(dir)) return null;

        Path last = null;
        int bestStep = -1;

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path entry : stream) {
                if (!Files.isDirectory(entry)) continue;
                String name = entry.getFileName().toString();
                if (!name.startsWith("checkpoint")) continue;
                int dashIdx = name.lastIndexOf('-');
                if (dashIdx < 0) continue;
                try {
                    int step = Integer.parseInt(name.substring(dashIdx + 1));
                    if (step > bestStep) {
                        bestStep = step;
                        last = entry;
                    }
                } catch (NumberFormatException ignored) {}
            }
        } catch (java.io.IOException ignored) {}

        return last;
    }

    /**
     * List all checkpoint directories under {@code dir}.
     *
     * @param dir output root
     * @return sorted list of checkpoint paths (newest first)
     */
    public static java.util.List<Path> list_checkpoints(Path dir) {
        java.util.List<Path> checkpoints = new java.util.ArrayList<>();
        Path last = get_last_checkpoint(dir);
        if (last != null) checkpoints.add(last);
        return checkpoints;
    }
}
