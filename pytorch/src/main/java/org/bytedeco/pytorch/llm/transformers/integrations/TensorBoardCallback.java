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
package org.bytedeco.pytorch.llm.transformers.integrations;

import java.nio.file.Path;

/**
 * TensorBoard integration callback.
 *
 * <p>Writes training metrics and model summaries to a TensorBoard log directory.
 *
 * <p>This is a stub. Full implementation requires the tensorboard Java API.
 */
public class TensorBoardCallback implements IntegrationCallback {

    private final Path logDir;
    private final String experimentName;

    public TensorBoardCallback(Path logDir) {
        this(logDir, null);
    }

    public TensorBoardCallback(Path logDir, String experimentName) {
        this.logDir = logDir;
        this.experimentName = experimentName;
    }

    @Override
    public void onTrainBegin(java.util.Map<String, Object> args) {
        // TODO: Initialize SummaryWriter with log_dir
        System.out.println("[TensorBoardCallback] onTrainBegin (stub) — logDir=" + logDir);
    }

    @Override
    public void onLog(int step, java.util.Map<String, Object> metrics) {
        // TODO: writer.add_scalar(tag, value, step) for each metric
        System.out.println("[TensorBoardCallback] step=" + step + " metrics=" + metrics + " (stub)");
    }

    @Override
    public void onTrainEnd(java.util.Map<String, Object> metrics) {
        // TODO: writer.close()
        System.out.println("[TensorBoardCallback] onTrainEnd (stub)");
    }
}
