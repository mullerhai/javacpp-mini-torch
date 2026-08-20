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

/**
 * Aim callback — tracks ML experiments with a self-hosted UI.
 *
 * <p>Reports training metrics to an Aim server for visualization.
 *
 * <p>This is a stub. Full implementation requires an Aim Java client.
 */
public class AimCallback implements IntegrationCallback {

    private final String repo;
    private final String experiment;

    public AimCallback() {
        this(null, null);
    }

    public AimCallback(String repo, String experiment) {
        this.repo = repo;
        this.experiment = experiment;
    }

    @Override
    public void onTrainBegin(java.util.Map<String, Object> args) {
        // TODO: Initialize Aim repo and experiment
        System.out.println("[AimCallback] onTrainBegin (stub)");
    }

    @Override
    public void onLog(int step, java.util.Map<String, Object> metrics) {
        // TODO: aim_repo.track(metrics, step=step)
        System.out.println("[AimCallback] step=" + step + " metrics=" + metrics + " (stub)");
    }

    @Override
    public void onTrainEnd(java.util.Map<String, Object> metrics) {
        System.out.println("[AimCallback] onTrainEnd (stub)");
    }
}
