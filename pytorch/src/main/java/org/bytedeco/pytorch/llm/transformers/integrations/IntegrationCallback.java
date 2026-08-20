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
 * Marker interface for Trainer callbacks that receive integration lifecycle events.
 *
 * <p>Callbacks can hook into training events such as:
 * <ul>
 *   <li>on train begin / end</li>
 *   <li>on log (step metrics)</li>
 *   <li>on evaluation (eval results)</li>
 *   <li>on save (checkpoint saved)</li>
 * </ul>
 *
 * <p>Implementations include WandbCallback, TensorBoardCallback, MLflowCallback, etc.
 */
public interface IntegrationCallback {

    /** Called once at the start of training. */
    default void onTrainBegin(java.util.Map<String, Object> args) {}

    /** Called once at the end of training. */
    default void onTrainEnd(java.util.Map<String, Object> metrics) {}

    /** Called during training with step-level metrics. */
    default void onLog(int step, java.util.Map<String, Object> metrics) {}

    /** Called after each evaluation. */
    default void onEvaluate(int step, java.util.Map<String, Object> metrics) {}

    /** Called after saving a checkpoint. */
    default void onSave(int step, String checkpointPath) {}
}
