/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "Classpath" exception),
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

import java.util.List;
import java.util.Map;

/**
 * Utility to strip unused columns from dataset features.
 *
 * <p>This is a standalone re-export of the helper from {@link dataset_utils}
 * for callers that prefer a direct import.
 */
public final class remove_unused_columns {

    private remove_unused_columns() {}

    /**
     * Return a new list of feature maps that only retain columns used by the model.
     *
     * @param features  input batch features
     * @param usedNames columns the model forward pass actually consumes
     * @return filtered feature maps
     */
    public static List<Map<String, Object>> apply(
            List<Map<String, Object>> features, List<String> usedNames) {
        return dataset_utils.remove_unused_columns(features, usedNames);
    }
}
