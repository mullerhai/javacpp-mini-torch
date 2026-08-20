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
package org.bytedeco.pytorch.llm.trl;

/**
 * HuggingFace TRL {@code trl.scripts.utils.ScriptArguments} (dataset CLI flags).
 */
public final class ScriptArguments {

    private final String datasetName;
    private final String datasetConfig;
    private final String datasetTrainSplit;
    private final String datasetTestSplit;

    private ScriptArguments(Builder b) {
        this.datasetName = b.datasetName;
        this.datasetConfig = b.datasetConfig;
        this.datasetTrainSplit = b.datasetTrainSplit;
        this.datasetTestSplit = b.datasetTestSplit;
    }

    public static Builder builder() { return new Builder(); }

    public String datasetName() { return datasetName; }
    public String datasetConfig() { return datasetConfig; }
    public String datasetTrainSplit() { return datasetTrainSplit; }
    public String datasetTestSplit() { return datasetTestSplit; }

    public static final class Builder {
        private String datasetName;
        private String datasetConfig;
        private String datasetTrainSplit = "train";
        private String datasetTestSplit = "test";

        public Builder datasetName(String v) { this.datasetName = v; return this; }
        public Builder dataset_name(String v) { return datasetName(v); }
        public Builder datasetConfig(String v) { this.datasetConfig = v; return this; }
        public Builder dataset_config(String v) { return datasetConfig(v); }
        public Builder datasetTrainSplit(String v) { this.datasetTrainSplit = v; return this; }
        public Builder dataset_train_split(String v) { return datasetTrainSplit(v); }
        public Builder datasetTestSplit(String v) { this.datasetTestSplit = v; return this; }
        public Builder dataset_test_split(String v) { return datasetTestSplit(v); }

        public ScriptArguments build() { return new ScriptArguments(this); }
    }
}
