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
package org.bytedeco.pytorch.llm.peft;

/**
 * HuggingFace {@code PrefixTuningConfig}. Prefix past-key-value injection is
 * not welded in this drop; {@link PeftModel#getPeftModel} rejects it explicitly.
 */
public final class PrefixTuningConfig extends PeftConfig {

    private final int numVirtualTokens;
    private final int encoderHiddenSize;
    private final int prefixProjection; // 0/1

    private PrefixTuningConfig(Builder b) {
        super(b);
        this.numVirtualTokens = b.numVirtualTokens;
        this.encoderHiddenSize = b.encoderHiddenSize;
        this.prefixProjection = b.prefixProjection;
    }

    public int numVirtualTokens() { return numVirtualTokens; }
    public int encoderHiddenSize() { return encoderHiddenSize; }
    public boolean prefixProjection() { return prefixProjection != 0; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends PeftConfig.Builder<Builder> {
        private int numVirtualTokens = 20;
        private int encoderHiddenSize = 0;
        private int prefixProjection = 0;

        public Builder() { peftType(PeftType.PREFIX_TUNING); }

        public Builder numVirtualTokens(int v) { this.numVirtualTokens = v; return this; }
        public Builder num_virtual_tokens(int v) { return numVirtualTokens(v); }
        public Builder encoderHiddenSize(int v) { this.encoderHiddenSize = v; return this; }
        public Builder encoder_hidden_size(int v) { return encoderHiddenSize(v); }
        public Builder prefixProjection(boolean v) { this.prefixProjection = v ? 1 : 0; return this; }
        public Builder prefix_projection(boolean v) { return prefixProjection(v); }

        @Override
        public PrefixTuningConfig build() { return new PrefixTuningConfig(this); }
    }
}
