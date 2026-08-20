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
 * HuggingFace {@code PromptTuningConfig} fields. Prompt-token injection into
 * the embedding stream is not welded in this drop — {@link PeftModel#getPeftModel}
 * rejects prompt-learning configs with a clear error rather than a silent no-op.
 */
public final class PromptTuningConfig extends PeftConfig {

    private final int numVirtualTokens;
    private final int tokenDim;
    private final String promptTuningInit; // TEXT | RANDOM
    private final String promptTuningInitText;
    private final String tokenizerNameOrPath;

    private PromptTuningConfig(Builder b) {
        super(b);
        this.numVirtualTokens = b.numVirtualTokens;
        this.tokenDim = b.tokenDim;
        this.promptTuningInit = b.promptTuningInit;
        this.promptTuningInitText = b.promptTuningInitText;
        this.tokenizerNameOrPath = b.tokenizerNameOrPath;
    }

    public int numVirtualTokens() { return numVirtualTokens; }
    public int tokenDim() { return tokenDim; }
    public String promptTuningInit() { return promptTuningInit; }
    public String promptTuningInitText() { return promptTuningInitText; }
    public String tokenizerNameOrPath() { return tokenizerNameOrPath; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends PeftConfig.Builder<Builder> {
        private int numVirtualTokens = 8;
        private int tokenDim = 0;
        private String promptTuningInit = "RANDOM";
        private String promptTuningInitText;
        private String tokenizerNameOrPath;

        public Builder() { peftType(PeftType.PROMPT_TUNING); }

        public Builder numVirtualTokens(int v) { this.numVirtualTokens = v; return this; }
        public Builder num_virtual_tokens(int v) { return numVirtualTokens(v); }
        public Builder tokenDim(int v) { this.tokenDim = v; return this; }
        public Builder token_dim(int v) { return tokenDim(v); }
        public Builder promptTuningInit(String v) { this.promptTuningInit = v; return this; }
        public Builder prompt_tuning_init(String v) { return promptTuningInit(v); }
        public Builder promptTuningInitText(String v) { this.promptTuningInitText = v; return this; }
        public Builder tokenizerNameOrPath(String v) { this.tokenizerNameOrPath = v; return this; }

        @Override
        public PromptTuningConfig build() { return new PromptTuningConfig(this); }
    }
}
