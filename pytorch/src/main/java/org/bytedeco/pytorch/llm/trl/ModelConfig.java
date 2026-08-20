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
 * HuggingFace TRL {@code trl.trainer.model_config.ModelConfig}.
 *
 * <p>Only the fields used by the SFT / VLM tutorials are modelled.
 */
public final class ModelConfig {

    private final String modelNameOrPath;
    private final String modelRevision;
    private final String torchDtype;
    private final boolean trustRemoteCode;
    private final String attnImplementation;
    private final boolean usePeft;
    private final int loraR;
    private final double loraAlpha;
    private final double loraDropout;
    private final String loraTargetModules;
    private final boolean loadIn4Bit;
    private final boolean loadIn8Bit;

    private ModelConfig(Builder b) {
        this.modelNameOrPath = b.modelNameOrPath;
        this.modelRevision = b.modelRevision;
        this.torchDtype = b.torchDtype;
        this.trustRemoteCode = b.trustRemoteCode;
        this.attnImplementation = b.attnImplementation;
        this.usePeft = b.usePeft;
        this.loraR = b.loraR;
        this.loraAlpha = b.loraAlpha;
        this.loraDropout = b.loraDropout;
        this.loraTargetModules = b.loraTargetModules;
        this.loadIn4Bit = b.loadIn4Bit;
        this.loadIn8Bit = b.loadIn8Bit;
    }

    public static Builder builder() { return new Builder(); }

    public String modelNameOrPath() { return modelNameOrPath; }
    public String modelRevision() { return modelRevision; }
    public String torchDtype() { return torchDtype; }
    public boolean trustRemoteCode() { return trustRemoteCode; }
    public String attnImplementation() { return attnImplementation; }
    public boolean usePeft() { return usePeft; }
    public int loraR() { return loraR; }
    public double loraAlpha() { return loraAlpha; }
    public double loraDropout() { return loraDropout; }
    public String loraTargetModules() { return loraTargetModules; }
    public boolean loadIn4Bit() { return loadIn4Bit; }
    public boolean loadIn8Bit() { return loadIn8Bit; }

    public static final class Builder {
        private String modelNameOrPath;
        private String modelRevision = "main";
        private String torchDtype = "bfloat16";
        private boolean trustRemoteCode = false;
        private String attnImplementation = null;
        private boolean usePeft = false;
        private int loraR = 16;
        private double loraAlpha = 32;
        private double loraDropout = 0.05;
        private String loraTargetModules = "all-linear";
        private boolean loadIn4Bit = false;
        private boolean loadIn8Bit = false;

        public Builder modelNameOrPath(String v) { this.modelNameOrPath = v; return this; }
        public Builder model_name_or_path(String v) { return modelNameOrPath(v); }
        public Builder modelRevision(String v) { this.modelRevision = v; return this; }
        public Builder model_revision(String v) { return modelRevision(v); }
        public Builder torchDtype(String v) { this.torchDtype = v; return this; }
        public Builder torch_dtype(String v) { return torchDtype(v); }
        public Builder trustRemoteCode(boolean v) { this.trustRemoteCode = v; return this; }
        public Builder trust_remote_code(boolean v) { return trustRemoteCode(v); }
        public Builder attnImplementation(String v) { this.attnImplementation = v; return this; }
        public Builder attn_implementation(String v) { return attnImplementation(v); }
        public Builder usePeft(boolean v) { this.usePeft = v; return this; }
        public Builder use_peft(boolean v) { return usePeft(v); }
        public Builder loraR(int v) { this.loraR = v; return this; }
        public Builder lora_r(int v) { return loraR(v); }
        public Builder loraAlpha(double v) { this.loraAlpha = v; return this; }
        public Builder lora_alpha(double v) { return loraAlpha(v); }
        public Builder loraDropout(double v) { this.loraDropout = v; return this; }
        public Builder lora_dropout(double v) { return loraDropout(v); }
        public Builder loraTargetModules(String v) { this.loraTargetModules = v; return this; }
        public Builder lora_target_modules(String v) { return loraTargetModules(v); }
        public Builder loadIn4Bit(boolean v) { this.loadIn4Bit = v; return this; }
        public Builder load_in_4bit(boolean v) { return loadIn4Bit(v); }
        public Builder loadIn8Bit(boolean v) { this.loadIn8Bit = v; return this; }
        public Builder load_in_8bit(boolean v) { return loadIn8Bit(v); }

        public ModelConfig build() { return new ModelConfig(this); }
    }
}
