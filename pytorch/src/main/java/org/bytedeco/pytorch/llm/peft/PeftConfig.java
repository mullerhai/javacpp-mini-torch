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

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Base PEFT configuration (mirrors Hugging Face {@code PeftConfig}).
 *
 * <p>Includes the standard fields shared across all PEFT adapter types:
 * {@code peft_type}, {@code task_type}, {@code base_model_name_or_path},
 * {@code revision}, {@code inference_mode}, {@code peft_version},
 * and {@code auto_mapping}.
 */
public class PeftConfig {
    private final PeftType peftType;
    private final String taskType;
    private final String baseModelNameOrPath;
    private final String revision;
    private final boolean inferenceMode;
    private final String peftVersion;
    private final Map<String, Object> autoMapping;

    protected PeftConfig(Builder<?> b) {
        this.peftType = b.peftType;
        this.taskType = b.taskType;
        this.baseModelNameOrPath = b.baseModelNameOrPath;
        this.revision = b.revision;
        this.inferenceMode = b.inferenceMode;
        this.peftVersion = b.peftVersion;
        this.autoMapping = b.autoMapping;
    }

    public PeftType peftType() {
        return peftType;
    }

    public String taskType() {
        return taskType;
    }

    /** Parsed {@link TaskType}; unknown strings fall back to {@link TaskType#CAUSAL_LM}. */
    public TaskType taskTypeEnum() {
        try {
            return TaskType.fromString(taskType);
        } catch (IllegalArgumentException e) {
            return TaskType.CAUSAL_LM;
        }
    }

    public String baseModelNameOrPath() {
        return baseModelNameOrPath;
    }

    public String revision() {
        return revision;
    }

    public boolean inferenceMode() {
        return inferenceMode;
    }

    public String peftVersion() {
        return peftVersion;
    }

    public Map<String, Object> autoMapping() {
        return autoMapping == null ? null : new LinkedHashMap<>(autoMapping);
    }

    /** True if the configuration is for prompt learning (prefix / prompt tuning). */
    public boolean isPromptLearning() {
        return peftType == PeftType.PREFIX_TUNING || peftType == PeftType.PROMPT_TUNING;
    }

    /** Serialize this config to the standard {@code adapter_config.json} map. */
    public Map<String, Object> toDict() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("peft_type", peftType != null ? peftType.name() : null);
        m.put("task_type", taskType);
        m.put("base_model_name_or_path", baseModelNameOrPath);
        m.put("revision", revision);
        m.put("inference_mode", inferenceMode);
        if (peftVersion != null) m.put("peft_version", peftVersion);
        if (autoMapping != null) m.put("auto_mapping", autoMapping);
        return m;
    }

    @SuppressWarnings("unchecked")
    public static class Builder<B extends Builder<B>> {
        private PeftType peftType = PeftType.LORA;
        private String taskType = "CAUSAL_LM";
        private String baseModelNameOrPath = null;
        private String revision = null;
        private boolean inferenceMode = false;
        private String peftVersion = null;
        private Map<String, Object> autoMapping = null;

        public B peftType(PeftType peftType) {
            this.peftType = peftType;
            return (B) this;
        }

        public B taskType(String taskType) {
            this.taskType = taskType;
            return (B) this;
        }

        public B taskType(TaskType taskType) {
            this.taskType = taskType == null ? "CAUSAL_LM" : taskType.name();
            return (B) this;
        }

        /** Snake alias matching Python {@code task_type=}. */
        public B task_type(String taskType) {
            return taskType(taskType);
        }

        public B task_type(TaskType taskType) {
            return taskType(taskType);
        }

        public B baseModelNameOrPath(String v) {
            this.baseModelNameOrPath = v;
            return (B) this;
        }

        public B base_model_name_or_path(String v) {
            return baseModelNameOrPath(v);
        }

        public B revision(String v) {
            this.revision = v;
            return (B) this;
        }

        public B inferenceMode(boolean v) {
            this.inferenceMode = v;
            return (B) this;
        }

        public B inference_mode(boolean v) {
            return inferenceMode(v);
        }

        public B peftVersion(String v) {
            this.peftVersion = v;
            return (B) this;
        }

        public B peft_version(String v) {
            return peftVersion(v);
        }

        public B autoMapping(Map<String, Object> v) {
            this.autoMapping = v;
            return (B) this;
        }

        public B auto_mapping(Map<String, Object> v) {
            return autoMapping(v);
        }

        @SuppressWarnings("unchecked")
        public B self() {
            return (B) this;
        }

        public PeftConfig build() {
            return new PeftConfig(this);
        }
    }

    public static Builder<?> builder() {
        return new Builder<>();
    }

    /**
     * Best-effort parse of HuggingFace {@code adapter_config.json} (already decoded
     * to a map). Subclasses overlay their own fields after calling this.
     */
    @SuppressWarnings("unchecked")
    public static void applyBaseDict(Builder<?> b, Map<String, Object> d) {
        if (b == null || d == null) return;
        Object pt = d.get("peft_type");
        if (pt != null) {
            try { b.peftType(PeftType.valueOf(String.valueOf(pt).toUpperCase())); }
            catch (Exception ignored) {}
        }
        Object tt = d.get("task_type");
        if (tt != null) b.taskType(String.valueOf(tt));
        Object base = d.get("base_model_name_or_path");
        if (base != null && !"null".equals(String.valueOf(base))) {
            b.baseModelNameOrPath(String.valueOf(base));
        }
        Object rev = d.get("revision");
        if (rev != null && !"null".equals(String.valueOf(rev))) b.revision(String.valueOf(rev));
        Object inf = d.get("inference_mode");
        if (inf instanceof Boolean) b.inferenceMode((Boolean) inf);
        Object ver = d.get("peft_version");
        if (ver != null) b.peftVersion(String.valueOf(ver));
        Object am = d.get("auto_mapping");
        if (am instanceof Map<?, ?> m) {
            b.autoMapping(new LinkedHashMap<>((Map<String, Object>) m));
        }
    }
}