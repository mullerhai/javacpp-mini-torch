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
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.transformers.tokenization;

import org.bytedeco.pytorch.llm.transformers.processor.Processor;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Enhanced chat templates supporting multimodal inputs.
 *
 * <p>Extends the base ChatTemplate with:
 * <ul>
 *   <li>Multimodal message processing (images, videos, audio)</li>
 *   <li>Special token replacement for multimodal placeholders</li>
 *   <li>Chat template application with processor integration</li>
 *   <li>Support for Qwen2.5-Omni, Gemma4, and other omni-modal models</li>
 * </ul>
 *
 * <p>Reference: HuggingFace chat templates, Qwen2.5-Omni, Gemma4
 */
public final class MultimodalChatTemplate {

    public enum Flavor {
        QWEN,           // ChatML style
        QWEN_OMNI,      // Qwen2.5-Omni with audio support
        LLAMA3,         // Llama-3 style
        MISTRAL,        // Mistral/ChatML
        GLM,            // GLM style
        GEMMA,          // Gemma multimodal
        Gemma4,         // Gemma4 omni-modal
        RAW             // Concatenate only
    }

    private final Flavor flavor;
    private final String rawTemplate;

    public MultimodalChatTemplate(Flavor flavor) {
        this(flavor, null);
    }

    public MultimodalChatTemplate(Flavor flavor, String rawTemplate) {
        this.flavor = flavor == null ? Flavor.RAW : flavor;
        this.rawTemplate = rawTemplate;
    }

    public static MultimodalChatTemplate qwen() { return new MultimodalChatTemplate(Flavor.QWEN); }
    public static MultimodalChatTemplate qwenOmni() { return new MultimodalChatTemplate(Flavor.QWEN_OMNI); }
    public static MultimodalChatTemplate llama3() { return new MultimodalChatTemplate(Flavor.LLAMA3); }
    public static MultimodalChatTemplate mistral() { return new MultimodalChatTemplate(Flavor.MISTRAL); }
    public static MultimodalChatTemplate glm() { return new MultimodalChatTemplate(Flavor.GLM); }
    public static MultimodalChatTemplate gemma() { return new MultimodalChatTemplate(Flavor.GEMMA); }
    public static MultimodalChatTemplate gemma4() { return new MultimodalChatTemplate(Flavor.Gemma4); }
    public static MultimodalChatTemplate raw() { return new MultimodalChatTemplate(Flavor.RAW); }

    /**
     * Detect flavor from tokenizer config.
     */
    public static Flavor detectFlavor(Path dir) {
        Path tc = dir.resolve("tokenizer_config.json");
        if (Files.isRegularFile(tc)) {
            try {
                String raw = Files.readString(tc, StandardCharsets.UTF_8);
                Map<String, Object> m = Json.decodeObject(raw);
                Object ct = m.get("chat_template");
                if (ct != null) {
                    String s = String.valueOf(ct).toLowerCase();
                    if (s.contains("qwen2_5_omni") || s.contains("omni")) return Flavor.QWEN_OMNI;
                    if (s.contains("gemma4") || s.contains("omni")) return Flavor.Gemma4;
                    if (s.contains("gemma")) return Flavor.GEMMA;
                    if (s.contains("im_start") || s.contains("chatml")) return Flavor.QWEN;
                    if (s.contains("start_header_id") || s.contains("llama")) return Flavor.LLAMA3;
                    if (s.contains("[INST]") || s.contains("mistral")) return Flavor.MISTRAL;
                    if (s.contains("<|user|>") || s.contains("glm")) return Flavor.GLM;
                }
            } catch (IOException ignored) {}
        }
        return Flavor.QWEN;  // Default
    }

    public static MultimodalChatTemplate detect(Path dir) {
        return new MultimodalChatTemplate(detectFlavor(dir));
    }

    /**
     * Apply chat template to messages with multimodal support.
     *
     * @param messages List of message maps with role and content
     * @param processor The processor for tokenization
     * @param addGenerationPrompt Whether to add assistant header
     * @return Formatted prompt string
     */
    public String applyChatTemplate(List<Map<String, Object>> messages, Processor processor,
                                     boolean addGenerationPrompt) {
        Objects.requireNonNull(messages, "messages");

        return switch (flavor) {
            case QWEN -> applyQwen(messages, addGenerationPrompt);
            case QWEN_OMNI -> applyQwenOmni(messages, addGenerationPrompt);
            case LLAMA3 -> applyLlama3(messages, addGenerationPrompt);
            case MISTRAL -> applyMistral(messages, addGenerationPrompt);
            case GLM -> applyGlm(messages, addGenerationPrompt);
            case GEMMA -> applyGemma(messages, addGenerationPrompt);
            case Gemma4 -> applyGemma4(messages, processor, addGenerationPrompt);
            case RAW -> applyRaw(messages);
        };
    }

    public String applyChatTemplate(List<Map<String, Object>> messages, Processor processor) {
        return applyChatTemplate(messages, processor, true);
    }

    // ============= Qwen Omni (Audio support) =============

    private String applyQwenOmni(List<Map<String, Object>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, Object> msg : messages) {
            String role = getRole(msg);
            sb.append("<|im_start|>").append(role).append('\n');

            Object content = msg.get("content");
            if (content instanceof List) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                for (Map<String, Object> item : contentList) {
                    String type = (String) item.get("type");
                    switch (type) {
                        case "text" -> sb.append(item.get("text"));
                        case "image" -> sb.append("<|image_pad>");
                        case "audio" -> sb.append("<|audio_pad>");
                        case "video" -> sb.append("<|video_pad>");
                    }
                }
            } else {
                sb.append(content);
            }

            sb.append("<|im_end|>\n");
        }
        if (addGen) {
            sb.append("<|im_start|>assistant\n");
        }
        return sb.toString();
    }

    // ============= Standard Qwen =============

    private String applyQwen(List<Map<String, Object>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, Object> msg : messages) {
            String role = getRole(msg);
            sb.append("<|im_start|>").append(role).append('\n');

            Object content = msg.get("content");
            if (content instanceof List) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                for (Map<String, Object> item : contentList) {
                    String type = (String) item.get("type");
                    switch (type) {
                        case "text" -> sb.append(item.get("text"));
                        case "image" -> sb.append("<|image_pad>");
                        case "video" -> sb.append("<|video_pad>");
                    }
                }
            } else {
                sb.append(content);
            }

            sb.append("<|im_end|>\n");
        }
        if (addGen) {
            sb.append("<|im_start|>assistant\n");
        }
        return sb.toString();
    }

    // ============= Llama3 =============

    private String applyLlama3(List<Map<String, Object>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        sb.append("<|begin_of_text|>");

        for (Map<String, Object> msg : messages) {
            String role = getRole(msg);
            // Map roles to Llama3 format
            String llamaRole = mapRoleToLlama3(role);

            sb.append("<|start_header_id|>").append(llamaRole)
              .append("<|end_header_id|>\n\n");

            Object content = msg.get("content");
            if (content instanceof List) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                for (Map<String, Object> item : contentList) {
                    String type = (String) item.get("type");
                    switch (type) {
                        case "text" -> sb.append(item.get("text"));
                        case "image" -> sb.append("<|image|>");  // Placeholder
                    }
                }
            } else {
                sb.append(content);
            }

            sb.append("<|eot_id|>");
        }

        if (addGen) {
            sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }

        return sb.toString();
    }

    // ============= Gemma =============

    private String applyGemma(List<Map<String, Object>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();

        for (Map<String, Object> msg : messages) {
            String role = getRole(msg);
            // Gemma uses simpler format
            String gemmaRole = role.equals("assistant") ? "model" : role;

            sb.append("<start_of_turn>").append(gemmaRole).append('\n');

            Object content = msg.get("content");
            if (content instanceof List) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                for (Map<String, Object> item : contentList) {
                    String type = (String) item.get("type");
                    switch (type) {
                        case "text" -> sb.append(item.get("text"));
                        case "image" -> sb.append("<image>");
                    }
                }
            } else {
                sb.append(content);
            }

            sb.append("<end_of_turn>\n");
        }

        if (addGen) {
            sb.append("<start_of_turn>model\n");
        }

        return sb.toString();
    }

    // ============= Gemma4 Omni =============

    private String applyGemma4(List<Map<String, Object>> messages, Processor processor,
                                boolean addGen) {
        StringBuilder sb = new StringBuilder();

        for (Map<String, Object> msg : messages) {
            String role = getRole(msg);
            // Gemma4 uses 'model' for assistant
            String gemmaRole = role.equals("assistant") ? "model" : role;

            sb.append("<start_of_turn>").append(gemmaRole).append('\n');

            Object content = msg.get("content");
            if (content instanceof List) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                for (Map<String, Object> item : contentList) {
                    String type = (String) item.get("type");
                    switch (type) {
                        case "text" -> sb.append(item.get("text"));
                        case "image" -> sb.append("<image>");
                        case "video" -> sb.append("<video>");
                        case "audio" -> sb.append("<audio>");
                    }
                }
            } else {
                sb.append(content);
            }

            sb.append("<end_of_turn>\n");
        }

        if (addGen) {
            sb.append("<start_of_turn>model\n");
        }

        return sb.toString();
    }

    // ============= Mistral =============

    private String applyMistral(List<Map<String, Object>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        String system = null;

        for (Map<String, Object> msg : messages) {
            String role = getRole(msg);
            if ("system".equals(role)) {
                system = String.valueOf(msg.get("content"));
                continue;
            }

            if ("user".equals(role)) {
                sb.append("[INST] ");
                if (system != null) {
                    sb.append(system).append("\n\n");
                }
                Object content = msg.get("content");
                if (content instanceof List) {
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                    for (Map<String, Object> item : contentList) {
                        String type = (String) item.get("type");
                        if ("text".equals(type)) {
                            sb.append(item.get("text"));
                        }
                    }
                } else {
                    sb.append(content);
                }
                sb.append(" [/INST]");
            } else if ("assistant".equals(role)) {
                Object content = msg.get("content");
                if (content instanceof List) {
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                    for (Map<String, Object> item : contentList) {
                        String type = (String) item.get("type");
                        if ("text".equals(type)) {
                            sb.append(' ').append(item.get("text"));
                        }
                    }
                } else {
                    sb.append(' ').append(content);
                }
                sb.append("</s>");
            }
        }

        return sb.toString();
    }

    // ============= GLM =============

    private String applyGlm(List<Map<String, Object>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();

        for (Map<String, Object> msg : messages) {
            String role = getRole(msg);
            String tag = switch (role) {
                case "system" -> "<|system|>";
                case "assistant" -> "<|assistant|>";
                case "observation" -> "<|observation|>";
                default -> "<|user|>";
            };

            sb.append(tag).append('\n');

            Object content = msg.get("content");
            if (content instanceof List) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                for (Map<String, Object> item : contentList) {
                    String type = (String) item.get("type");
                    switch (type) {
                        case "text" -> sb.append(item.get("text"));
                        case "image" -> sb.append("<|image|>");
                    }
                }
            } else {
                sb.append(content);
            }
        }

        if (addGen) {
            sb.append("<|assistant|>\n");
        }

        return sb.toString();
    }

    // ============= Raw =============

    private String applyRaw(List<Map<String, Object>> messages) {
        StringBuilder sb = new StringBuilder();

        for (Map<String, Object> msg : messages) {
            if (sb.length() > 0) sb.append('\n');

            Object content = msg.get("content");
            if (content instanceof List) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                for (Map<String, Object> item : contentList) {
                    String type = (String) item.get("type");
                    if ("text".equals(type)) {
                        sb.append(item.get("text"));
                    }
                }
            } else {
                sb.append(content);
            }
        }

        return sb.toString();
    }

    // ============= Helpers =============

    private static String getRole(Map<String, Object> msg) {
        Object role = msg.get("role");
        return role == null ? "user" : String.valueOf(role);
    }

    private static String mapRoleToLlama3(String role) {
        return switch (role) {
            case "system" -> "system";
            case "user" -> "user";
            case "assistant" -> "assistant";
            case "tool" -> "tool";
            default -> role;
        };
    }

    public Flavor flavor() { return flavor; }
}
