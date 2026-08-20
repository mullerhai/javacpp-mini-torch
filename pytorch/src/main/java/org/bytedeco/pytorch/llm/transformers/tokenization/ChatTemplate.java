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
package org.bytedeco.pytorch.llm.transformers.tokenization;

import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Minimal chat templates for Instruct models (no full Jinja engine).
 *
 * <p>Supported flavors:
 * <ul>
 *   <li>{@code qwen} — ChatML {@code <|im_start|>role\n…<|im_end|>}</li>
 *   <li>{@code llama3} — Llama-3 header style</li>
 *   <li>{@code glm} — GLM-Edge {@code <|system|>}/{@code <|user|>}/{@code <|assistant|>}</li>
 *   <li>{@code raw} — concatenate content only</li>
 * </ul>
 */
public final class ChatTemplate {

    public enum Flavor { QWEN, LLAMA3, MISTRAL, GLM, RAW, CUSTOM }

    private final Flavor flavor;
    /** Raw Jinja (or ChatML) template when {@link Flavor#CUSTOM}. */
    private final String customTemplate;

    public ChatTemplate(Flavor flavor) {
        this(flavor, null);
    }

    public ChatTemplate(Flavor flavor, String customTemplate) {
        this.flavor = flavor == null ? Flavor.RAW : flavor;
        this.customTemplate = customTemplate;
    }

    public static ChatTemplate qwen() { return new ChatTemplate(Flavor.QWEN); }
    public static ChatTemplate llama3() { return new ChatTemplate(Flavor.LLAMA3); }
    public static ChatTemplate mistral() { return new ChatTemplate(Flavor.MISTRAL); }
    public static ChatTemplate glm() { return new ChatTemplate(Flavor.GLM); }
    public static ChatTemplate raw() { return new ChatTemplate(Flavor.RAW); }

    /**
     * HuggingFace {@code tokenizer.chat_template = jinja}. Stores the raw string
     * (including TRL {@code {% generation %}} markers) and renders a ChatML /
     * Llama-3 / Mistral subset. Not a full Jinja VM — see {@link #applyCustom}.
     */
    public static ChatTemplate custom(String jinja) {
        return new ChatTemplate(Flavor.CUSTOM, jinja);
    }

    public static ChatTemplate forModelType(PretrainedConfig.ModelType type) {
        if (type == null) return raw();
        return switch (type) {
            case QWEN -> qwen();
            case LLAMA -> llama3();
            case MISTRAL -> mistral();
            case GLM -> glm();
            default -> raw();
        };
    }

    /** Detect from tokenizer_config.json chat_template string or model type. */
    public static ChatTemplate detect(Path dir, PretrainedConfig cfg) {
        Path tc = dir.resolve("tokenizer_config.json");
        if (Files.isRegularFile(tc)) {
            try {
                String raw = Files.readString(tc, StandardCharsets.UTF_8);
                Map<String, Object> m = Json.decodeObject(raw);
                Object ct = m.get("chat_template");
                if (ct != null) {
                    String s = String.valueOf(ct).toLowerCase(Locale.ROOT);
                    if (s.contains("im_start") || s.contains("chatml")) return qwen();
                    if (s.contains("<|user|>") || s.contains("<|assistant|>") || s.contains("glm")) return glm();
                    if (s.contains("start_header_id") || s.contains("llama")) return llama3();
                    if (s.contains("[INST]") || s.contains("mistral")) return mistral();
                }
            } catch (IOException ignored) {}
        }
        return forModelType(cfg == null ? null : cfg.modelType());
    }

    /**
     * @param messages list of {@code {role, content}} maps
     * @param addGenerationPrompt append assistant header for generation
     */
    public String apply(List<Map<String, String>> messages, boolean addGenerationPrompt) {
        Objects.requireNonNull(messages, "messages");
        return switch (flavor) {
            case QWEN -> applyQwen(messages, addGenerationPrompt);
            case LLAMA3 -> applyLlama3(messages, addGenerationPrompt);
            case MISTRAL -> applyMistral(messages, addGenerationPrompt);
            case GLM -> applyGlm(messages, addGenerationPrompt);
            case RAW -> applyRaw(messages);
            case CUSTOM -> applyCustom(messagesToObject(messages), addGenerationPrompt).text();
        };
    }

    /**
     * Apply to {@code List<Map<String,Object>>} messages (HF conversational
     * rows whose {@code content} may be a nested list).
     */
    @SuppressWarnings("unchecked")
    public String applyObject(List<? extends Map<String, ?>> messages, boolean addGenerationPrompt) {
        Objects.requireNonNull(messages, "messages");
        List<Map<String, Object>> obj = new ArrayList<>(messages.size());
        for (Map<String, ?> m : messages) {
            obj.add((Map<String, Object>) (Map<?, ?>) m);
        }
        if (flavor == Flavor.CUSTOM) {
            return applyCustom(obj, addGenerationPrompt).text();
        }
        List<Map<String, String>> str = new ArrayList<>(obj.size());
        for (Map<String, Object> m : obj) {
            Map<String, String> row = new java.util.LinkedHashMap<>();
            row.put("role", roleOfMultimodal(m));
            Object c = m.get("content");
            row.put("content", c == null ? "" : String.valueOf(c instanceof List ? flattenTextContent(c) : c));
            str.add(row);
        }
        return apply(str, addGenerationPrompt);
    }

    public String apply(List<Map<String, String>> messages) {
        return apply(messages, true);
    }

    /**
     * Apply chat template with multimodal content support.
     *
     * <p>Messages can contain nested content with text, images, videos, audio.
     *
     * @param messages List of message maps with role and content
     * @param addGenerationPrompt append assistant header for generation
     */
    @SuppressWarnings("unchecked")
    public String applyMultimodal(List<Map<String, Object>> messages, boolean addGenerationPrompt) {
        Objects.requireNonNull(messages, "messages");
        return switch (flavor) {
            case QWEN -> applyQwenMultimodal(messages, addGenerationPrompt);
            case LLAMA3 -> applyLlama3Multimodal(messages, addGenerationPrompt);
            case MISTRAL -> applyMistralMultimodal(messages, addGenerationPrompt);
            case GLM -> applyGlmMultimodal(messages, addGenerationPrompt);
            case RAW -> applyRawMultimodal(messages);
            case CUSTOM -> applyCustom(messages, addGenerationPrompt).text();
        };
    }

    public String applyMultimodal(List<Map<String, Object>> messages) {
        return applyMultimodal(messages, true);
    }

    private static String roleOfMultimodal(Map<String, Object> m) {
        Object r = m.get("role");
        return r == null ? "user" : String.valueOf(r);
    }

    @SuppressWarnings("unchecked")
    private static String applyQwenMultimodal(List<Map<String, Object>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, Object> msg : messages) {
            sb.append("<|im_start|>").append(roleOfMultimodal(msg)).append('\n');

            Object content = msg.get("content");
            if (content instanceof List) {
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                for (Map<String, Object> item : contentList) {
                    String type = String.valueOf(item.get("type"));
                    switch (type) {
                        case "text" -> sb.append(item.get("text"));
                        case "image" -> sb.append("<|image_pad|>");
                        case "video" -> sb.append("<|video_pad|>");
                        case "audio" -> sb.append("<|audio_pad|>");
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

    @SuppressWarnings("unchecked")
    private static String applyLlama3Multimodal(List<Map<String, Object>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        sb.append("<|begin_of_text|>");
        for (Map<String, Object> msg : messages) {
            String role = roleOfMultimodal(msg);
            sb.append("<|start_header_id|>").append(role).append("<|end_header_id|>\n\n");

            Object content = msg.get("content");
            if (content instanceof List) {
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                for (Map<String, Object> item : contentList) {
                    String type = String.valueOf(item.get("type"));
                    if ("text".equals(type)) {
                        sb.append(item.get("text"));
                    } else if ("image".equals(type)) {
                        sb.append("<|image|>");
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

    @SuppressWarnings("unchecked")
    private static String applyMistralMultimodal(List<Map<String, Object>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        String system = null;
        for (Map<String, Object> msg : messages) {
            String role = roleOfMultimodal(msg);
            if ("system".equals(role)) {
                Object content = msg.get("content");
                system = content instanceof String ? String.valueOf(content) : "";
                continue;
            }
            if ("user".equals(role)) {
                sb.append("[INST] ");
                if (system != null) {
                    sb.append(system).append("\n\n");
                    system = null;
                }
                Object content = msg.get("content");
                if (content instanceof List) {
                    List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                    for (Map<String, Object> item : contentList) {
                        if ("text".equals(String.valueOf(item.get("type")))) {
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
                    List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                    for (Map<String, Object> item : contentList) {
                        if ("text".equals(String.valueOf(item.get("type")))) {
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

    @SuppressWarnings("unchecked")
    private static String applyGlmMultimodal(List<Map<String, Object>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, Object> msg : messages) {
            String role = roleOfMultimodal(msg);
            String tag = switch (role) {
                case "system" -> "<|system|>";
                case "assistant" -> "<|assistant|>";
                case "observation" -> "<|observation|>";
                default -> "<|user|>";
            };
            sb.append(tag).append('\n');

            Object content = msg.get("content");
            if (content instanceof List) {
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                for (Map<String, Object> item : contentList) {
                    String type = String.valueOf(item.get("type"));
                    if ("text".equals(type)) {
                        sb.append(item.get("text"));
                    } else if ("image".equals(type)) {
                        sb.append("<|image|>");
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

    @SuppressWarnings("unchecked")
    private static String applyRawMultimodal(List<Map<String, Object>> messages) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, Object> msg : messages) {
            if (sb.length() > 0) sb.append('\n');

            Object content = msg.get("content");
            if (content instanceof List) {
                List<Map<String, Object>> contentList = (List<Map<String, Object>>) content;
                for (Map<String, Object> item : contentList) {
                    if ("text".equals(String.valueOf(item.get("type")))) {
                        sb.append(item.get("text"));
                    }
                }
            } else {
                sb.append(content);
            }
        }
        return sb.toString();
    }

    private static String roleOf(Map<String, String> m) {
        String r = m.get("role");
        return r == null ? "user" : r;
    }

    private static String contentOf(Map<String, String> m) {
        String c = m.get("content");
        return c == null ? "" : c;
    }

    private static String applyQwen(List<Map<String, String>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> msg : messages) {
            sb.append("<|im_start|>").append(roleOf(msg)).append('\n')
              .append(contentOf(msg)).append("<|im_end|>\n");
        }
        if (addGen) {
            sb.append("<|im_start|>assistant\n");
        }
        return sb.toString();
    }

    private static String applyLlama3(List<Map<String, String>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        sb.append("<|begin_of_text|>");
        for (Map<String, String> msg : messages) {
            sb.append("<|start_header_id|>").append(roleOf(msg)).append("<|end_header_id|>\n\n")
              .append(contentOf(msg)).append("<|eot_id|>");
        }
        if (addGen) {
            sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        return sb.toString();
    }

    private static String applyMistral(List<Map<String, String>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        String system = null;
        for (Map<String, String> msg : messages) {
            if ("system".equals(roleOf(msg))) {
                system = contentOf(msg);
                break;
            }
        }
        for (Map<String, String> msg : messages) {
            String role = roleOf(msg);
            if ("system".equals(role)) continue;
            if ("user".equals(role)) {
                sb.append("[INST] ");
                if (system != null) {
                    sb.append(system).append("\n\n");
                    system = null; // only once
                }
                sb.append(contentOf(msg)).append(" [/INST]");
            } else if ("assistant".equals(role)) {
                sb.append(' ').append(contentOf(msg)).append("</s>");
            }
        }
        // addGen: mistral expects model to continue after [/INST]
        return sb.toString();
    }

    /**
     * GLM-Edge / ChatGLM chat format from tokenizer chat_template:
     * {@code <|system|>\n…\n<|user|>\n…\n<|assistant|>\n…}
     */
    private static String applyGlm(List<Map<String, String>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> msg : messages) {
            String role = roleOf(msg);
            String tag = switch (role) {
                case "system" -> "<|system|>";
                case "assistant" -> "<|assistant|>";
                case "observation" -> "<|observation|>";
                default -> "<|user|>";
            };
            sb.append(tag).append('\n').append(contentOf(msg));
        }
        if (addGen) {
            sb.append("<|assistant|>\n");
        }
        return sb.toString();
    }

    private static String applyRaw(List<Map<String, String>> messages) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> msg : messages) {
            if (sb.length() > 0) sb.append('\n');
            sb.append(contentOf(msg));
        }
        return sb.toString();
    }

    public Flavor flavor() {
        return flavor;
    }

    /** Raw Jinja / ChatML string when this instance was built via {@link #custom}. */
    public String customTemplate() {
        return customTemplate;
    }

    /**
     * Result of applying a custom (Jinja-subset) template.
     *
     * @param text rendered prompt
     * @param generationCharSpans {@code [start, end)} character offsets of
     *        assistant completions marked by {@code {% generation %}} (or, when
     *        the template has no such tags, every assistant turn)
     */
    public record ChatTemplateResult(String text, List<int[]> generationCharSpans) {}

    /**
     * Minimal renderer for TRL SFT {@code assistant_only_loss}.
     *
     * <p>Not a full Jinja VM. Supported subset (matches the Qwen-2.5 modified
     * template in the SFT tutorial and transformers#34172):
     * <ul>
     *   <li>ChatML ({@code <|im_start|>role\\ncontent<|im_end|>})</li>
     *   <li>Llama-3 ({@code <|start_header_id|>})</li>
     *   <li>Mistral {@code [INST]}</li>
     *   <li>{@code {% generation %}...{% endgeneration %}} spans used as the
     *       completion mask</li>
     * </ul>
     * The {@code {% if tools %}} branch is not expanded; calling with a
     * non-empty {@code tools} key throws.
     */
    public ChatTemplateResult applyCustom(List<Map<String, Object>> messages, boolean addGenerationPrompt) {
        Objects.requireNonNull(messages, "messages");
        for (Map<String, Object> m : messages) {
            if (m != null && m.containsKey("tools") && m.get("tools") != null) {
                throw new IllegalArgumentException(
                        "Custom chat template tools branch is not implemented; strip tools from messages");
            }
        }
        String tmpl = customTemplate == null ? "" : customTemplate;
        String lower = tmpl.toLowerCase(Locale.ROOT);
        if (lower.contains("[inst]") || lower.contains("mistral")) {
            String text = applyMistral(toStringMessages(messages), addGenerationPrompt);
            return new ChatTemplateResult(text, assistantSpansMistral(text));
        }
        if (lower.contains("start_header_id") || lower.contains("<|begin_of_text|>")) {
            String text = applyLlama3(toStringMessages(messages), addGenerationPrompt);
            return new ChatTemplateResult(text, assistantSpansLlama3(text));
        }
        // Default: Qwen ChatML. Honor {% generation %} by recording assistant turns.
        return renderQwenWithGenerationSpans(messages, addGenerationPrompt, tmpl.contains("{% generation %}"));
    }

    private static ChatTemplateResult renderQwenWithGenerationSpans(
            List<Map<String, Object>> messages, boolean addGen, boolean hasGenerationTag) {
        StringBuilder sb = new StringBuilder();
        List<int[]> spans = new ArrayList<>();
        for (Map<String, Object> msg : messages) {
            String role = roleOfMultimodal(msg);
            String content = flattenTextContent(msg.get("content"));
            sb.append("<|im_start|>").append(role).append('\n');
            int contentStart = sb.length();
            sb.append(content).append("<|im_end|>\n");
            if ("assistant".equals(role)) {
                // Span covers assistant content (and we include the following im_end
                // so the EOS of the completion is trained, matching TRL).
                spans.add(new int[]{contentStart, sb.length()});
            }
        }
        if (addGen) {
            sb.append("<|im_start|>assistant\n");
        }
        if (!hasGenerationTag) {
            // No {% generation %} in the stored template: still mask assistant
            // turns (TRL assistant_only_loss conversational default).
        }
        return new ChatTemplateResult(sb.toString(), spans);
    }

    private static List<int[]> assistantSpansLlama3(String text) {
        List<int[]> spans = new ArrayList<>();
        String header = "<|start_header_id|>assistant<|end_header_id|>\n\n";
        String eot = "<|eot_id|>";
        int from = 0;
        while (true) {
            int i = text.indexOf(header, from);
            if (i < 0) break;
            int start = i + header.length();
            int end = text.indexOf(eot, start);
            if (end < 0) end = text.length();
            else end += eot.length();
            spans.add(new int[]{start, end});
            from = end;
        }
        return spans;
    }

    private static List<int[]> assistantSpansMistral(String text) {
        List<int[]> spans = new ArrayList<>();
        String instEnd = "[/INST]";
        int from = 0;
        while (true) {
            int i = text.indexOf(instEnd, from);
            if (i < 0) break;
            int start = i + instEnd.length();
            int end = text.indexOf("[INST]", start);
            if (end < 0) end = text.length();
            spans.add(new int[]{start, end});
            from = end;
        }
        return spans;
    }

    @SuppressWarnings("unchecked")
    private static String flattenTextContent(Object content) {
        if (content == null) return "";
        if (content instanceof List<?> list) {
            StringBuilder sb = new StringBuilder();
            for (Object item : list) {
                if (item instanceof Map<?, ?> m) {
                    Object t = m.get("text");
                    if (t != null) sb.append(t);
                    else if (m.get("content") != null) sb.append(m.get("content"));
                } else if (item != null) {
                    sb.append(item);
                }
            }
            return sb.toString();
        }
        return String.valueOf(content);
    }

    private static List<Map<String, String>> toStringMessages(List<Map<String, Object>> messages) {
        List<Map<String, String>> out = new ArrayList<>(messages.size());
        for (Map<String, Object> m : messages) {
            Map<String, String> row = new java.util.LinkedHashMap<>();
            row.put("role", roleOfMultimodal(m));
            row.put("content", flattenTextContent(m.get("content")));
            out.add(row);
        }
        return out;
    }

    private static List<Map<String, Object>> messagesToObject(List<Map<String, String>> messages) {
        List<Map<String, Object>> out = new ArrayList<>(messages.size());
        for (Map<String, String> m : messages) {
            Map<String, Object> row = new java.util.LinkedHashMap<>();
            row.put("role", m.get("role"));
            row.put("content", m.get("content"));
            out.add(row);
        }
        return out;
    }
}
