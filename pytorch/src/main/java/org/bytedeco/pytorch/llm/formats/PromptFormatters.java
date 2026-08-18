/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.formats;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Mirrors the many prompt-formatting functions used across the parent Python collections:
 * Stanford Alpaca, Guanaco, ChatGLM, Llama-2 system prompt, Phi dialogstudio, Gemma turn, etc.
 *
 * <p>Each formatter exposes:
 * <ul>
 *   <li>{ #format(List)} that takes a list of {@code (role, content)} maps (or an Alpaca
 *       record) and returns a single {@code String} prompt.</li>
 *   <li>{ #responseTemplate()} returning the substring that separates prompt from completion
 *       for {@code DataCollatorForCompletionOnlyLM}.</li>
 *   <li>{#instructionTemplate()} returning the substring that marks the start of an instruction.</li>
 * </ul>
 */
public final class PromptFormatters {

    private PromptFormatters() {}

    /** Stanford Alpaca standard template. */
    public static final class Alpaca implements Formatter {
        @Override public String format(List<Map<String, String>> msgs) {
            StringBuilder sb = new StringBuilder();
            sb.append("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n");
            for (Map<String, String> m : msgs) {
                if ("system".equals(m.get("role"))) continue;
                if ("instruction".equals(m.get("role"))) {
                    sb.append("### Instruction:\n").append(m.get("content")).append("\n\n");
                } else if ("input".equals(m.get("role"))) {
                    sb.append("### Input:\n").append(m.get("content")).append("\n\n");
                } else if ("response".equals(m.get("role")) || "assistant".equals(m.get("role")) || "output".equals(m.get("role"))) {
                    sb.append("### Response:\n").append(m.get("content"));
                }
            }
            return sb.toString();
        }
        @Override public String instructionTemplate() { return "### Instruction:"; }
        @Override public String responseTemplate() { return "### Response:"; }
    }

    /** Guanaco / QLora format with separator tags. */
    public static final class Guanaco implements Formatter {
        @Override public String format(List<Map<String, String>> msgs) {
            StringBuilder sb = new StringBuilder();
            for (Map<String, String> m : msgs) {
                if ("system".equals(m.get("role"))) continue;
                if ("user".equals(m.get("role")) || "human".equals(m.get("role"))) {
                    sb.append("### Human: ").append(m.get("content")).append("\n");
                } else if ("assistant".equals(m.get("role")) || "gpt".equals(m.get("role"))) {
                    sb.append("### Assistant: ").append(m.get("content")).append("\n");
                }
            }
            return sb.toString();
        }
        @Override public String instructionTemplate() { return "### Human:"; }
        @Override public String responseTemplate() { return "### Assistant:"; }
    }

    /** Llama-2-chat template. */
    public static final class Llama2Chat implements Formatter {
        private final String system;
        public Llama2Chat() { this("You are a helpful, respectful and honest assistant."); }
        public Llama2Chat(String system) { this.system = system; }
        @Override public String format(List<Map<String, String>> msgs) {
            StringBuilder sb = new StringBuilder();
            boolean hasSystem = false;
            for (Map<String, String> m : msgs) {
                if ("system".equals(m.get("role"))) { hasSystem = true; sb.append("<<SYS>>").append(m.get("content")).append("<</SYS>>\n"); break; }
            }
            if (!hasSystem) sb.append("<<SYS>>").append(system).append("<</SYS>>\n");
            int i = 0;
            for (Map<String, String> m : msgs) {
                String role = m.get("role");
                if ("system".equals(role)) continue;
                if (i % 2 == 0) sb.append("[INST] ");
                else sb.append(" [/INST] ");
                if ("user".equals(role)) sb.append(m.get("content"));
                else sb.append(m.get("content")).append(" ");
                if (i % 2 == 1) sb.append("</s>");
                i++;
            }
            return sb.toString();
        }
        @Override public String instructionTemplate() { return "[INST]"; }
        @Override public String responseTemplate() { return "[/INST]"; }
    }

    /** ChatGLM / Qwen standard template with round tags. */
    public static final class ChatGLM implements Formatter {
        @Override public String format(List<Map<String, String>> msgs) {
            StringBuilder sb = new StringBuilder();
            for (Map<String, String> m : msgs) {
                String role = m.get("role");
                String content = m.get("content");
                if ("user".equals(role)) sb.append("[Round ").append(1).append("]\n\n问：").append(content).append("\n\n");
                else if ("assistant".equals(role)) sb.append("答：").append(content).append("\n\n");
                else if ("system".equals(role)) sb.append(content).append("\n\n");
            }
            return sb.toString();
        }
        @Override public String instructionTemplate() { return "问："; }
        @Override public String responseTemplate() { return "答："; }
    }

    /** Gemma turn-based format used by Gemma-2 / Gemma-ko. */
    public static final class GemmaTurn implements Formatter {
        @Override public String format(List<Map<String, String>> msgs) {
            StringBuilder sb = new StringBuilder("<bos>");
            for (Map<String, String> m : msgs) {
                if ("system".equals(m.get("role"))) continue;
                String tag = "user".equals(m.get("role")) ? "user" : "model";
                sb.append("<start_of_turn>").append(tag).append("\n").append(m.get("content")).append("<end_of_turn>\n");
            }
            sb.append("<eos>");
            return sb.toString();
        }
        @Override public String instructionTemplate() { return "<start_of_turn>user"; }
        @Override public String responseTemplate() { return "<start_of_turn>model"; }
    }

    /** StableVicuna prompt template. */
    public static final class StableVicuna implements Formatter {
        @Override public String format(List<Map<String, String>> msgs) {
            StringBuilder sb = new StringBuilder("A chat between a curious human and an artificial intelligence assistant. ");
            for (Map<String, String> m : msgs) {
                String role = m.get("role");
                if ("user".equals(role) || "human".equals(role)) sb.append("### Human: ").append(m.get("content")).append(" \n");
                else if ("assistant".equals(role) || "gpt".equals(role)) sb.append("### Assistant: ").append(m.get("content")).append(" \n");
                else if ("system".equals(role)) sb.insert(0, m.get("content") + " ");
            }
            return sb.toString();
        }
        @Override public String instructionTemplate() { return "### Human:"; }
        @Override public String responseTemplate() { return "### Assistant:"; }
    }

    /** Phi-1.5 DialogStudio-style (system + input + response). */
    public static final class PhiDialog implements Formatter {
        @Override public String format(List<Map<String, String>> msgs) {
            StringBuilder sb = new StringBuilder();
            String system = "Below is a conversation between a human and an AI agent. Write a summary of the conversation.";
            String user = "";
            String response = "";
            for (Map<String, String> m : msgs) {
                String role = m.get("role");
                if ("system".equals(role)) system = m.get("content");
                else if ("user".equals(role) || "input".equals(role)) user = m.get("content");
                else if ("response".equals(role) || "assistant".equals(role)) response = m.get("content");
            }
            sb.append("### Instruction: ").append(system).append("\n\n### Input:\n").append(user).append("\n\n### Response:\n").append(response);
            return sb.toString();
        }
        @Override public String instructionTemplate() { return "### Instruction:"; }
        @Override public String responseTemplate() { return "### Response:"; }
    }

    /** RLHF preference pair format used in beyondguo/RLHF/reward_modeling.py. */
    public static final class RLHFPair implements Formatter {
        @Override public String format(List<Map<String, String>> msgs) {
            StringBuilder sb = new StringBuilder();
            for (Map<String, String> m : msgs) {
                String role = m.get("role");
                if ("user".equals(role) || "prompt".equals(role) || "question".equals(role)) sb.append("问：").append(m.get("content")).append("\n\n");
                else if ("response".equals(role) || "assistant".equals(role) || "answer".equals(role)) sb.append("答：").append(m.get("content"));
            }
            return sb.toString();
        }
        @Override public String instructionTemplate() { return "问："; }
        @Override public String responseTemplate() { return "答："; }
    }

    /** Samantha assistant persona format used in Ex13. */
    public static final class Samantha implements Formatter {
        @Override public String format(List<Map<String, String>> msgs) {
            StringBuilder sb = new StringBuilder();
            for (Map<String, String> m : msgs) {
                String role = m.get("role");
                if ("system".equals(role)) sb.append(m.get("content")).append("\n\n");
                else if ("user".equals(role)) sb.append("User: ").append(m.get("content")).append("\n");
                else if ("assistant".equals(role)) sb.append("Assistant: ").append(m.get("content")).append("\n");
            }
            return sb.toString();
        }
        @Override public String instructionTemplate() { return "User:"; }
        @Override public String responseTemplate() { return "Assistant:"; }
    }

    /** KULLM + Gemma format used by russellgeum/LLM-Finetuning-Tutorial. */
    public static final class KullmGemma implements Formatter {
        @Override public String format(List<Map<String, String>> msgs) {
            StringBuilder sb = new StringBuilder("<bos>");
            String instruction = "", input = "", output = "";
            for (Map<String, String> m : msgs) {
                String role = m.get("role");
                if ("instruction".equals(role)) instruction = m.get("content");
                else if ("input".equals(role)) input = m.get("content");
                else if ("response".equals(role) || "output".equals(role)) output = m.get("content");
                else if ("user".equals(role)) instruction = m.get("content");
                else if ("assistant".equals(role)) output = m.get("content");
            }
            sb.append("<start_of_turn>user\n").append(instruction);
            if (input != null && !input.isEmpty()) sb.append(" ").append(input);
            sb.append("<end_of_turn>\n\n<start_of_turn>model\n").append(output).append("<end_of_turn><eos>");
            return sb.toString();
        }
        @Override public String instructionTemplate() { return "<start_of_turn>user"; }
        @Override public String responseTemplate() { return "<start_of_turn>model"; }
    }

    /** EEVE-style Korean prompt. */
    public static final class EEVE implements Formatter {
        @Override public String format(List<Map<String, String>> msgs) {
            StringBuilder sb = new StringBuilder();
            for (Map<String, String> m : msgs) {
                if ("user".equals(m.get("role"))) sb.append("User:\n").append(m.get("content"));
                else if ("assistant".equals(m.get("role"))) sb.append("\n\nAssistant:\n").append(m.get("content"));
            }
            return sb.toString();
        }
        @Override public String instructionTemplate() { return "User:"; }
        @Override public String responseTemplate() { return "Assistant:"; }
    }

    public interface Formatter {
        String format(List<Map<String, String>> messages);
        String instructionTemplate();
        String responseTemplate();
    }

    /** Helpers for converting raw dataset rows into (role,content) maps. */
    public static List<Map<String, String>> fromAlpacaRow(Map<String, Object> row) {
        List<Map<String, String>> out = new ArrayList<>();
        out.add(msg("instruction", str(row.get("instruction"))));
        out.add(msg("input", str(row.get("input"))));
        out.add(msg("response", str(row.get("output"))));
        return out;
    }
    public static List<Map<String, String>> fromChatmlRow(Map<String, Object> row) {
        List<Map<String, String>> out = new ArrayList<>();
        Object m = row.get("messages");
        if (m instanceof List) {
            for (Object o : (List<?>) m) {
                if (o instanceof Map) {
                    Map<?, ?> mm = (Map<?, ?>) o;
                    out.add(msg(str(mm.get("role")), str(mm.get("content"))));
                }
            }
        }
        return out;
    }
    public static List<Map<String, String>> fromRLHFRow(Map<String, Object> row) {
        List<Map<String, String>> out = new ArrayList<>();
        out.add(msg("user", str(row.get("prompt"))));
        out.add(msg("response", str(row.get("chosen"))));
        return out;
    }

    public static Map<String, String> msg(String role, String content) {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("role", role == null ? "user" : role);
        m.put("content", content == null ? "" : content);
        return m;
    }
    public static String str(Object o) { return o == null ? "" : o.toString(); }
}