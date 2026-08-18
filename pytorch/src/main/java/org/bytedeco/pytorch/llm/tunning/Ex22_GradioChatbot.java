/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.tunning;

import java.util.List;

import org.bytedeco.pytorch.llm.generation.GenerationConfig;
import org.bytedeco.pytorch.llm.generation.StoppingCriteria;
import org.bytedeco.pytorch.llm.generation.TextIteratorStreamer;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.ui.ChatbotDemo;

/** Ex22 — Gradio Chatbot demo with streaming. */
public final class Ex22_GradioChatbot {

    public static final String NAME = "Ex22_GradioChatbot";

    public static void run(FastTokenizer tokenizer) {
        TunningSupport.banner(22, "Gradio Chatbot");

        TextIteratorStreamer streamer = new TextIteratorStreamer();
        StoppingCriteria criteria = StoppingCriteria.stopOnTokens(20);
        criteria.addStopToken(tokenizer.eosId());

        ChatbotDemo.InMemoryRenderer renderer = new ChatbotDemo.InMemoryRenderer();
        ChatbotDemo chat = new ChatbotDemo(
                (msg, history) -> "[bot reply to '" + msg + "']",
                criteria,
                streamer,
                renderer);
        String reply = chat.reply("Hello", List.of());
        System.out.println("Chat reply: " + reply);
    }

    public static void main(String[] args) {
        try (FastTokenizer t = TunningSupport.tokenizerFor("chatbot")) {
            run(t);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
