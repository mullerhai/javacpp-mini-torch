/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import org.bytedeco.pytorch.llm.generation.StoppingCriteria;
import org.bytedeco.pytorch.llm.generation.TextIteratorStreamer;
import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The LLM pipeline the Web Demo's HTTP layer talks to.
 *
 * <p>Responsibilities:
 * <ul>
 *   <li>Hold a tokenizer + a list of {@link ToolEntry} entries (base model + adapters).</li>
 *   <li>Manage a {@link SessionRegistry} of {@link ChatSession}s.</li>
 *   <li>Resolve a {@link ChatTurn.RequestParams} into a {@link GenerationConfig} +
 *       {@link StoppingCriteria} pair, then drive an {@link InferenceBridge#predictStreaming} call.</li>
 *   <li>Append each chunk to the active {@link TextIteratorStreamer} and forward to a
 *       {@link ChunkListener} (the SSE handler wires itself as the listener).</li>
 *   <li>Support cancel mid-turn via {@code ChatSession#cancelRequested} — the engine checks
 *       this between tokens and aborts.</li>
 * </ul>
 *
 * <p>The engine is intentionally framework-agnostic: callers in {@code samples/llm/web}
 * inject the {@link InferenceBridge}, {@link PromptFormatter}, and tool list. The engine
 * itself does not depend on {@code BaseModel}, {@code ToolkitDemo}, or {@code PromptGenerator}.
 */
public final class ChatEngine {

    private final InferenceBridge bridge;
    private final Map<String, ToolEntry> tools = new LinkedHashMap<>();
    private final PromptFormatter promptFormatter;
    private final SessionRegistry sessions;
    private final List<GenerationConfigTemplate> templates = GenerationConfigTemplate.defaults();
    private final ExecutorService executor;
    private final int maxContext;
    private final String mode;

    public ChatEngine(FastTokenizer tokenizer,
                      InferenceBridge bridge,
                      List<ToolEntry> tools,
                      PromptFormatter promptFormatter,
                      long idleTtlMs,
                      int maxSessions,
                      int maxContext,
                      int executorThreads) {
        this.bridge = Objects.requireNonNull(bridge, "bridge");
        this.promptFormatter = promptFormatter == null ? PromptFormatter.simple() : promptFormatter;
        this.sessions = new SessionRegistry(idleTtlMs, maxSessions);
        this.maxContext = maxContext;
        if (tools != null) {
            for (ToolEntry t : tools) this.tools.put(t.name, t);
        }
        if (this.tools.isEmpty()) {
            this.tools.put("base", new ToolEntry("base", null));
        }
        this.executor = Executors.newFixedThreadPool(Math.max(2, executorThreads),
                r -> {
                    Thread th = new Thread(r, "ChatEngine-Worker");
                    th.setDaemon(true);
                    return th;
                });
        this.sessions.startEvictor();
        this.mode = !bridge.isStub() ? "real" : "stub";
        // tokenizer is held indirectly via InferenceBridge; suppress "unused" warning.
        if (tokenizer == null && !bridge.isStub()) {
            throw new IllegalArgumentException("tokenizer is required when bridge is not in stub mode");
        }
    }

    public String mode() { return mode; }

    public SessionRegistry sessions() { return sessions; }
    public ExecutorService executor() { return executor; }
    public List<GenerationConfigTemplate> templates() { return templates; }
    public int maxContext() { return maxContext; }
    public PromptFormatter promptFormatter() { return promptFormatter; }

    public List<Map<String, Object>> listTools() {
        List<Map<String, Object>> out = new ArrayList<>(tools.size());
        for (Map.Entry<String, ToolEntry> e : tools.entrySet()) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("name", e.getKey());
            m.put("adapterPath", e.getValue().adapterPath);
            m.put("description", e.getValue().description.isEmpty()
                    ? ("Tool " + e.getKey()) : e.getValue().description);
            out.add(m);
        }
        return out;
    }

    public GenerationConfigTemplate findTemplate(String name) {
        if (name == null) return null;
        for (GenerationConfigTemplate t : templates) {
            if (t.name.equalsIgnoreCase(name)) return t;
        }
        return null;
    }

    /**
     * Resolve a {@link ChatSession} for the given id, creating one if absent. Always
     * touches {@code lastAccessMillis}.
     */
    public ChatSession openSession(String sessionId) {
        return sessions.getOrCreate(sessionId);
    }

    public void closeSession(String sessionId) {
        sessions.remove(sessionId);
    }

    /**
     * Build the prompt string for a session + turn, formatted via the configured
     * {@link PromptFormatter}. The history is passed in chronological order.
     */
    public String buildPrompt(ChatSession session, ChatTurn turn) {
        List<ChatTurn> past = session.snapshotHistory();
        return promptFormatter.format(session, turn, past);
    }

    /**
     * Build the {@link GenerationConfig} for this turn — baseline template + per-turn
     * overrides + EOS token ids from the tokenizer.
     */
    public GenerationConfig buildGenerationConfig(ChatSession session, ChatTurn turn) {
        GenerationConfigTemplate tpl = findTemplate(session.currentTemplate);
        if (tpl == null) tpl = GenerationConfigTemplate.balanced();
        GenerationConfig.Builder b = GenerationConfig.builder()
                .doSample(tpl.doSample)
                .temperature(tpl.temperature)
                .topK(tpl.topK)
                .topP(tpl.topP)
                .repetitionPenalty(tpl.repetitionPenalty)
                .maxNewTokens(tpl.maxNewTokens);
        FastTokenizer tok = bridge.tokenizer();
        if (tok != null) {
            int eos = tok.eosId();
            if (eos >= 0) b.eosTokenId(eos);
        }
        turn.request.applyTo(b);
        return b.build();
    }

    /**
     * Build a {@link StoppingCriteria} instance carrying the turn's stop_strings and
     * repetition_limit, plus the tokenizer EOS.
     */
    public StoppingCriteria buildStoppingCriteria(ChatSession session, ChatTurn turn, GenerationConfig gen) {
        StoppingCriteria sc = StoppingCriteria.stopOnTokens(gen.maxNewTokens);
        FastTokenizer tok = bridge.tokenizer();
        if (tok != null) {
            int eos = tok.eosId();
            if (eos >= 0) sc.addStopToken(eos);
        }
        if (turn.request.stopStrings != null) {
            for (String s : turn.request.stopStrings) sc.addStopString(s, tok);
        }
        if (turn.request.repetitionLimit != null) {
            sc.addRepetitionLimit(turn.request.repetitionLimit.intValue());
        }
        GenerationConfigTemplate tpl = findTemplate(session.currentTemplate);
        if (tpl != null && tpl.repetitionLimit > 0 && turn.request.repetitionLimit == null) {
            sc.addRepetitionLimit(tpl.repetitionLimit);
        }
        return sc;
    }

    /**
     * Run a chat turn asynchronously. Returns a {@link CompletableFuture} that completes
     * when the turn is done. The provided {@link ChunkListener} receives every chunk.
     *
     * <p>If the session is already busy, the call throws {@link IllegalStateException}.
     */
    public CompletableFuture<ChatTurn> stream(ChatSession session, ChatTurn turn, ChunkListener listener) {
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(turn, "turn");
        if (session.busy) {
            throw new IllegalStateException("Session " + session.sessionId + " is busy");
        }
        session.busy = true;
        session.cancelRequested = false;
        GenerationConfig gen = buildGenerationConfig(session, turn);
        StoppingCriteria sc = buildStoppingCriteria(session, turn, gen);
        TextIteratorStreamer streamer = new TextIteratorStreamer();
        session.activeStreamer = streamer;
        session.activeListener = listener;
        String prompt = buildPrompt(session, turn);

        return CompletableFuture.supplyAsync(() -> {
            long start = System.currentTimeMillis();
            AtomicInteger tokenCount = new AtomicInteger(0);
            try {
                bridge.predictStreaming(prompt, gen, streamer, sc,
                        (chunk, tokenId) -> {
                            tokenCount.incrementAndGet();
                            turn.assistantTokens.add(chunk);
                            turn.tokensGenerated = tokenCount.get();
                            if (listener != null) listener.onChunk(session, turn, chunk);
                        },
                        null);
                if (session.cancelRequested) {
                    turn.stopReason = ChatTurn.STOP_USER_CANCEL;
                } else {
                    turn.stopReason = ChatTurn.STOP_EOS;
                }
            } catch (InferenceBridge.StopGenerationException sge) {
                turn.stopReason = session.cancelRequested ? ChatTurn.STOP_USER_CANCEL : ChatTurn.STOP_STRING;
            } catch (Throwable t) {
                turn.stopReason = ChatTurn.STOP_ERROR;
                turn.errorMessage = t.getMessage();
                if (listener != null) listener.onError(session, turn, t);
            } finally {
                long elapsed = System.currentTimeMillis() - start;
                turn.elapsedMs = elapsed;
                turn.tokensGenerated = tokenCount.get();
                turn.tokensPerSecond = elapsed > 0 ? tokenCount.get() * 1000.0 / elapsed : 0.0;
                turn.finishedAtMillis = System.currentTimeMillis();
                session.appendTurn(turn);
                session.busy = false;
                session.activeStreamer = null;
                session.activeListener = null;
                if (listener != null) listener.onComplete(session, turn);
            }
            return turn;
        }, executor);
    }

    public boolean cancel(String sessionId) {
        ChatSession s = sessions.peek(sessionId);
        if (s == null) return false;
        s.cancelRequested = true;
        TextIteratorStreamer st = s.activeStreamer;
        if (st != null) {
            try { st.end(); } catch (Throwable ignored) {}
        }
        return true;
    }

    /** Best-effort shutdown of the worker pool. */
    public void shutdown() {
        sessions.closeAll();
        executor.shutdownNow();
    }
}