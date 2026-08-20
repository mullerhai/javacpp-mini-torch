/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import org.bytedeco.pytorch.llm.generation.TextIteratorStreamer;

import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Manages the lifecycle of all active {@link ChatSession}s.
 *
 * <p>Backed by a {@link ConcurrentHashMap}; new sessions are created lazily on first
 * reference. A background sweeper thread ({@link #startEvictor()}) evicts sessions idle
 * for more than {@code idleTtlMs}.
 *
 * <p>One per {@code ChatEngine}. Thread-safe.
 */
public final class SessionRegistry {

    private final ConcurrentHashMap<String, ChatSession> sessions = new ConcurrentHashMap<>();
    private final long idleTtlMs;
    private final int maxSessions;
    private final AtomicLong generated = new AtomicLong(0L);
    private volatile Thread sweeper;
    private volatile boolean running = false;

    public SessionRegistry(long idleTtlMs, int maxSessions) {
        this.idleTtlMs = idleTtlMs > 0 ? idleTtlMs : 30L * 60L * 1000L;
        this.maxSessions = maxSessions > 0 ? maxSessions : 200;
    }

    /** Construct with sane defaults (30 min idle, 200 sessions). */
    public SessionRegistry() { this(30L * 60L * 1000L, 200); }

    public synchronized ChatSession getOrCreate(String sessionId) {
        if (sessionId == null || sessionId.isEmpty()) {
            sessionId = nextId();
        }
        ChatSession existing = sessions.get(sessionId);
        if (existing != null) {
            existing.touch();
            return existing;
        }
        if (sessions.size() >= maxSessions) {
            evictIdle();
            if (sessions.size() >= maxSessions) {
                throw new IllegalStateException("Max sessions reached (" + maxSessions + ")");
            }
        }
        ChatSession fresh = new ChatSession(sessionId);
        sessions.put(sessionId, fresh);
        return fresh;
    }

    public ChatSession peek(String sessionId) {
        if (sessionId == null) return null;
        ChatSession s = sessions.get(sessionId);
        if (s != null) s.touch();
        return s;
    }

    public void remove(String sessionId) {
        if (sessionId == null) return;
        ChatSession s = sessions.remove(sessionId);
        if (s != null && s.activeStreamer != null) {
            try { s.activeStreamer.end(); } catch (Throwable ignored) {}
        }
    }

    public Collection<ChatSession> all() { return sessions.values(); }

    public int size() { return sessions.size(); }

    public long idleTtlMs() { return idleTtlMs; }

    public synchronized void startEvictor() {
        if (running) return;
        running = true;
        sweeper = new Thread(this::evictLoop, "SessionRegistry-Evictor");
        sweeper.setDaemon(true);
        sweeper.start();
    }

    public synchronized void stopEvictor() {
        running = false;
        if (sweeper != null) sweeper.interrupt();
        sweeper = null;
    }

    public synchronized void closeAll() {
        stopEvictor();
        for (ChatSession s : sessions.values()) {
            TextIteratorStreamer st = s.activeStreamer;
            if (st != null) {
                try { st.end(); } catch (Throwable ignored) {}
            }
        }
        sessions.clear();
    }

    /** Generate a session id. The format is "S" + monotonic counter base36. */
    public String nextId() {
        long n = generated.incrementAndGet();
        long ts = System.currentTimeMillis();
        return "S" + Long.toString(ts, 36) + "-" + Long.toString(n, 36);
    }

    /** Manually evict idle sessions; called automatically by the sweeper. */
    public synchronized int evictIdle() {
        long now = System.currentTimeMillis();
        int removed = 0;
        java.util.Iterator<ChatSession> it = sessions.values().iterator();
        while (it.hasNext()) {
            ChatSession s = it.next();
            if (s.busy) continue; // never evict a session mid-turn
            if (now - s.lastAccessMillis > idleTtlMs) {
                TextIteratorStreamer st = s.activeStreamer;
                if (st != null) {
                    try { st.end(); } catch (Throwable ignored) {}
                }
                it.remove();
                removed++;
            }
        }
        return removed;
    }

    private void evictLoop() {
        while (running) {
            try {
                Thread.sleep(60_000L); // sweep every minute
                evictIdle();
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return;
            } catch (Throwable ignored) {
                // swallow — sweeper must never die
            }
        }
    }
}