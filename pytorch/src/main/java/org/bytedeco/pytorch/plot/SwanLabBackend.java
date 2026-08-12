/*
 * TrainingBackend adapter for SwanLabClient (Phase 5).
 */
package org.bytedeco.pytorch.plot;

import org.bytedeco.pytorch.plot.swanlab.SwanLabClient;
import org.bytedeco.pytorch.plot.swanlab.SwanLabLocalServer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

public final class SwanLabBackend implements TrainingBackend {

    private final SwanLabClient client;
    private final SwanLabLocalServer server;
    private final boolean ownsServer;

    private SwanLabBackend(SwanLabClient client, SwanLabLocalServer server, boolean ownsServer) {
        this.client = client;
        this.server = server;
        this.ownsServer = ownsServer;
    }

    public static SwanLabBackend of(String runName, Map<String, ?> config) throws IOException {
        return of(runName, config, null);
    }

    public static SwanLabBackend of(String runName, Map<String, ?> config, SwanLabLocalServer server)
            throws IOException {
        SwanLabLocalServer local = server;
        boolean owns = false;
        if (local == null) {
            local = SwanLabLocalServer.start(0);
            owns = true;
        }
        SwanLabClient client = SwanLabClient.newBuilder()
                .offline(local)
                .workspace("local").project("default")
                .build();
        try { client.init(config); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IOException(e); }
        return new SwanLabBackend(client, local, owns);
    }

    @Override public String name() { return "swanlab"; }

    @Override public boolean isReady() { return true; /* offline always ready */ }

    @Override
    public String init(String runName, Map<String, ?> config) throws IOException {
        try { client.init(config); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IOException(e); }
        return runName;
    }

    @Override
    public void finish() throws IOException {
        try { client.finish(); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IOException(e); }
    }

    @Override
    public void log(Map<String, ? extends Number> metrics, long step) throws IOException {
        try { client.log(metrics, step); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IOException(e); }
    }

    @Override
    public void logSummary(Map<String, ?> summary) throws IOException {
        try { client.logSummary(summary); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IOException(e); }
    }

    @Override
    public void logImage(String name, byte[] png, long step) throws IOException {
        try { client.logImage(name, png, step, java.util.Map.of()); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IOException(e); }
    }

    @Override
    public void logArtifact(String name, Path file, String type) throws IOException {
        if (file == null || !Files.exists(file)) return;
        logSummary(Map.of(
                "artifact/" + name + "/path", file.toString(),
                "artifact/" + name + "/type", type == null ? "model" : type,
                "artifact/" + name + "/bytes", Files.size(file)));
    }

    public SwanLabClient client() { return client; }
    public SwanLabLocalServer localServer() { return server; }

    @Override
    public void close() {
        try { client.close(); } catch (Exception ignored) {}
        if (ownsServer && server != null) {
            try { server.close(); } catch (Exception ignored) {}
        }
    }
}