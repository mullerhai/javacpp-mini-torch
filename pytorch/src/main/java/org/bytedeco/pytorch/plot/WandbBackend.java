/*
 * TrainingBackend adapter for WandbClient (Phase 5).
 */
package org.bytedeco.pytorch.plot;

import org.bytedeco.pytorch.plot.wandb.WandbClient;
import org.bytedeco.pytorch.plot.wandb.WandbLocalServer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** Adapter: WandbClient → TrainingBackend. */
public final class WandbBackend implements TrainingBackend {

    private final WandbClient client;
    private final WandbLocalServer server; // nullable
    private final boolean ownsServer;

    private WandbBackend(WandbClient client, WandbLocalServer server, boolean ownsServer) {
        this.client = client;
        this.server = server;
        this.ownsServer = ownsServer;
    }

    /** Auto-detect mode: try remote, fallback to LocalServer on failure. */
    public static WandbBackend of(String runName, Map<String, ?> config) throws IOException {
        return of(runName, config, null);
    }

    /** Pass existing local server (offline) or null for auto-detect. */
    public static WandbBackend of(String runName, Map<String, ?> config, WandbLocalServer server)
            throws IOException {
        WandbLocalServer local = server;
        boolean owns = false;
        if (local == null) {
            // try remote first
            try {
                WandbClient client = WandbClient.newBuilder()
                        .host("api.wandb.ai").port(443).useHttps(true)
                        .build();
                client.initRun(runName, config);
                return new WandbBackend(client, null, false);
            } catch (Exception e) {
                // fallback to local
            }
            local = WandbLocalServer.start(0);
            owns = true;
        }
        WandbClient client = WandbClient.newBuilder()
                .offline(local)
                .entity("local").project("default")
                .build();
        try { client.initRun(runName, config); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IOException(e); }
        return new WandbBackend(client, local, owns);
    }

    @Override public String name() { return "wandb"; }

    @Override public boolean isReady() {
        return true; /* client maintains its own retry / fallback */
    }

    @Override
    public String init(String runName, Map<String, ?> config) throws IOException {
        try { client.initRun(runName, config); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IOException(e); }
        return runName;
    }

    @Override
    public void finish() throws IOException {
        try { client.finish(); } catch (Exception e) { throw new IOException(e); }
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
        // Store as summary entry with metadata; full W&B-style artifact upload
        // requires server-side support. LocalServer stores it under the run dir.
        logSummary(Map.of(
                "artifact/" + name + "/path", file.toString(),
                "artifact/" + name + "/type", type == null ? "model" : type,
                "artifact/" + name + "/bytes", Files.size(file)));
    }

    /** Access underlying WandbClient for advanced APIs (e.g. alert / watch). */
    public WandbClient client() { return client; }
    public WandbLocalServer localServer() { return server; }

    @Override
    public void close() {
        try { client.close(); } catch (Exception ignored) {}
        if (ownsServer && server != null) {
            try { server.close(); } catch (Exception ignored) {}
        }
    }
}