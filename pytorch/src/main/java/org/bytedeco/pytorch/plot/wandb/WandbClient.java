package org.bytedeco.pytorch.plot.wandb;

import org.bytedeco.pytorch.StringTensorDictItem;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.plot.tensorboard.PngEncoder;
import org.bytedeco.pytorch.nn.Module;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Lightweight WandB-compatible experiment tracker for JavaCPP / LibTorch.
 *
 * <p>Two modes:
 * <ul>
 *   <li><b>Remote</b> — HTTP REST against a WandB-compatible backend
 *       ({@code host:port/api/...}).</li>
 *   <li><b>Offline / local</b> — spin {@link WandbLocalServer} (embedded JDK
 *       {@link com.sun.net.httpserver.HttpServer}) that stores runs on disk and
 *       serves a simple HTML dashboard so you can open a browser URL and verify
 *       metrics, heatmaps, images, tables without a cloud API key.</li>
 * </ul>
 *
 * <pre>{@code
 * // Offline demo (no API key required):
 * try (WandbLocalServer server = WandbLocalServer.start(0);   // ephemeral port
 *      WandbClient wb = WandbClient.newBuilder()
 *              .offline(server)
 *              .entity("local").project("demo").build()) {
 *     wb.initRun("exp1", Map.of("lr", "1e-3"));
 *     wb.log(Map.of("loss", 0.42, "acc", 0.91), 1);
 *     wb.logHeatmap("cm", matrix, 1);
 *     wb.logImage("sample", imageTensor, 1);
 *     System.out.println("open " + server.uiUrl());
 * }
 * }</pre>
 */
public final class WandbClient implements AutoCloseable {

    public enum ChartType { LINE, BAR, HISTOGRAM, SURFACE, HEATMAP, SCATTER, PIE }

    private final HttpClient http;
    private final URI baseUri;                 // …/api
    private final String apiKey;
    private final String entity;
    private final String project;
    private final WandbLocalServer local;      // non-null in offline mode
    private final Path runDir;                 // offline artifact dir (optional)
    private String runId;
    private String runName;
    private final AtomicLong stepCounter = new AtomicLong(0);
    private boolean finished;
    // Enterprise enhancements
    private final int maxRetries;
    private final long retryBackoffMs;
    private final int imageGridMaxCols;
    private final boolean raiseExceptions;
    private final ExecutorService asyncExecutor;
    private final boolean ownAsyncExecutor;
    private final AtomicBoolean closedFlag = new AtomicBoolean(false);

    private WandbClient(Builder b) {
        this.local = b.localServer;
        this.apiKey = b.apiKey == null ? "local-key" : b.apiKey;
        this.entity = Objects.requireNonNull(b.entity, "entity");
        this.project = Objects.requireNonNull(b.project, "project");
        this.runDir = b.runDir;
        this.maxRetries = b.maxRetries;
        this.retryBackoffMs = b.retryBackoffMs;
        this.imageGridMaxCols = b.imageGridMaxCols;
        this.raiseExceptions = b.raiseExceptions;
        if (b.asyncExecutor != null) {
            this.asyncExecutor = b.asyncExecutor;
            this.ownAsyncExecutor = false;
        } else {
            this.asyncExecutor = Executors.newCachedThreadPool(r -> {
                Thread t = new Thread(r, "wandb-async");
                t.setDaemon(true);
                return t;
            });
            this.ownAsyncExecutor = true;
        }
        this.http = HttpClient.newBuilder().connectTimeout(b.connectTimeout).build();
        if (local != null) {
            this.baseUri = URI.create(local.apiBase());
        } else {
            String scheme = b.useHttps ? "https" : "http";
            this.baseUri = URI.create(String.format("%s://%s:%d/api", scheme, b.host, b.port));
        }
    }

    public static Builder newBuilder() { return new Builder(); }

    public String runId() { return runId; }
    public String runName() { return runName; }
    public String entity() { return entity; }
    public String project() { return project; }
    public String uiUrl() {
        if (local != null && runId != null) {
            return local.uiUrl() + "/runs/" + entity + "/" + project + "/" + runId;
        }
        return baseUri.resolve("../" + entity + "/" + project).toString();
    }

    // =========================================================================
    // Run lifecycle
    // =========================================================================

    public void initRun(String name) throws IOException, InterruptedException {
        initRun(name, Map.of());
    }

    public void initRun(String name, Map<String, ?> config) throws IOException, InterruptedException {
        this.runName = name == null ? "run-" + System.currentTimeMillis() : name;
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("name", runName);
        payload.put("entity", entity);
        payload.put("project", project);
        payload.put("config", config == null ? Map.of() : config);
        payload.put("created_at", Instant.now().toString());
        Map<String, Object> resp = post("/runs", payload);
        this.runId = String.valueOf(resp.getOrDefault("id",
                UUID.randomUUID().toString().replace("-", "").substring(0, 8)));
        this.finished = false;
        if (runDir != null) {
            Files.createDirectories(runDir.resolve(runId));
            Files.writeString(runDir.resolve(runId).resolve("config.json"),
                    Json.encode(payload), StandardCharsets.UTF_8);
        }
    }

    public void finish() throws IOException, InterruptedException {
        if (runId == null || finished) return;
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("status", "finished");
        payload.put("finished_at", Instant.now().toString());
        post("/runs/" + runId, payload);
        finished = true;
    }

    // =========================================================================
    // Metrics
    // =========================================================================

    public void log(Map<String, ? extends Number> metrics, long step)
            throws IOException, InterruptedException {
        requireRun();
        stepCounter.set(Math.max(stepCounter.get(), step));
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("step", step);
        payload.put("metrics", metrics);
        payload.put("timestamp", Instant.now().toString());
        post("/metrics", payload);
    }

    public void log(Map<String, ? extends Number> metrics) throws IOException, InterruptedException {
        log(metrics, stepCounter.incrementAndGet());
    }

    public void logMetrics(Map<String, Number> metrics, long step, Map<String, Object> chartOpts)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("step", step);
        payload.put("metrics", metrics);
        if (chartOpts != null) payload.put("chart", chartOpts);
        payload.put("timestamp", Instant.now().toString());
        post("/metrics", payload);
    }

    // =========================================================================
    // Charts
    // =========================================================================

    public void logChart(String name, ChartType type, List<List<Double>> series,
                         String[] legends, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", type.name().toLowerCase());
        payload.put("step", step);
        payload.put("series", series);
        if (legends != null) payload.put("legend", legends);
        post("/charts", payload);
    }

    public void logHeatmap(String name, double[][] matrix, long step)
            throws IOException, InterruptedException {
        logHeatmap(name, matrix, step, null);
    }

    public void logHeatmap(String name, double[][] matrix, long step, Map<String, Object> opts)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", "heatmap");
        payload.put("step", step);
        payload.put("matrix", matrix);
        if (opts != null) payload.put("opts", opts);
        post("/charts", payload);
    }

    public void logHeatmap(String name, Tensor t, long step)
            throws IOException, InterruptedException {
        logHeatmap(name, tensorToMatrix(t), step, null);
    }

    public void logHistogram(String name, double[] values, int bins, long step)
            throws IOException, InterruptedException {
        requireRun();
        if (values == null) throw new IllegalArgumentException("values must not be null");
        if (bins <= 0) bins = Math.min(30, Math.max(1, values.length));
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", "histogram");
        payload.put("step", step);
        payload.put("values", values);
        payload.put("bins", bins);
        post("/charts", payload);
    }

    /** Tensor-input convenience. */
    public void logHistogram(String name, Tensor t, int bins, long step)
            throws IOException, InterruptedException {
        logHistogram(name, tensorToFloatArray(t), bins, step);
    }

    /** Tensor-input convenience, default bin count. */
    public void logHistogram(String name, Tensor t, long step)
            throws IOException, InterruptedException {
        logHistogram(name, t, 64, step);
    }

    public void logScatter(String name, double[][] points, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", "scatter");
        payload.put("step", step);
        payload.put("points", points);
        post("/charts", payload);
    }

    /** 3D surface chart (height-field Z[x,y]) — Plotly type=surface. */
    public void logSurface(String name, double[][] Z, long step)
            throws IOException, InterruptedException {
        logSurface(name, Z, step, null);
    }

    public void logSurface(String name, double[][] Z, long step, Map<String, Object> opts)
            throws IOException, InterruptedException {
        requireRun();
        if (Z == null || Z.length == 0) throw new IllegalArgumentException("Z must be non-empty");
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", "surface");
        payload.put("step", step);
        payload.put("matrix", Z);
        if (opts != null) payload.put("opts", opts);
        post("/charts", payload);
    }

    /** Polar scatter — points expressed as (angle, radius) pairs. */
    public void logPolar(String name, double[][] points, long step)
            throws IOException, InterruptedException {
        logPolar(name, points, step, null);
    }

    public void logPolar(String name, double[][] points, long step, Map<String, Object> opts)
            throws IOException, InterruptedException {
        requireRun();
        if (points == null || points.length == 0) throw new IllegalArgumentException("points empty");
        if (points[0].length != 2) throw new IllegalArgumentException("polar points must be Nx2");
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", "polar");
        payload.put("step", step);
        payload.put("points", points);
        if (opts != null) payload.put("opts", opts);
        post("/charts", payload);
    }

    /** Box plot — each column of {@code data} is one box. */
    public void logBox(String name, double[][] data, long step)
            throws IOException, InterruptedException {
        logBox(name, data, step, null);
    }

    public void logBox(String name, double[][] data, long step, Map<String, Object> opts)
            throws IOException, InterruptedException {
        requireRun();
        if (data == null || data.length == 0) throw new IllegalArgumentException("data empty");
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", "box");
        payload.put("step", step);
        payload.put("values", data);
        if (opts != null) payload.put("opts", opts);
        post("/charts", payload);
    }

    /** Pie chart — values + labels. */
    public void logPie(String name, double[] values, String[] labels, long step)
            throws IOException, InterruptedException {
        requireRun();
        if (values == null) throw new IllegalArgumentException("values must not be null");
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", "pie");
        payload.put("step", step);
        payload.put("values", values);
        if (labels != null) payload.put("labels", labels);
        post("/charts", payload);
    }

    /**
     * Upload an artifact (model / dataset / file) to the run. Recorded as both
     * a summary entry (path/bytes/type) and stored under {@link #runDir} when
     * an offline server is configured.
     */
    public void logArtifact(String name, Path file, String type)
            throws IOException, InterruptedException {
        requireRun();
        Objects.requireNonNull(file, "file");
        if (!Files.exists(file)) throw new IOException("artifact file does not exist: " + file);
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("type", type == null ? "model" : type);
        payload.put("path", file.toString());
        payload.put("bytes", Files.size(file));
        post("/artifacts", payload);
        // Always record into run dir under offline mode.
        if (runDir != null) {
            Path dst = runDir.resolve(runId).resolve("artifacts");
            Files.createDirectories(dst);
            Files.copy(file, dst.resolve(file.getFileName()), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /**
     * Watch a model by periodically recording its parameter histogram. The
     * caller is responsible for invoking this once per training step. Returns
     * the number of parameters captured.
     */
    public int watchModel(String prefix, Module model, long step)
            throws IOException, InterruptedException {
        requireRun();
        Objects.requireNonNull(model, "model");
        int captured = 0;
        try {
            for (int i = 0; i < model.named_parameters().size(); i++) {
                StringTensorDictItem item = model.named_parameters().get(i);
                Tensor p = item.value();
//                Tensor p = model.named_parameters().get(name);
                if (p == null || !p.defined() || p.numel() == 0) continue;
                String fullTag = (prefix == null || prefix.isEmpty() ? "" : prefix + "/") + item.key().toString();
                logHistogram(fullTag, tensorToFloatArray(p), 64, step);
                captured++;
            }
        } catch (Exception e) {
            // best-effort; do not interrupt training
            if (raiseExceptions) throw new IOException("wandb.watchModel failed: " + e.getMessage(), e);
        }
        return captured;
    }

    /**
     * Send an alert (no-op for the embedded LocalServer which just records it).
     */
    public void alert(String title, String text, String level) throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("title", title);
        payload.put("text", text);
        payload.put("level", level == null ? "INFO" : level);
        post("/alerts", payload);
    }

    private static double[] tensorToFloatArray(Tensor t) {
        if (t == null || !t.defined()) return new double[0];
        Tensor c = t.contiguous().cpu().to(org.bytedeco.pytorch.global.torch.kFloat()).flatten();
        long n = c.numel();
        double[] out = new double[(int) Math.min(n, Integer.MAX_VALUE)];
        org.bytedeco.javacpp.FloatPointer p = c.data_ptr_float();
        for (int i = 0; i < out.length; i++) out[i] = p.get(i);
        return out;
    }

    // =========================================================================
    // Tables / images / text / audio
    // =========================================================================

    public void logTable(String tableName, String[] columns, List<String[]> rows)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", tableName);
        payload.put("columns", columns);
        payload.put("rows", rows);
        post("/tables", payload);
    }

    public void logImage(String name, byte[] pngBytes, long step, Map<String, Object> opts)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("step", step);
        payload.put("bytes", pngBytes); // Json encodes as base64
        payload.put("format", "png");
        if (opts != null) payload.put("opts", opts);
        post("/images", payload);
        if (runDir != null) {
            Path img = runDir.resolve(runId).resolve("images");
            Files.createDirectories(img);
            Files.write(img.resolve(name.replace('/', '_') + "_s" + step + ".png"), pngBytes);
        }
    }

    public void logImage(String name, Tensor image, long step)
            throws IOException, InterruptedException {
        ImageBuf buf = tensorToPng(image);
        Map<String, Object> opts = new LinkedHashMap<>();
        opts.put("height", buf.height);
        opts.put("width", buf.width);
        opts.put("channels", buf.channels);
        logImage(name, buf.png, step, opts);
    }

    public void logImages(String name, Tensor batchNchw, long step)
            throws IOException, InterruptedException {
        // Build a real image grid instead of dropping images.
        if (batchNchw.dim() != 4) {
            logImage(name, batchNchw, step);
            return;
        }
        byte[] png = tensorBatchToGridPng(batchNchw, imageGridMaxCols);
        logImage(name, png, step, Map.of("kind", "image_grid", "n", batchNchw.size(0)));
    }

    /** Full NCHW → N-panel image grid. Returns the PNG bytes (or null on bad input). */
    public static byte[] tensorBatchToGridPng(Tensor batch, int maxCols) {
        if (batch.dim() != 4 || !batch.defined()) return null;
        int n = (int) batch.size(0);
        int c = (int) batch.size(1);
        int h = (int) batch.size(2);
        int w = (int) batch.size(3);
        if (n <= 0 || c <= 0 || h <= 0 || w <= 0) return null;
        int cols = Math.max(1, Math.min(maxCols, n));
        int rows = (n + cols - 1) / cols;
        int padding = 2;
        int outC = (c == 1) ? 3 : (c >= 3 ? 3 : c);
        int gh = h + 2 * padding;
        int gw = w + 2 * padding;
        int outH = rows * gh;
        int outW = cols * gw;
        float[] grid = new float[outC * outH * outW];
        // fill black background
        // (default-zero is fine)
        float[] chw = toFloatArray(batch.contiguous().cpu().to(org.bytedeco.pytorch.global.torch.kFloat()));
        for (int i = 0; i < n; i++) {
            int row = i / cols;
            int col = i % cols;
            int top = row * gh + padding;
            int left = col * gw + padding;
            for (int ci = 0; ci < outC; ci++) {
                int srcC = (outC == 3 && c >= 3) ? ci : 0;
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        float v = chw[i * c * h * w + srcC * h * w + y * w + x];
                        grid[ci * outH * outW + (top + y) * outW + (left + x)] = v;
                    }
                }
            }
        }
        return PngEncoder.encodeFloatHWC(toHwc(grid, outC, outH, outW), outH, outW, outC);
    }

    private static float[] toHwc(float[] chw, int c, int h, int w) {
        float[] hwc = new float[c * h * w];
        for (int ci = 0; ci < c; ci++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    hwc[(y * w + x) * c + ci] = chw[ci * h * w + y * w + x];
        return hwc;
    }

    public void logText(String name, String text, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("step", step);
        payload.put("text", text);
        post("/text", payload);
    }

    public void logAudio(String name, float[] mono, int sampleRate, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("name", name);
        payload.put("step", step);
        payload.put("sample_rate", sampleRate);
        // store as list (JSON) — local server can render a sparkline; full WAV optional
        List<Double> samples = new ArrayList<>(Math.min(mono.length, 4000));
        int stride = Math.max(1, mono.length / 4000);
        for (int i = 0; i < mono.length; i += stride) samples.add((double) mono[i]);
        payload.put("waveform", samples);
        payload.put("n_samples", mono.length);
        post("/audio", payload);
    }

    public void logAudio(String name, Tensor waveform, int sampleRate, long step)
            throws IOException, InterruptedException {
        logAudio(name, toFloatArray(waveform), sampleRate, step);
    }

    public void logSummary(Map<String, ?> summary) throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("summary", summary);
        post("/summary", payload);
    }

    // =========================================================================
    // HTTP
    // =========================================================================

    private void requireRun() {
        if (runId == null) throw new IllegalStateException("call initRun() first");
        if (closedFlag.get()) throw new IllegalStateException("client is closed");
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> post(String path, Map<String, Object> payload)
            throws IOException, InterruptedException {
        // Fast path: in-process local server (no real HTTP roundtrip needed, but we still
        // go through HTTP so the protocol is exercised end-to-end).
        String json = Json.encode(payload);
        HttpRequest req = HttpRequest.newBuilder(baseUri.resolve(path.startsWith("/") ? path.substring(1) : path))
                .timeout(Duration.ofSeconds(15))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8))
                .build();
        IOException lastIo = null;
        InterruptedException lastInterrupt = null;
        int attempts = Math.max(1, maxRetries + 1);
        for (int i = 0; i < attempts; i++) {
            try {
                HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
                if (resp.statusCode() >= 400) {
                    String body = resp.body();
                    // Retry on 5xx (transient); fail on 4xx
                    if (resp.statusCode() < 500 || i == attempts - 1) {
                        throw new IOException("WandB " + path + " failed HTTP " + resp.statusCode()
                                + ": " + body);
                    }
                    if (retryBackoffMs > 0) Thread.sleep(retryBackoffMs * (1L << i));
                    continue;
                }
                if (resp.body() == null || resp.body().isBlank()) return new LinkedHashMap<>();
                Object decoded = Json.decode(resp.body());
                if (decoded instanceof Map) return (Map<String, Object>) decoded;
                Map<String, Object> wrap = new LinkedHashMap<>();
                wrap.put("value", decoded);
                return wrap;
            } catch (IOException e) {
                lastIo = e;
                if (i == attempts - 1) break;
                if (retryBackoffMs > 0) Thread.sleep(retryBackoffMs * (1L << i));
            } catch (InterruptedException e) {
                lastInterrupt = e;
                if (i == attempts - 1) break;
                if (retryBackoffMs > 0) Thread.sleep(retryBackoffMs * (1L << i));
            }
        }
        if (lastInterrupt != null) throw lastInterrupt;
        throw lastIo != null ? lastIo : new IOException("WandB " + path + " failed (unknown)");
    }

    /** Async variant that returns immediately and posts in the background. */
    public CompletableFuture<Void> postAsync(String path, Map<String, Object> payload) {
        requireRun();
        return CompletableFuture.runAsync(() -> {
            try {
                post(path, payload);
            } catch (Exception e) {
                if (raiseExceptions) throw new RuntimeException(e);
            }
        }, asyncExecutor);
    }

    /** Send a single log() call asynchronously (best-effort). */
    public CompletableFuture<Void> logAsync(Map<String, ? extends Number> metrics, long step) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("run_id", runId);
        payload.put("step", step);
        payload.put("metrics", metrics);
        payload.put("timestamp", Instant.now().toString());
        return postAsync("/metrics", payload);
    }

    @Override
    public void close() {
        if (!closedFlag.compareAndSet(false, true)) return;
        try { finish(); } catch (Exception ignored) { /* best-effort */ }
        if (ownAsyncExecutor) {
            asyncExecutor.shutdown();
            try {
                if (!asyncExecutor.awaitTermination(2, TimeUnit.SECONDS)) asyncExecutor.shutdownNow();
            } catch (InterruptedException ie) {
                asyncExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    // =========================================================================
    // Tensor helpers
    // =========================================================================

    static float[] toFloatArray(Tensor tensor) {
        if (tensor == null || !tensor.defined()) return new float[0];
        Tensor c = tensor.contiguous().cpu().to(org.bytedeco.pytorch.global.torch.kFloat()).flatten();
        long n = c.numel();
        float[] data = new float[(int) Math.min(n, Integer.MAX_VALUE)];
        org.bytedeco.javacpp.FloatPointer p = c.data_ptr_float();
        for (int i = 0; i < data.length; i++) data[i] = p.get(i);
        return data;
    }

    static double[][] tensorToMatrix(Tensor t) {
        if (t.dim() != 2) throw new IllegalArgumentException("expected 2D tensor");
        int rows = (int) t.size(0), cols = (int) t.size(1);
        float[] flat = toFloatArray(t);
        double[][] m = new double[rows][cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                m[r][c] = flat[r * cols + c];
        return m;
    }

    static final class ImageBuf {
        final byte[] png; final int height, width, channels;
        ImageBuf(byte[] png, int h, int w, int c) { this.png = png; height = h; width = w; channels = c; }
    }

    static ImageBuf tensorToPng(Tensor t) {
        long nd = t.dim();
        int c, h, w;
        float[] chw;
        if (nd == 2) {
            h = (int) t.size(0); w = (int) t.size(1); c = 1;
            chw = toFloatArray(t);
        } else if (nd == 3) {
            long d0 = t.size(0);
            if (d0 == 1 || d0 == 3 || d0 == 4) {
                c = (int) d0; h = (int) t.size(1); w = (int) t.size(2);
                chw = toFloatArray(t);
            } else {
                h = (int) d0; w = (int) t.size(1); c = (int) t.size(2);
                float[] hwc = toFloatArray(t);
                chw = new float[c * h * w];
                for (int ci = 0; ci < c; ci++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++)
                            chw[ci * h * w + hi * w + wi] = hwc[(hi * w + wi) * c + ci];
            }
        } else {
            throw new IllegalArgumentException("image tensor must be 2D/3D");
        }
        float[] hwc = new float[h * w * c];
        for (int ci = 0; ci < c; ci++)
            for (int hi = 0; hi < h; hi++)
                for (int wi = 0; wi < w; wi++)
                    hwc[(hi * w + wi) * c + ci] = chw[ci * h * w + hi * w + wi];
        int outC = c;
        if (c == 1) {
            float[] rgb = new float[h * w * 3];
            for (int i = 0; i < h * w; i++) {
                float v = hwc[i];
                rgb[i * 3] = v; rgb[i * 3 + 1] = v; rgb[i * 3 + 2] = v;
            }
            hwc = rgb; outC = 3;
        }
        return new ImageBuf(PngEncoder.encodeFloatHWC(hwc, h, w, outC), h, w, outC);
    }

    // =========================================================================
    // Builder
    // =========================================================================

    public static final class Builder {
        private String host = "localhost";
        private int port = 8080;
        private boolean useHttps = false;
        private Duration connectTimeout = Duration.ofSeconds(5);
        private String apiKey;
        private String entity = "local";
        private String project = "pytorch";
        private WandbLocalServer localServer;
        private Path runDir;
        // Enterprise enhancements
        private int maxRetries = 3;
        private long retryBackoffMs = 200L;
        private int imageGridMaxCols = 8;
        private boolean raiseExceptions = false;
        private ExecutorService asyncExecutor;

        public Builder host(String host) { this.host = host; return this; }
        public Builder port(int port) { this.port = port; return this; }
        public Builder useHttps(boolean v) { this.useHttps = v; return this; }
        public Builder connectTimeout(Duration d) { this.connectTimeout = d; return this; }
        public Builder apiKey(String apiKey) { this.apiKey = apiKey; return this; }
        public Builder entity(String entity) { this.entity = entity; return this; }
        public Builder project(String project) { this.project = project; return this; }
        /** Attach an in-process {@link WandbLocalServer} (offline mode). */
        public Builder offline(WandbLocalServer server) { this.localServer = server; return this; }
        public Builder runDir(Path dir) { this.runDir = dir; return this; }
        /** Convenience: also accept WANDB_API_KEY from the environment. */
        public Builder fromEnv() {
            String k = System.getenv("WANDB_API_KEY");
            if (k != null && !k.isBlank()) this.apiKey = k;
            String e = System.getenv("WANDB_ENTITY");
            if (e != null && !e.isBlank()) this.entity = e;
            String p = System.getenv("WANDB_PROJECT");
            if (p != null && !p.isBlank()) this.project = p;
            return this;
        }
        /** Max number of retries on transient (5xx, IOException) HTTP failures. Default 3. */
        public Builder maxRetries(int n) { this.maxRetries = Math.max(0, n); return this; }
        /** Base backoff between retries (exponential). Default 200ms. */
        public Builder retryBackoffMs(long ms) { this.retryBackoffMs = Math.max(0L, ms); return this; }
        /** Maximum columns when auto-building NCHW image grids. Default 8. */
        public Builder imageGridMaxCols(int n) { this.imageGridMaxCols = Math.max(1, n); return this; }
        /** When true, async failures are surfaced as RuntimeExceptions instead of swallowed. */
        public Builder raiseExceptions(boolean v) { this.raiseExceptions = v; return this; }
        /** Inject a custom executor for {@link #logAsync}/{@link #postAsync}. */
        public Builder asyncExecutor(ExecutorService ex) { this.asyncExecutor = ex; return this; }

        public WandbClient build() { return new WandbClient(this); }
    }
}
