package org.bytedeco.pytorch.plot.swanlab;

import org.bytedeco.pytorch.StringTensorDictItem;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.plot.wandb.WandbClient;
import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.plot.tensorboard.PngEncoder;

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
 * SwanLab-compatible experiment tracker for JavaCPP / LibTorch.
 *
 * <p>Mirrors the Python API surface ({@code swanlab.init / log / finish}) with
 * first-class support for scalars, heatmaps, images, text, audio, tables and
 * charts. Works against a remote SwanLab backend <em>or</em> an embedded
 * {@link SwanLabLocalServer} for fully-offline benchmarks (open the UI URL in
 * a browser to verify).
 *
 * <pre>{@code
 * try (SwanLabLocalServer server = SwanLabLocalServer.start(0);
 *      SwanLabClient swan = SwanLabClient.newBuilder()
 *              .offline(server)
 *              .project("demo")
 *              .experiment("exp1")
 *              .build()) {
 *     swan.init(Map.of("lr", 1e-3, "bs", 32));
 *     swan.log(Map.of("loss", 0.5, "acc", 0.8), 1);
 *     swan.logHeatmap("cm", matrix, 1);
 *     swan.logImage("sample", imageTensor, 1);
 *     System.out.println("open " + swan.uiUrl());
 * }
 * }</pre>
 */
public final class SwanLabClient implements AutoCloseable {

    private final HttpClient http;
    private final URI baseUri;
    private final String apiKey;
    private final String workspace;
    private final String project;
    private final String experiment;
    private final SwanLabLocalServer local;
    private final Path runDir;
    private String runId;
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

    private SwanLabClient(Builder b) {
        this.local = b.localServer;
        this.apiKey = b.apiKey == null ? "local-key" : b.apiKey;
        this.workspace = b.workspace == null ? "local" : b.workspace;
        this.project = Objects.requireNonNull(b.project, "project");
        this.experiment = b.experiment == null ? "exp" : b.experiment;
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
                Thread t = new Thread(r, "swanlab-async");
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
            // Default SwanLab cloud host; override via builder for self-hosted.
            this.baseUri = URI.create(String.format("%s://%s:%d/api/v1",
                    scheme, b.host, b.port));
        }
    }

    public static Builder newBuilder() { return new Builder(); }

    public String runId() { return runId; }
    public String project() { return project; }
    public String experiment() { return experiment; }
    public String workspace() { return workspace; }

    public String uiUrl() {
        if (local != null && runId != null) {
            return local.uiUrl() + "/experiments/" + workspace + "/" + project + "/" + runId;
        }
        return baseUri.resolve("../").toString();
    }

    // =========================================================================
    // Lifecycle  (swanlab.init / finish)
    // =========================================================================

    public void init() throws IOException, InterruptedException {
        init(Map.of());
    }

    public void init(Map<String, ?> config) throws IOException, InterruptedException {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("workspace", workspace);
        payload.put("project", project);
        payload.put("experiment", experiment);
        payload.put("config", config == null ? Map.of() : config);
        payload.put("created_at", Instant.now().toString());
        payload.put("framework", "javacpp-pytorch");
        Map<String, Object> resp = post("/experiments", payload);
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
        post("/experiments/" + runId, payload);
        finished = true;
    }

    // =========================================================================
    // log (scalars)
    // =========================================================================

    /** {@code swanlab.log({"loss": 0.1}, step=n)} */
    public void log(Map<String, ? extends Number> metrics, long step)
            throws IOException, InterruptedException {
        requireRun();
        stepCounter.set(Math.max(stepCounter.get(), step));
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("step", step);
        payload.put("metrics", metrics);
        payload.put("timestamp", Instant.now().toString());
        post("/logs", payload);
    }

    public void log(Map<String, ? extends Number> metrics) throws IOException, InterruptedException {
        log(metrics, stepCounter.incrementAndGet());
    }

    // =========================================================================
    // Charts / heatmap / histogram / scatter
    // =========================================================================

    public void logHeatmap(String name, double[][] matrix, long step)
            throws IOException, InterruptedException {
        logHeatmap(name, matrix, step, null);
    }

    public void logHeatmap(String name, double[][] matrix, long step, Map<String, Object> opts)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("type", "heatmap");
        payload.put("step", step);
        payload.put("matrix", matrix);
        if (opts != null) payload.put("opts", opts);
        post("/media/charts", payload);
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
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("type", "histogram");
        payload.put("step", step);
        payload.put("values", values);
        payload.put("bins", bins);
        post("/media/charts", payload);
    }

    /** Tensor-input convenience. */
    public void logHistogram(String name, Tensor t, int bins, long step)
            throws IOException, InterruptedException {
        logHistogram(name, toFloatArrayDouble(t), bins, step);
    }

    public void logHistogram(String name, Tensor t, long step)
            throws IOException, InterruptedException {
        logHistogram(name, t, 64, step);
    }

    /** 3D surface chart (height-field Z[x,y]). */
    public void logSurface(String name, double[][] Z, long step)
            throws IOException, InterruptedException {
        logSurface(name, Z, step, null);
    }

    public void logSurface(String name, double[][] Z, long step, Map<String, Object> opts)
            throws IOException, InterruptedException {
        requireRun();
        if (Z == null || Z.length == 0) throw new IllegalArgumentException("Z must be non-empty");
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("type", "surface");
        payload.put("step", step);
        payload.put("matrix", Z);
        if (opts != null) payload.put("opts", opts);
        post("/media/charts", payload);
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
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("type", "polar");
        payload.put("step", step);
        payload.put("points", points);
        if (opts != null) payload.put("opts", opts);
        post("/media/charts", payload);
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
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("type", "box");
        payload.put("step", step);
        payload.put("values", data);
        if (opts != null) payload.put("opts", opts);
        post("/media/charts", payload);
    }

    /** Image grid from NCHW batch (similar to wandb's grid). */
    public void logImages(String name, Tensor batchNchw, long step)
            throws IOException, InterruptedException {
        requireRun();
        if (batchNchw.dim() != 4) {
            logImage(name, batchNchw, step);
            return;
        }
        byte[] png = WandbClient.tensorBatchToGridPng(batchNchw, imageGridMaxCols);
        if (png == null) {
            logImage(name, batchNchw.select(0, 0), step);
            return;
        }
        logImage(name, png, step, Map.of("kind", "image_grid", "n", batchNchw.size(0)));
    }

    /** Upload an artifact (model / dataset / file). */
    public void logArtifact(String name, Path file, String type)
            throws IOException, InterruptedException {
        requireRun();
        Objects.requireNonNull(file, "file");
        if (!Files.exists(file)) throw new IOException("artifact file does not exist: " + file);
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("type", type == null ? "model" : type);
        payload.put("path", file.toString());
        payload.put("bytes", Files.size(file));
        post("/artifacts", payload);
        if (runDir != null) {
            Path dst = runDir.resolve(runId).resolve("artifacts");
            Files.createDirectories(dst);
            Files.copy(file, dst.resolve(file.getFileName()), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /**
     * Watch a model — capture histograms of all named parameters.
     * Returns the number of parameters captured.
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
                if (p == null || !p.defined() || p.numel() == 0) continue;
                String fullTag = (prefix == null || prefix.isEmpty() ? "" : prefix + "/") + item.key().toString();
                logHistogram(fullTag, p, 64, step);
                captured++;
            }
        } catch (Exception e) {
            if (raiseExceptions) throw new IOException("swanlab.watchModel failed: " + e.getMessage(), e);
        }
        return captured;
    }

    private static double[] toFloatArrayDouble(Tensor t) {
        if (t == null || !t.defined()) return new double[0];
        Tensor c = t.contiguous().cpu().to(org.bytedeco.pytorch.global.torch.kFloat()).flatten();
        long n = c.numel();
        double[] out = new double[(int) Math.min(n, Integer.MAX_VALUE)];
        org.bytedeco.javacpp.FloatPointer p = c.data_ptr_float();
        for (int i = 0; i < out.length; i++) out[i] = p.get(i);
        return out;
    }

    public void logScatter(String name, double[][] points, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("type", "scatter");
        payload.put("step", step);
        payload.put("points", points);
        post("/media/charts", payload);
    }

    public void logLineChart(String name, List<List<Double>> series, String[] legends, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("type", "line");
        payload.put("step", step);
        payload.put("series", series);
        if (legends != null) payload.put("legend", legends);
        post("/media/charts", payload);
    }

    public void logBar(String name, double[] values, String[] labels, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("type", "bar");
        payload.put("step", step);
        payload.put("values", values);
        if (labels != null) payload.put("labels", labels);
        post("/media/charts", payload);
    }

    // =========================================================================
    // Multimodal media
    // =========================================================================

    public void logImage(String name, byte[] pngBytes, long step, Map<String, Object> opts)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("step", step);
        payload.put("bytes", pngBytes);
        payload.put("format", "png");
        if (opts != null) payload.put("opts", opts);
        post("/media/images", payload);
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

    public void logText(String name, String text, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("step", step);
        payload.put("text", text);
        post("/media/text", payload);
    }

    public void logAudio(String name, float[] mono, int sampleRate, long step)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("step", step);
        payload.put("sample_rate", sampleRate);
        List<Double> samples = new ArrayList<>(Math.min(mono.length, 4000));
        int stride = Math.max(1, mono.length / 4000);
        for (int i = 0; i < mono.length; i += stride) samples.add((double) mono[i]);
        payload.put("waveform", samples);
        payload.put("n_samples", mono.length);
        post("/media/audio", payload);
    }

    public void logAudio(String name, Tensor waveform, int sampleRate, long step)
            throws IOException, InterruptedException {
        logAudio(name, toFloatArray(waveform), sampleRate, step);
    }

    public void logTable(String name, String[] columns, List<String[]> rows)
            throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("name", name);
        payload.put("columns", columns);
        payload.put("rows", rows);
        post("/media/tables", payload);
    }

    public void logSummary(Map<String, ?> summary) throws IOException, InterruptedException {
        requireRun();
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("summary", summary);
        post("/summary", payload);
    }

    // =========================================================================
    // HTTP
    // =========================================================================

    private void requireRun() {
        if (runId == null) throw new IllegalStateException("call init() first");
        if (closedFlag.get()) throw new IllegalStateException("client is closed");
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> post(String path, Map<String, Object> payload)
            throws IOException, InterruptedException {
        String json = Json.encode(payload);
        String p = path.startsWith("/") ? path.substring(1) : path;
        HttpRequest req = HttpRequest.newBuilder(baseUri.resolve(p))
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
                    if (resp.statusCode() < 500 || i == attempts - 1) {
                        throw new IOException("SwanLab " + path + " failed HTTP " + resp.statusCode()
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
        throw lastIo != null ? lastIo : new IOException("SwanLab " + path + " failed (unknown)");
    }

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

    public CompletableFuture<Void> logAsync(Map<String, ? extends Number> metrics, long step) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("experiment_id", runId);
        payload.put("step", step);
        payload.put("metrics", metrics);
        payload.put("timestamp", Instant.now().toString());
        return postAsync("/logs", payload);
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
    // Tensor helpers (shared pattern with WandB)
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
        private String host = "api.swanlab.cn";
        private int port = 443;
        private boolean useHttps = true;
        private Duration connectTimeout = Duration.ofSeconds(5);
        private String apiKey;
        private String workspace = "local";
        private String project = "pytorch";
        private String experiment = "exp";
        private SwanLabLocalServer localServer;
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
        public Builder workspace(String workspace) { this.workspace = workspace; return this; }
        public Builder project(String project) { this.project = project; return this; }
        public Builder experiment(String experiment) { this.experiment = experiment; return this; }
        public Builder offline(SwanLabLocalServer server) {
            this.localServer = server;
            this.useHttps = false;
            return this;
        }
        public Builder runDir(Path dir) { this.runDir = dir; return this; }
        public Builder fromEnv() {
            String k = System.getenv("SWANLAB_API_KEY");
            if (k != null && !k.isBlank()) this.apiKey = k;
            String w = System.getenv("SWANLAB_WORKSPACE");
            if (w != null && !w.isBlank()) this.workspace = w;
            String p = System.getenv("SWANLAB_PROJECT");
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

        public SwanLabClient build() { return new SwanLabClient(this); }
    }
}
