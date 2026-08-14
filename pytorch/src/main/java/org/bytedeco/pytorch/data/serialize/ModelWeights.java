package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.safetensors.LoadOptions;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.safetensors.ShardedSafeTensors;
import org.bytedeco.pytorch.nn.Module;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Auto-detect weight file format and load as {@code Map&lt;String, Tensor&gt;}.
 *
 * <p>Supported:
 * <ul>
 *   <li>{@code .safetensors} — native JavaCPP path ({@link SafeTensors})</li>
 *   <li>{@code .pth} / {@code .pt} — Python {@code torch.save} ZIP → {@link TorchPthReader}</li>
 *   <li>HF model directories / {@code model.safetensors.index.json} → {@link ShardedSafeTensors}</li>
 *   <li>magic-byte sniff when extension is missing/wrong</li>
 * </ul>
 *
 * <p>Optional: convert Python checkpoints to safetensors next to the source so
 * subsequent loads skip pickle entirely. Honours {@link LoadOptions}
 * ({@code weights_only}, {@code map_location}, {@code strict}, zero-copy, dtype).
 */
public final class ModelWeights {
    public enum Format { SAFETENSORS, TORCH_PTH_ZIP, PICKLE, BIN_MICROLENS, BIN_NAMED, BIN_GENERIC, UNKNOWN }

    private ModelWeights() {}

    public static Format detect(File file) throws IOException {
        if (file == null || !file.isFile()) return Format.UNKNOWN;
        String name = file.getName().toLowerCase(Locale.ROOT);
        if (name.endsWith(".safetensors")) return Format.SAFETENSORS;
        if (name.endsWith(".pkl") || name.endsWith(".pickle")) {
            // Standalone pickle files - will be handled by TorchPthReader.loadPickleStateDict
            if (TorchPthReader.isStandalonePickle(file)) {
                return Format.PICKLE;
            }
            return Format.UNKNOWN;
        }
        if (name.endsWith(".pth") || name.endsWith(".pt") || name.endsWith(".bin")) {
            if (TorchPthReader.isZipTorch(file)) return Format.TORCH_PTH_ZIP;
            // .bin is often raw pytorch_model.bin (also zip torch) or something else
            if (isSafetensorsMagic(file)) return Format.SAFETENSORS;
            // Detect custom binary formats for .bin files
            if (name.endsWith(".bin")) {
                return detectBinFormat(file);
            }
            return Format.UNKNOWN;
        }
        if (isSafetensorsMagic(file)) return Format.SAFETENSORS;
        if (TorchPthReader.isZipTorch(file)) return Format.TORCH_PTH_ZIP;
        return Format.UNKNOWN;
    }

    /**
     * Detect specific .bin format based on file structure.
     */
    private static Format detectBinFormat(File file) throws IOException {
        byte[] magic = new byte[8];
        try (InputStream in = Files.newInputStream(file.toPath())) {
            int read = in.read(magic);
            if (read < 4) return Format.UNKNOWN;

            // Check for our custom binary format (MicroLens) - "MLNS" magic
            if (magic[0] == 'M' && magic[1] == 'L' && magic[2] == 'N' && magic[3] == 'S') {
                return Format.BIN_MICROLENS;
            }

            // Check for pickle protocol
            if (magic[0] >= 0x80 && magic[0] <= 0x8F) {
                return Format.PICKLE;
            }

            // Check for named-tensor binary format
            if (looksLikeNamedBinHeader(file)) {
                return Format.BIN_NAMED;
            }

            // Default: treat as generic binary state-dict
            return Format.BIN_GENERIC;
        }
    }

    /**
     * Check if the file starts with a small int32 name length followed by an ASCII
     * string (named-tensor binary format).
     */
    private static boolean looksLikeNamedBinHeader(File file) throws IOException {
        if (file.length() < 8) return false;
        try (InputStream in = Files.newInputStream(file.toPath())) {
            byte[] m4 = new byte[4];
            if (in.read(m4) < 4) return false;
            int firstInt = ((m4[0] & 0xff))
                     | ((m4[1] & 0xff) << 8)
                     | ((m4[2] & 0xff) << 16)
                     | ((m4[3] & 0xff) << 24);
            // Reject obvious non-format magic bytes
            if (firstInt == 'P' | (firstInt & 0xff) == 'M' || firstInt < 0) return false;
            int nameLen = firstInt;
            if (nameLen <= 0 || nameLen > 256 || nameLen > file.length() - 4) return false;
            // Read name + 8 trailing bytes (dtype + ndims)
            byte[] rest = new byte[nameLen + 8];
            if (in.read(rest) < rest.length) return false;
            for (int i = 0; i < nameLen; i++) {
                int b = rest[i] & 0xff;
                if (b < 0x20 || b > 0x7e) return false;
            }
            int dtype = ((rest[nameLen] & 0xff))
                      | ((rest[nameLen + 1] & 0xff) << 8)
                      | ((rest[nameLen + 2] & 0xff) << 16)
                      | ((rest[nameLen + 3] & 0xff) << 24);
            int ndims = ((rest[nameLen + 4] & 0xff))
                      | ((rest[nameLen + 5] & 0xff) << 8)
                      | ((rest[nameLen + 6] & 0xff) << 16)
                      | ((rest[nameLen + 7] & 0xff) << 24);
            return dtype >= 0 && dtype <= 4 && ndims >= 0 && ndims <= 8;
        }
    }

    public static Format detect(Path path) throws IOException {
        return detect(path.toFile());
    }

    /**
     * Load tensors from a single weight file (auto format).
     */
    public static Map<String, Tensor> load(File file) throws IOException {
        return load(file, true);
    }

    /**
     * @param convertPthToSafe when true and input is a ZIP .pth, also write a
     *                         sibling {@code .safetensors} for faster reloads
     */
    public static Map<String, Tensor> load(File file, boolean convertPthToSafe) throws IOException {
        return load(file, convertPthToSafe, LoadOptions.defaults());
    }

    /**
     * Full options path — {@code torch.load}-compatible.
     * When {@code opts.weightsOnly} is true (or always for this method), returns
     * only the tensor map. Directories / index.json are accepted.
     */
    public static Map<String, Tensor> load(File file, boolean convertPthToSafe, LoadOptions opts)
            throws IOException {
        Objects.requireNonNull(file, "file");
        if (opts == null) opts = LoadOptions.defaults();

        if (file.isDirectory()) {
            return loadFromDirectory(file.toPath(), convertPthToSafe, opts);
        }
        String lower = file.getName().toLowerCase(Locale.ROOT);
        if (lower.endsWith("index.json")) {
            return ShardedSafeTensors.loadIndex(file.toPath(), opts);
        }

        Format fmt = detect(file);
        switch (fmt) {
            case SAFETENSORS:
                return SafeTensors.loadFile(file, opts);
            case TORCH_PTH_ZIP: {
                Map<String, Tensor> sd = TorchPthReader.loadStateDict(file);
                if (convertPthToSafe && !sd.isEmpty()) {
                    File out = PthToSafeTensors.defaultOutput(file);
                    if (!out.exists() || out.lastModified() < file.lastModified()) {
                        try {
                            PthToSafeTensors.convert(file, out);
                        } catch (Exception ignored) {
                            // conversion is best-effort cache; load still returns tensors
                        }
                    }
                }
                return SafeTensors.applyMapLocation(sd, opts);
            }
            case PICKLE:
                // Standalone Python pickle - delegate to TorchPthReader
                return TorchPthReader.loadPickleStateDict(file);
            case BIN_MICROLENS:
            case BIN_NAMED:
            case BIN_GENERIC:
                // Custom binary formats - delegate to PyTorchModelLoader
                return PyTorchModelLoader.loadBin(file, opts);
            default:
                throw new IOException("Unrecognized weight format: " + file
                    + " (expected .safetensors, torch ZIP .pth/.pt, pickle .pkl, or binary .bin)");
        }
    }

    /** {@code torch.load(path, map_location=..., weights_only=True)} style. */
    public static Map<String, Tensor> load(File file, LoadOptions opts) throws IOException {
        return load(file, /*convertPthToSafe=*/true, opts);
    }

    public static Map<String, Tensor> load(File file, Device mapLocation, boolean weightsOnly)
            throws IOException {
        return load(file, true, LoadOptions.builder()
                .mapLocation(mapLocation)
                .weightsOnly(weightsOnly)
                .build());
    }

    public static Map<String, Tensor> load(Path path) throws IOException {
        return load(path.toFile());
    }

    public static Map<String, Tensor> load(Path path, LoadOptions opts) throws IOException {
        return load(path.toFile(), opts);
    }

    public static Map<String, Tensor> load(String path) throws IOException {
        return load(new File(path));
    }

    public static Map<String, Tensor> load(String path, LoadOptions opts) throws IOException {
        return load(new File(path), opts);
    }

    /**
     * Load and inject into a module. Auto-detects format.
     *
     * @return parameters written
     */
    public static int loadIntoModule(Module module, File file, boolean strict) throws IOException {
        Map<String, Tensor> w = load(file, true, LoadOptions.builder().strict(strict).build());
        return SafeTensors.loadIntoModule(module, w, strict);
    }

    public static int loadIntoModule(Module module, File file, LoadOptions opts) throws IOException {
        if (opts == null) opts = LoadOptions.defaults();
        Map<String, Tensor> w = load(file, true, opts);
        return SafeTensors.loadIntoModule(module, w, opts.strict);
    }

    /**
     * Auto-detect format, load tensors, and build a trainable typed
     * {@link WeightBagModule}: nested hierarchy + real Linear/Embedding/…
     * leaves with hyperparameters inferred from shapes and names.
     *
     * <p>This is the primary path for "arbitrary safetensors / .pth → Module
     * so we can fine-tune": no architecture class required, structure and
     * layer names match the Python state-dict.
     *
     * <pre>{@code
     *   WeightBagModule bag = ModelWeights.toModule("model.safetensors");
     *   // or: ModelWeights.toModule("model.pth")
     *   bag.freezePrefix("embedding_layer.");
     *   Adam opt = new Adam(bag.parameters(), new AdamOptions(1e-3));
     *   bag.saveSafetensors(new File("finetuned.safetensors"));
     * }</pre>
     */
    public static WeightBagModule toModule(File file) throws IOException {
        return toModule(file, true);
    }

    public static WeightBagModule toModule(File file, boolean requiresGrad) throws IOException {
        return toModule(file, requiresGrad, LoadOptions.defaults());
    }

    public static WeightBagModule toModule(File file, boolean requiresGrad, LoadOptions opts)
            throws IOException {
        // Delegate to WeightBagModule loaders (structure meta + Sequential gap-fill + opts)
        return WeightBagModule.fromFile(file, requiresGrad, opts);
    }

    public static WeightBagModule toModule(File file, boolean requiresGrad,
                                            Device mapLocation, boolean strict) throws IOException {
        return toModule(file, requiresGrad, LoadOptions.builder()
                .mapLocation(mapLocation)
                .strict(strict)
                .build());
    }

    public static WeightBagModule toModule(Path path) throws IOException {
        return toModule(path.toFile());
    }

    public static WeightBagModule toModule(Path path, LoadOptions opts) throws IOException {
        return toModule(path.toFile(), true, opts);
    }

    public static WeightBagModule toModule(String path) throws IOException {
        return toModule(new File(path));
    }

    public static WeightBagModule toModule(String path, LoadOptions opts) throws IOException {
        return toModule(new File(path), true, opts);
    }

    /**
     * Scan directory for weights (prefer safetensors) and build a bag Module.
     */
    public static WeightBagModule toModuleFromDirectory(Path dir) throws IOException {
        return toModuleFromDirectory(dir, true);
    }

    public static WeightBagModule toModuleFromDirectory(Path dir, boolean requiresGrad)
            throws IOException {
        return toModuleFromDirectory(dir, requiresGrad, LoadOptions.defaults());
    }

    public static WeightBagModule toModuleFromDirectory(Path dir, boolean requiresGrad,
                                                         LoadOptions opts) throws IOException {
        if (opts == null) opts = LoadOptions.defaults();
        // Prefer sharded HF layout (index + shards) when present
        try {
            if (Files.isDirectory(dir) && !ShardedSafeTensors.resolveShards(dir).isEmpty()) {
                return WeightBagModule.fromSafetensors(dir.toFile(), requiresGrad, opts);
            }
        } catch (IOException ignored) {}
        Map<String, Tensor> w = loadFromDirectory(dir, true, opts);
        return WeightBagModule.fromTyped(w, requiresGrad);
    }

    /**
     * Scan a directory for weight files in preference order:
     * model.safetensors.index.json → model.safetensors → *.safetensors shards
     * → model.pth / pytorch_model.bin / *.pth.
     */
    public static Map<String, Tensor> loadFromDirectory(Path dir) throws IOException {
        return loadFromDirectory(dir, true);
    }

    public static Map<String, Tensor> loadFromDirectory(Path dir, boolean convertPthToSafe)
            throws IOException {
        return loadFromDirectory(dir, convertPthToSafe, LoadOptions.defaults());
    }

    public static Map<String, Tensor> loadFromDirectory(Path dir, boolean convertPthToSafe,
                                                         LoadOptions opts) throws IOException {
        if (dir == null || !Files.isDirectory(dir)) {
            throw new IOException("not a directory: " + dir);
        }
        if (opts == null) opts = LoadOptions.defaults();

        // Prefer HF index / numbered shards via ShardedSafeTensors
        try {
            List<Path> shards = ShardedSafeTensors.resolveShards(dir);
            if (!shards.isEmpty()) {
                return ShardedSafeTensors.loadDirectory(dir, opts);
            }
        } catch (IOException ignored) {
            // fall through to legacy scan
        }

        // Prefer single safetensors
        Path single = dir.resolve("model.safetensors");
        if (Files.isRegularFile(single)) {
            return SafeTensors.loadFile(single.toFile(), opts);
        }
        List<Path> safes = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir, "*.safetensors")) {
            for (Path p : ds) {
                String n = p.getFileName().toString();
                if (n.endsWith(".partial")) continue;
                safes.add(p);
            }
        }
        if (!safes.isEmpty()) {
            safes.sort(Path::compareTo);
            Map<String, Tensor> all = new LinkedHashMap<>();
            for (Path p : safes) {
                all.putAll(SafeTensors.loadAsTensors(p.toFile(), opts.zeroCopy));
            }
            if (opts.dequantFp8) {
                all = ShardedSafeTensors.tryDequantFp8(all);
            }
            return SafeTensors.applyMapLocation(all, opts);
        }
        // Fall back to .pth / .pt / pytorch_model.bin
        for (String name : new String[]{
            "model.pth", "pytorch_model.bin", "model.pt", "checkpoint.pth", "weights.pth"
        }) {
            Path p = dir.resolve(name);
            if (Files.isRegularFile(p)) {
                return load(p.toFile(), convertPthToSafe, opts);
            }
        }
        List<Path> pths = new ArrayList<>();
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir)) {
            for (Path p : ds) {
                String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
                if (n.endsWith(".pth") || n.endsWith(".pt") || n.endsWith(".bin")) {
                    pths.add(p);
                }
            }
        }
        if (!pths.isEmpty()) {
            pths.sort(Path::compareTo);
            // merge all (unusual but useful for multi-file dumps)
            Map<String, Tensor> all = new LinkedHashMap<>();
            for (Path p : pths) {
                try {
                    all.putAll(load(p.toFile(), convertPthToSafe, opts));
                } catch (IOException ignored) { /* skip non-torch bins */ }
            }
            if (!all.isEmpty()) return all;
        }
        throw new IOException("No loadable weights (.safetensors / .pth / .pt) in " + dir);
    }

    private static boolean isSafetensorsMagic(File file) throws IOException {
        // safetensors starts with u64 little-endian header length — not a stable magic,
        // but files are never ZIP PK\x03\x04. Heuristic: extension or non-zip + readable header.
        if (TorchPthReader.isZipTorch(file)) return false;
        if (file.length() < 16) return false;
        try (InputStream in = Files.newInputStream(file.toPath())) {
            byte[] b = in.readNBytes(8);
            if (b.length < 8) return false;
            long headerLen = 0;
            for (int i = 0; i < 8; i++) headerLen |= ((long) (b[i] & 0xFF)) << (8 * i);
            // reasonable header: 2 bytes .. 100 MB
            return headerLen >= 2 && headerLen < 100_000_000L && headerLen + 8 < file.length();
        }
    }
}
