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
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.transformers.image_processing;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.transformers.utils.T;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HuggingFace {@code image_processing_utils.ImageProcessor} port.
 *
 * <p>Subclass of {@link BaseImageProcessor} that adds:
 * <ul>
 *   <li>JSON-based configuration loading ({@link #fromPretrained(Path)})</li>
 *   <li>Batch processing ({@link #process(List)}) returning the canonical
 *       {@code ``{pixel_values, image_sizes}``} map</li>
 *   <li>Per-call kwargs overrides</li>
 * </ul>
 */
public class ImageProcessor extends BaseImageProcessor {

    public ImageProcessor() {
        super();
    }

    public ImageProcessor(Map<String, Integer> size,
                          int resample,
                          boolean doResize,
                          boolean doNormalize,
                          float[] imageMean,
                          float[] imageStd) {
        super(size, resample, doResize, doNormalize, imageMean, imageStd);
    }

    /**
     * Load the image processor configuration from a local HF snapshot
     * directory containing {@code preprocessor_config.json}.
     *
     * <p>The JSON parsing is intentionally lightweight: we read the file as
     * text, scan for the recognized keys, and ignore everything else.
     * Recognized keys mirror {@code transformers.image_processing_utils}:
     * {@code size}, {@code resample}, {@code do_resize}, {@code do_normalize},
     * {@code image_mean}, {@code image_std}, {@code do_rescale},
     * {@code rescale_factor}.
     */
    public static ImageProcessor fromPretrained(Path dir) throws IOException {
        Objects.requireNonNull(dir, "dir");
        Path json = dir.resolve("preprocessor_config.json");
        if (!Files.isRegularFile(json)) {
            // No preprocessor_config.json: return defaults.
            return new ImageProcessor();
        }
        String text = new String(Files.readAllBytes(json), "UTF-8");
        Map<String, Integer> size = parseSize(text);
        int resample = parseInt(text, "resample", DEFAULT_RESAMPLE);
        boolean doResize = parseBool(text, "do_resize", true);
        boolean doNormalize = parseBool(text, "do_normalize", true);
        float[] mean = parseFloatArray(text, "image_mean", DEFAULT_MEAN);
        float[] std = parseFloatArray(text, "image_std", DEFAULT_STD);
        return new ImageProcessor(size, resample, doResize, doNormalize, mean, std);
    }

    /** Alias matching HF Python naming. */
    public static ImageProcessor from_pretrained(Path dir) throws IOException {
        return fromPretrained(dir);
    }

    // ---------------------------------------------------------------------
    // Processing entry points
    // ---------------------------------------------------------------------

    @Override
    public Tensor process(Tensor image, Map<String, Object> kwargs) {
        if (kwargs != null) {
            applyKwargs(kwargs);
        }
        return to_pixel_values(image);
    }

    /**
     * Process a batch of images. Returns a map with:
     * <ul>
     *   <li>{@code pixel_values}: NCHW tensor</li>
     *   <li>{@code image_sizes}: N×2 long tensor of (height, width) pairs</li>
     * </ul>
     */
    public Map<String, Tensor> process(List<Tensor> images) {
        return process(images, java.util.Collections.emptyMap());
    }

    public Map<String, Tensor> process(List<Tensor> images, Map<String, Object> kwargs) {
        Objects.requireNonNull(images, "images");
        if (kwargs != null) applyKwargs(kwargs);
        List<Tensor> processed = new ArrayList<>(images.size());
        long[][] sizes = new long[images.size()][2];
        for (int i = 0; i < images.size(); i++) {
            Tensor img = images.get(i);
            long h = img.size(img.dim() - 2);
            long w = img.size(img.dim() - 1);
            sizes[i][0] = h;
            sizes[i][1] = w;
            processed.add(to_pixel_values(img));
        }
        Tensor pixelValues = T.stack(processed.toArray(new Tensor[0]));
        Tensor imageSizes = torch.tensor(flatten(sizes)).reshape(images.size(), 2);
        Map<String, Tensor> out = new LinkedHashMap<>();
        out.put("pixel_values", pixelValues);
        out.put("image_sizes", imageSizes);
        return out;
    }

    // ---------------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------------

    private void applyKwargs(Map<String, Object> kwargs) {
        Object s = kwargs.get("size");
        if (s instanceof Map<?, ?> m) {
            Map<String, Integer> typed = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : m.entrySet()) {
                typed.put(e.getKey().toString(), ((Number) e.getValue()).intValue());
            }
            size(typed);
        }
        Object res = kwargs.get("resample");
        if (res instanceof Number n) resample(n.intValue());
        Object dr = kwargs.get("do_resize");
        if (dr instanceof Boolean b) doResize(b);
        Object dn = kwargs.get("do_normalize");
        if (dn instanceof Boolean b) doNormalize(b);
        Object im = kwargs.get("image_mean");
        if (im instanceof float[] arr) imageMean(arr);
        else if (im instanceof double[] arr) {
            float[] f = new float[arr.length];
            for (int i = 0; i < arr.length; i++) f[i] = (float) arr[i];
            imageMean(f);
        }
        Object isd = kwargs.get("image_std");
        if (isd instanceof float[] arr) imageStd(arr);
        else if (isd instanceof double[] arr) {
            float[] f = new float[arr.length];
            for (int i = 0; i < arr.length; i++) f[i] = (float) arr[i];
            imageStd(f);
        }
    }

    // ----- JSON parsing helpers (string-scan, no third-party deps) -----

    private static Map<String, Integer> parseSize(String json) {
        Map<String, Integer> out = new LinkedHashMap<>();
        // size can be either {"height": 224, "width": 224} or {"shortest_edge": 224}
        java.util.regex.Pattern block = java.util.regex.Pattern.compile(
                "\"size\"\\s*:\\s*\\{([^}]*)\\}", java.util.regex.Pattern.DOTALL);
        java.util.regex.Matcher m = block.matcher(json);
        if (!m.find()) {
            // Try single int form: "size": 224
            java.util.regex.Pattern one = java.util.regex.Pattern.compile("\"size\"\\s*:\\s*(\\d+)");
            java.util.regex.Matcher m2 = one.matcher(json);
            if (m2.find()) {
                int v = Integer.parseInt(m2.group(1));
                out.put("height", v);
                out.put("width", v);
            }
            return out;
        }
        java.util.regex.Pattern kv = java.util.regex.Pattern.compile(
                "\"(\\w+)\"\\s*:\\s*(\\d+)");
        java.util.regex.Matcher m2 = kv.matcher(m.group(1));
        while (m2.find()) {
            out.put(m2.group(1), Integer.parseInt(m2.group(2)));
        }
        if (!out.containsKey("height") && !out.containsKey("width")) {
            // shortest_edge → square
            int se = out.getOrDefault("shortest_edge", 224);
            out.put("height", se);
            out.put("width", se);
        }
        return out;
    }

    private static int parseInt(String json, String key, int defaultValue) {
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(
                "\"" + key + "\"\\s*:\\s*(\\d+)");
        java.util.regex.Matcher m = p.matcher(json);
        return m.find() ? Integer.parseInt(m.group(1)) : defaultValue;
    }

    private static boolean parseBool(String json, String key, boolean defaultValue) {
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(
                "\"" + key + "\"\\s*:\\s*(true|false)");
        java.util.regex.Matcher m = p.matcher(json);
        if (!m.find()) return defaultValue;
        return Boolean.parseBoolean(m.group(1));
    }

    private static float[] parseFloatArray(String json, String key, float[] defaultValue) {
        java.util.regex.Pattern p = java.util.regex.Pattern.compile(
                "\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
        java.util.regex.Matcher m = p.matcher(json);
        if (!m.find()) return defaultValue.clone();
        String body = m.group(1);
        String[] parts = body.split(",");
        float[] out = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            String t = parts[i].trim();
            if (t.isEmpty()) return defaultValue.clone();
            out[i] = Float.parseFloat(t);
        }
        return out;
    }

    private static long[] flatten(long[][] m) {
        long[] out = new long[m.length * m[0].length];
        for (int i = 0; i < m.length; i++) {
            System.arraycopy(m[i], 0, out, i * m[i].length, m[i].length);
        }
        return out;
    }
}