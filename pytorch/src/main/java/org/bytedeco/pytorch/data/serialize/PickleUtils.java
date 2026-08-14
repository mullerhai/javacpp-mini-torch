package org.bytedeco.pytorch.data.serialize;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.io.*;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;

/**
 * Utility class for loading Python pickle files.
 * 
 * <p>This is a pure-Java implementation that does not depend on Razorvine.
 * It handles standalone .pkl files and PyTorch model checkpoints.</p>
 */
public class PickleUtils {

    /**
     * Load a standalone Python pickle file using our pure-Java unpickler.
     *
     * @param file The .pkl file to load
     * @return The unpickled object (typically a Map for state_dict)
     */
    public static Object loadPickle(File file) throws IOException {
        if (file == null || !file.isFile()) {
            throw new IOException("not a file: " + file);
        }

        byte[] data = Files.readAllBytes(file.toPath());
        UnpickleReader reader = new UnpickleReader(data);
        return reader.load();
    }

    /**
     * Load a standalone Python pickle file using our pure-Java unpickler.
     */
    public static Object loadPickle(byte[] data) throws IOException {
        UnpickleReader reader = new UnpickleReader(data);
        return reader.load();
    }

    /**
     * Load a standalone pickle and extract state_dict tensors.
     *
     * @param file The .pkl file to load
     * @return Map from parameter name to Tensor
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Tensor> loadPickleStateDict(File file) throws IOException {
        Object root = loadPickle(file);
        return extractStateDict(root);
    }

    @SuppressWarnings("unchecked")
    static Map<String, Tensor> extractStateDict(Object root) {
        if (root == null) return Map.of();
        if (root instanceof Map) {
            Map<?, ?> m = (Map<?, ?>) root;
            // Prefer common checkpoint wrappers
            for (String key : new String[]{"model_state_dict", "state_dict", "model", "module", "net"}) {
                Object v = m.get(key);
                if (v instanceof Map && looksLikeStateDict((Map<?, ?>) v)) {
                    return toTensorMap((Map<?, ?>) v);
                }
            }
            if (looksLikeStateDict(m)) {
                return toTensorMap(m);
            }
        }
        return Map.of();
    }

    static boolean looksLikeStateDict(Map<?, ?> m) {
        if (m.isEmpty()) return false;
        int tensors = 0, keys = 0;
        for (Map.Entry<?, ?> e : m.entrySet()) {
            if (!(e.getKey() instanceof String)) return false;
            keys++;
            if (e.getValue() instanceof Tensor) tensors++;
        }
        return keys > 0 && tensors * 2 >= keys;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Tensor> toTensorMap(Map<?, ?> m) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : m.entrySet()) {
            if (e.getKey() instanceof String && e.getValue() instanceof Tensor) {
                out.put((String) e.getKey(), (Tensor) e.getValue());
            }
        }
        return out;
    }

    /**
     * Detect if data looks like a pickle file.
     */
    public static boolean isPickle(byte[] data) {
        if (data == null || data.length < 2) return false;
        
        // Protocol 2+: starts with PROTO (0x80)
        if ((data[0] & 0xff) == 0x80) {
            return data.length >= 2 && (data[1] & 0xff) >= 2 && (data[1] & 0xff) <= 5;
        }
        
        // Protocol 0/1: starts with marker or common opcodes
        int first = data[0] & 0xff;
        return first == '(' || first == 'd' || first == '}' || first == 'l' || first == ']'
                || first == '(' || first == '}' || first == ']' || first == 'l';
    }

    /**
     * Check if file is a pickle.
     */
    public static boolean isPickle(File file) throws IOException {
        if (file == null || !file.isFile() || file.length() < 2) return false;
        try (InputStream in = new FileInputStream(file)) {
            int b0 = in.read();
            if (b0 < 0) return false;
            int b1 = in.read();
            if (b1 < 0) return false;
            
            // Protocol 2+
            if (b0 == 0x80 && b1 >= 2 && b1 <= 5) return true;
            
            // Protocol 0/1
            return b0 == '(' || b0 == 'd' || b0 == '}' || b0 == 'l' || b0 == ']';
        }
    }
}
