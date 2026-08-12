/*
 * Minimal YAML 1.1 subset for Docker Compose and Kubernetes manifests.
 *
 * No SnakeYAML / Jackson — hand-rolled like utils.json.Json.
 * Supports: block maps, block lists, scalars (quoted/plain), comments, multi-doc (---).
 * Does NOT support: anchors/aliases, complex tags, flow sequences beyond simple [a, b],
 * full folded blocks. Enough for model-service deploy configs.
 */
package org.bytedeco.pytorch.utils.yaml;

import java.io.IOException;
import java.lang.reflect.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.*;

/**
 * Encode / decode a practical YAML subset used by Compose + K8s.
 *
 * <pre>{@code
 * Map<String, Object> doc = Yaml.load("""
 *   apiVersion: apps/v1
 *   kind: Deployment
 *   metadata:
 *     name: ranker
 *   spec:
 *     replicas: 2
 *     """);
 * String out = Yaml.dump(doc);
 * List<Object> all = Yaml.loadAll(multiDocText);
 * }</pre>
 *
 * <p>Enterprise features added:
 * <ul>
 *   <li>Type-safe POJO binding: {@code Yaml.loadAs(path, MyConfig.class)}</li>
 *   <li>Path-based navigation: {@code Yaml.get(doc, "/spec/replicas")}</li>
 *   <li>Deep merge: {@code Yaml.merge(base, override)}</li>
 *   <li>Environment expansion: {@code Yaml.expandEnv("${VAR:-default}")}</li>
 *   <li>Schema validation: {@code Yaml.validate(doc, schema)}</li>
 *   <li>Diff/Patch: {@code Yaml.diff(base, override)}</li>
 * </ul>
 */
public final class Yaml {

    private Yaml() {}

    // =========================================================================
    // Public load / dump (legacy API)
    // =========================================================================

    public static Object load(String text) throws IOException {
        List<Object> docs = loadAll(text);
        if (docs.isEmpty()) return null;
        return docs.get(0);
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> loadMap(String text) throws IOException {
        Object v = load(text);
        if (v == null) return new LinkedHashMap<>();
        if (v instanceof Map) return (Map<String, Object>) v;
        throw new IOException("expected YAML mapping, got " + v.getClass().getSimpleName());
    }

    public static List<Object> loadAll(String text) throws IOException {
        if (text == null || text.isBlank()) return List.of();
        Parser p = new Parser(text);
        return p.parseDocuments();
    }

    public static Object load(Path path) throws IOException {
        return load(Files.readString(path, StandardCharsets.UTF_8));
    }

    public static Map<String, Object> loadMap(Path path) throws IOException {
        return loadMap(Files.readString(path, StandardCharsets.UTF_8));
    }

    public static List<Object> loadAll(Path path) throws IOException {
        return loadAll(Files.readString(path, StandardCharsets.UTF_8));
    }

    public static String dump(Object value) {
        StringBuilder sb = new StringBuilder(256);
        Dumper d = new Dumper(sb);
        d.dumpDocument(value);
        return sb.toString();
    }

    /** Dump multiple documents separated by {@code ---}. */
    public static String dumpAll(Collection<?> documents) {
        if (documents == null || documents.isEmpty()) return "";
        StringBuilder sb = new StringBuilder(512);
        boolean first = true;
        for (Object doc : documents) {
            if (!first) sb.append("---\n");
            first = false;
            Dumper d = new Dumper(sb);
            d.dumpDocument(doc);
            if (sb.length() > 0 && sb.charAt(sb.length() - 1) != '\n') sb.append('\n');
        }
        return sb.toString();
    }

    public static void dump(Path path, Object value) throws IOException {
        Files.writeString(path, dump(value), StandardCharsets.UTF_8);
    }

    public static void dumpAll(Path path, Collection<?> documents) throws IOException {
        Files.writeString(path, dumpAll(documents), StandardCharsets.UTF_8);
    }

    // =========================================================================
    // POJO binding
    // =========================================================================

    /**
     * Load YAML and bind to a POJO class.
     */
    public static <T> T loadAs(String text, Class<T> clazz) throws IOException {
        Object doc = load(text);
        return bindTo(doc, clazz);
    }

    /**
     * Load YAML file and bind to a POJO class.
     */
    public static <T> T loadAs(Path path, Class<T> clazz) throws IOException {
        Object doc = load(path);
        return bindTo(doc, clazz);
    }

    private static <T> T bindTo(Object doc, Class<T> clazz) throws IOException {
        if (doc == null) {
            try {
                return clazz.newInstance();
            } catch (Exception e) {
                throw new IOException("Cannot instantiate " + clazz.getName(), e);
            }
        }
        if (clazz.isInstance(doc)) return clazz.cast(doc);
        if (doc instanceof Map) {
            return bindMapTo((Map<?, ?>) doc, clazz);
        }
        throw new IOException("cannot bind " + doc.getClass() + " to " + clazz);
    }

    private static <T> T bindMapTo(Map<?, ?> map, Class<T> clazz) throws IOException {
        try {
            Constructor<T> cons = clazz.getDeclaredConstructor();
            cons.setAccessible(true);
            T instance = cons.newInstance();
            for (Map.Entry<?, ?> e : map.entrySet()) {
                String key = String.valueOf(e.getKey());
                Object value = e.getValue();
                String setterName = "set" + Character.toUpperCase(key.charAt(0)) + key.substring(1);
                try {
                    Method setter = findSetter(clazz, setterName);
                    if (setter != null) {
                        setter.setAccessible(true);
                        Object converted = convertValue(value, setter.getParameterTypes()[0]);
                        setter.invoke(instance, converted);
                    }
                } catch (NoSuchMethodException ignored) {
                    try {
                        Field field = findField(clazz, key);
                        if (field != null) {
                            field.setAccessible(true);
                            Object converted = convertValue(value, field.getType());
                            field.set(instance, converted);
                        }
                    } catch (NoSuchFieldException ignored2) {}
                }
            }
            return instance;
        } catch (Exception e) {
            throw new IOException("Failed to bind to " + clazz.getName(), e);
        }
    }

    private static Method findSetter(Class<?> clazz, String setterName) {
        for (Method m : clazz.getDeclaredMethods()) {
            if (m.getName().equals(setterName) && m.getParameterCount() == 1) {
                return m;
            }
        }
        return null;
    }

    private static Field findField(Class<?> clazz, String fieldName) {
        for (Field f : clazz.getDeclaredFields()) {
            if (f.getName().equalsIgnoreCase(fieldName) || f.getName().equals(fieldName)) {
                return f;
            }
        }
        return null;
    }

    private static Object convertValue(Object value, Class<?> targetType) throws Exception {
        if (value == null) return null;
        if (targetType.isInstance(value)) return value;

        if (targetType == String.class) return String.valueOf(value);
        if (targetType == int.class || targetType == Integer.class) {
            if (value instanceof Number) return ((Number) value).intValue();
            return Integer.parseInt(String.valueOf(value));
        }
        if (targetType == long.class || targetType == Long.class) {
            if (value instanceof Number) return ((Number) value).longValue();
            return Long.parseLong(String.valueOf(value));
        }
        if (targetType == double.class || targetType == Double.class) {
            if (value instanceof Number) return ((Number) value).doubleValue();
            return Double.parseDouble(String.valueOf(value));
        }
        if (targetType == float.class || targetType == Float.class) {
            if (value instanceof Number) return ((Number) value).floatValue();
            return Float.parseFloat(String.valueOf(value));
        }
        if (targetType == boolean.class || targetType == Boolean.class) {
            if (value instanceof Boolean) return value;
            String s = String.valueOf(value).toLowerCase();
            return "true".equals(s) || "yes".equals(s) || "on".equals(s);
        }
        if (targetType == Instant.class && value instanceof String) {
            return Instant.parse((String) value);
        }
        if (targetType == LocalDate.class && value instanceof String) {
            return LocalDate.parse((String) value);
        }
        if (targetType == LocalDateTime.class && value instanceof String) {
            return LocalDateTime.parse((String) value);
        }
        if (value instanceof Map && !targetType.isPrimitive() && targetType != Object.class) {
            return bindMapTo((Map<?, ?>) value, (Class) targetType);
        }
        return value;
    }

    // =========================================================================
    // Path-based navigation (RFC 6901 JSON Pointer)
    // =========================================================================

    /**
     * Get value at JSON Pointer path (e.g. "/spec/replicas", "/items/0/name").
     */
    @SuppressWarnings("unchecked")
    public static Object get(Object root, String path) {
        if (root == null || path == null) return null;
        String[] parts = parsePath(path);
        Object cur = root;
        for (int i = 0; i < parts.length; i++) {
            if (cur == null) return null;
            String part = parts[i];
            if (cur instanceof Map) {
                Map<?, ?> m = (Map<?, ?>) cur;
                cur = m.get(part);
            } else if (cur instanceof List) {
                List<?> l = (List<?>) cur;
                if ("-".equals(part)) {
                    cur = l.isEmpty() ? null : l.get(l.size() - 1);
                } else {
                    int idx;
                    try { idx = Integer.parseInt(part); }
                    catch (NumberFormatException e) { return null; }
                    if (idx < 0 || idx >= l.size()) return null;
                    cur = l.get(idx);
                }
            } else {
                return null;
            }
        }
        return cur;
    }

    public static String getString(Object root, String path) {
        Object v = get(root, path);
        return v == null ? null : String.valueOf(v);
    }

    public static int getInt(Object root, String path, int def) {
        Object v = get(root, path);
        if (v == null) return def;
        if (v instanceof Number) return ((Number) v).intValue();
        try { return Integer.parseInt(String.valueOf(v).trim()); }
        catch (NumberFormatException e) { return def; }
    }

    public static long getLong(Object root, String path, long def) {
        Object v = get(root, path);
        if (v == null) return def;
        if (v instanceof Number) return ((Number) v).longValue();
        try { return Long.parseLong(String.valueOf(v).trim()); }
        catch (NumberFormatException e) { return def; }
    }

    public static boolean getBool(Object root, String path, boolean def) {
        Object v = get(root, path);
        if (v == null) return def;
        if (v instanceof Boolean) return (Boolean) v;
        String s = String.valueOf(v).trim().toLowerCase();
        if ("true".equals(s) || "yes".equals(s) || "on".equals(s)) return true;
        if ("false".equals(s) || "no".equals(s) || "off".equals(s)) return false;
        return def;
    }

    private static String[] parsePath(String path) {
        if (path == null || path.isEmpty()) return new String[0];
        if (!path.startsWith("/")) path = "/" + path;
        return path.substring(1).split("/");
    }

    /**
     * Set value at JSON Pointer path, creating intermediate structures.
     */
    @SuppressWarnings("unchecked")
    public static Object set(Object root, String path, Object value) {
        if (root == null) root = new LinkedHashMap<>();
        if (!(root instanceof Map || root instanceof List)) {
            throw new IllegalArgumentException("Root must be Map or List for set");
        }
        String[] parts = parsePath(path);
        if (parts.length == 0) return value;

        Object cur = root;
        for (int i = 0; i < parts.length - 1; i++) {
            String part = parts[i];
            if (cur instanceof Map) {
                Map<?, ?> m = (Map<?, ?>) cur;
                Object existing = m.get(part);
                if (existing == null) {
                    existing = createContainer(parts, i + 1);
                    ((Map<Object, Object>) cur).put(part, existing);
                }
                cur = existing;
            } else if (cur instanceof List) {
                List<?> l = (List<?>) cur;
                int idx = "-".equals(part) ? l.size() - 1 : Integer.parseInt(part);
                while (((List<?>) l).size() <= idx) {
                    ((List<Object>) l).add(createContainer(parts, i + 1));
                }
                cur = l.get(idx);
            }
        }

        String lastPart = parts[parts.length - 1];
        if (cur instanceof Map) {
            ((Map<Object, Object>) cur).put(lastPart, value);
        } else if (cur instanceof List) {
            List<?> l = (List<?>) cur;
            int idx = "-".equals(lastPart) ? l.size() - 1 : Integer.parseInt(lastPart);
            while (((List<?>) l).size() <= idx) {
                ((List<Object>) l).add(null);
            }
            ((List<Object>) l).set(idx, value);
        }
        return root;
    }

    private static Object createContainer(String[] parts, int fromIndex) {
        String next = fromIndex < parts.length ? parts[fromIndex] : null;
        if (next != null) {
            try { Integer.parseInt(next); return new ArrayList<>(); }
            catch (NumberFormatException e) { return new LinkedHashMap<>(); }
        }
        return new LinkedHashMap<>();
    }

    /**
     * Delete value at JSON Pointer path.
     */
    @SuppressWarnings("unchecked")
    public static Object delete(Object root, String path) {
        if (root == null || path == null) return root;
        String[] parts = parsePath(path);
        if (parts.length == 0) return root;

        Object cur = root;
        for (int i = 0; i < parts.length - 1; i++) {
            if (cur instanceof Map) {
                cur = ((Map<?, ?>) cur).get(parts[i]);
            } else if (cur instanceof List) {
                List<?> l = (List<?>) cur;
                int idx = "-".equals(parts[i]) ? l.size() - 1 : Integer.parseInt(parts[i]);
                if (idx < 0 || idx >= l.size()) return root;
                cur = l.get(idx);
            } else { return root; }
            if (cur == null) return root;
        }

        String lastPart = parts[parts.length - 1];
        if (cur instanceof Map) {
            ((Map<?, ?>) cur).remove(lastPart);
        } else if (cur instanceof List) {
            List<?> l = (List<?>) cur;
            int idx = "-".equals(lastPart) ? l.size() - 1 : Integer.parseInt(lastPart);
            if (idx >= 0 && idx < l.size()) {
                ((List<Object>) l).remove(idx);
            }
        }
        return root;
    }

    // =========================================================================
    // Deep merge
    // =========================================================================

    /**
     * Deep merge two YAML documents. Values in override take precedence.
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> merge(Map<String, Object> base, Map<String, Object> override) {
        if (base == null) return override != null ? new LinkedHashMap<>(override) : new LinkedHashMap<>();
        if (override == null) return new LinkedHashMap<>(base);
        Map<String, Object> result = new LinkedHashMap<>(base);
        for (Map.Entry<String, Object> e : override.entrySet()) {
            Object baseVal = result.get(e.getKey());
            if (baseVal instanceof Map && e.getValue() instanceof Map) {
                result.put(e.getKey(), merge((Map<String, Object>) baseVal, (Map<String, Object>) e.getValue()));
            } else {
                result.put(e.getKey(), deepCopy(e.getValue()));
            }
        }
        return result;
    }

    public static Map<String, Object> mergeAll(Map<String, Object>... docs) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (Map<String, Object> doc : docs) {
            if (doc != null) result = merge(result, doc);
        }
        return result;
    }

    private static Object deepCopy(Object o) {
        if (o == null) return null;
        if (o instanceof Map) {
            Map<Object, Object> m = new LinkedHashMap<>();
            for (Map.Entry<Object, Object> e : ((Map<Object, Object>) o).entrySet()) {
                m.put(e.getKey(), deepCopy(e.getValue()));
            }
            return m;
        }
        if (o instanceof List) {
            List<Object> l = new ArrayList<>();
            for (Object item : (List<?>) o) l.add(deepCopy(item));
            return l;
        }
        if (o instanceof byte[]) return ((byte[]) o).clone();
        if (o instanceof int[]) return ((int[]) o).clone();
        if (o instanceof long[]) return ((long[]) o).clone();
        return o;
    }

    // =========================================================================
    // Environment variable expansion
    // =========================================================================

    /**
     * Expand ${VAR} and ${VAR:-default} in a string.
     */
    public static String expandEnv(String text) {
        return expandEnv(text, System.getenv());
    }

    public static String expandEnv(String text, Map<String, String> env) {
        if (text == null) return null;
        java.util.regex.Pattern p = java.util.regex.Pattern.compile("\\$\\{([^}:]+)(?::-([^}]*))?\\}");
        java.util.regex.Matcher m = p.matcher(text);
        StringBuilder sb = new StringBuilder();
        while (m.find()) {
            String var = m.group(1);
            String def = m.group(2);
            String replacement = env.getOrDefault(var, def != null ? def : "");
            m.appendReplacement(sb, java.util.regex.Matcher.quoteReplacement(replacement));
        }
        m.appendTail(sb);
        return sb.toString();
    }

    /**
     * Expand env vars recursively in a YAML document.
     */
    @SuppressWarnings("unchecked")
    public static Object expandEnvDoc(Object doc) {
        if (doc == null) return null;
        if (doc instanceof String) return expandEnv((String) doc);
        if (doc instanceof Map) {
            Map<Object, Object> out = new LinkedHashMap<>();
            for (Map.Entry<Object, Object> e : ((Map<Object, Object>) doc).entrySet()) {
                out.put(e.getKey(), expandEnvDoc(e.getValue()));
            }
            return out;
        }
        if (doc instanceof List) {
            List<Object> out = new ArrayList<>();
            for (Object item : (List<?>) doc) out.add(expandEnvDoc(item));
            return out;
        }
        return doc;
    }

    // =========================================================================
    // Schema validation
    // =========================================================================

    /**
     * Validate a document against a schema.
     * Schema keys: required[], type, properties{}, items{}, enum[], pattern, minimum, maximum.
     */
    public static List<ValidationError> validate(Object doc, Map<String, Object> schema) {
        List<ValidationError> errors = new ArrayList<>();
        validateNode(doc, schema, "", errors);
        return errors;
    }

    @SuppressWarnings("unchecked")
    private static void validateNode(Object node, Map<String, Object> schema, String path,
                                    List<ValidationError> errors) {
        if (schema == null) return;

        List<String> required = (List<String>) schema.get("required");
        if (required != null && node instanceof Map) {
            Map<?, ?> m = (Map<?, ?>) node;
            for (String req : required) {
                if (!m.containsKey(req)) {
                    errors.add(new ValidationError(path + "/" + req, "required field missing"));
                }
            }
        }

        String expectedType = (String) schema.get("type");
        if (expectedType != null && !validateType(node, expectedType)) {
            errors.add(new ValidationError(path, "expected type " + expectedType + " but got " +
                    (node == null ? "null" : node.getClass().getSimpleName())));
        }

        Map<String, Object> props = (Map<String, Object>) schema.get("properties");
        if (props != null && node instanceof Map) {
            Map<?, ?> m = (Map<?, ?>) node;
            for (Map.Entry<String, Object> e : props.entrySet()) {
                String key = e.getKey();
                Object propSchema = e.getValue();
                if (propSchema instanceof Map) {
                    Object propVal = m.get(key);
                    validateNode(propVal, (Map<String, Object>) propSchema, path + "/" + key, errors);
                }
            }
        }

        Map<String, Object> itemsSchema = (Map<String, Object>) schema.get("items");
        if (itemsSchema != null && node instanceof List) {
            List<?> l = (List<?>) node;
            for (int i = 0; i < l.size(); i++) {
                validateNode(l.get(i), itemsSchema, path + "/" + i, errors);
            }
        }

        List<Object> enumVals = (List<Object>) schema.get("enum");
        if (enumVals != null && node != null) {
            boolean found = false;
            for (Object ev : enumVals) {
                if (Objects.equals(node, ev) || Objects.equals(String.valueOf(node), String.valueOf(ev))) {
                    found = true; break;
                }
            }
            if (!found) {
                errors.add(new ValidationError(path, "value must be one of: " + enumVals));
            }
        }

        String pattern = (String) schema.get("pattern");
        if (pattern != null && node instanceof String) {
            if (!((String) node).matches(pattern)) {
                errors.add(new ValidationError(path, "value '" + node + "' does not match pattern: " + pattern));
            }
        }

        if (node instanceof Number) {
            Number n = (Number) node;
            Object min = schema.get("minimum");
            Object max = schema.get("maximum");
            if (min instanceof Number && n.doubleValue() < ((Number) min).doubleValue()) {
                errors.add(new ValidationError(path, "value " + n + " < minimum " + min));
            }
            if (max instanceof Number && n.doubleValue() > ((Number) max).doubleValue()) {
                errors.add(new ValidationError(path, "value " + n + " > maximum " + max));
            }
        }
    }

    private static boolean validateType(Object node, String expectedType) {
        if (node == null) return "null".equals(expectedType);
        if ("string".equals(expectedType)) return node instanceof String;
        if ("number".equals(expectedType)) return node instanceof Number;
        if ("integer".equals(expectedType)) return node instanceof Number && !((Number) node).toString().contains(".");
        if ("boolean".equals(expectedType)) return node instanceof Boolean;
        if ("array".equals(expectedType) || "list".equals(expectedType)) return node instanceof List;
        if ("object".equals(expectedType) || "map".equals(expectedType)) return node instanceof Map;
        return true;
    }

    // =========================================================================
    // Diff / Patch
    // =========================================================================

    /**
     * Compute diff between two YAML documents.
     */
    public static YamlPatch diff(Object base, Object override) {
        YamlPatch patch = new YamlPatch();
        diffNodes(base, override, "", patch);
        return patch;
    }

    @SuppressWarnings("unchecked")
    private static void diffNodes(Object base, Object override, String path, YamlPatch patch) {
        if (Objects.equals(base, override)) return;

        if (base instanceof Map && override instanceof Map) {
            Map<String, Object> bm = (Map<String, Object>) base;
            Map<String, Object> om = (Map<String, Object>) override;
            Set<String> allKeys = new LinkedHashSet<>();
            allKeys.addAll(bm.keySet());
            allKeys.addAll(om.keySet());
            for (String key : allKeys) {
                String childPath = path.isEmpty() ? "/" + key : path + "/" + key;
                diffNodes(bm.get(key), om.get(key), childPath, patch);
            }
        } else if (base instanceof List && override instanceof List) {
            List<Object> bl = (List<Object>) base;
            List<Object> ol = (List<Object>) override;
            int maxLen = Math.max(bl.size(), ol.size());
            for (int i = 0; i < maxLen; i++) {
                String childPath = path + "/" + i;
                diffNodes(i < bl.size() ? bl.get(i) : null,
                          i < ol.size() ? ol.get(i) : null, childPath, patch);
            }
        } else {
            patch.add(new YamlPatch.Operation("replace", path, override));
        }
    }

    /**
     * Apply a patch to a document.
     */
    public static Object patch(Object doc, YamlPatch patch) {
        Object result = deepCopy(doc);
        for (YamlPatch.Operation op : patch.operations()) {
            String opName = op.op;
            if ("add".equals(opName) || "replace".equals(opName)) {
                set(result, op.path, op.value);
            } else if ("remove".equals(opName)) {
                delete(result, op.path);
            }
        }
        return result;
    }

    // =========================================================================
    // Navigation helpers (same spirit as HttpJson)
    // =========================================================================

    @SuppressWarnings("unchecked")
    public static Map<String, Object> asMap(Object o) {
        if (o == null) return Map.of();
        if (o instanceof Map) return (Map<String, Object>) o;
        throw new IllegalArgumentException("expected map, got " + o.getClass().getSimpleName());
    }

    @SuppressWarnings("unchecked")
    public static List<Object> asList(Object o) {
        if (o == null) return List.of();
        if (o instanceof List) return (List<Object>) o;
        throw new IllegalArgumentException("expected list, got " + o.getClass().getSimpleName());
    }

    public static Object dig(Object root, String... path) {
        return get(root, "/" + String.join("/", path));
    }

    public static String asString(Object o) {
        return o == null ? null : String.valueOf(o);
    }

    public static int asInt(Object o, int def) {
        if (o instanceof Number) return ((Number) o).intValue();
        if (o instanceof String) {
            try { return Integer.parseInt(((String) o).trim()); } catch (NumberFormatException ignored) {}
        }
        return def;
    }

    public static long asLong(Object o, long def) {
        if (o instanceof Number) return ((Number) o).longValue();
        if (o instanceof String) {
            try { return Long.parseLong(((String) o).trim()); } catch (NumberFormatException ignored) {}
        }
        return def;
    }

    public static boolean asBool(Object o, boolean def) {
        if (o instanceof Boolean) return (Boolean) o;
        if (o instanceof String) {
            String t = ((String) o).trim().toLowerCase();
            if ("true".equals(t) || "yes".equals(t) || "on".equals(t)) return true;
            if ("false".equals(t) || "no".equals(t) || "off".equals(t)) return false;
        }
        return def;
    }

    public static Map<String, Object> mapOf(Object... kv) {
        if (kv == null || kv.length == 0) return new LinkedHashMap<>();
        if ((kv.length & 1) != 0) throw new IllegalArgumentException("odd kv length");
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) {
            m.put(String.valueOf(kv[i]), kv[i + 1]);
        }
        return m;
    }

    // =========================================================================
    // Parser
    // =========================================================================

    private static final class Parser {
        final String[] rawLines;
        int line;

        Parser(String text) {
            String norm = text.replace("\r\n", "\n").replace('\r', '\n');
            this.rawLines = norm.split("\n", -1);
            this.line = 0;
        }

        List<Object> parseDocuments() throws IOException {
            List<Object> docs = new ArrayList<>();
            skipEmptyAndCommentsAndDocMarkers();
            while (line < rawLines.length) {
                if (isDocEnd(currentRaw())) {
                    line++;
                    skipEmptyAndCommentsAndDocMarkers();
                    continue;
                }
                if (isDocStart(currentRaw())) {
                    line++;
                    skipEmptyAndComments();
                }
                if (line >= rawLines.length) break;
                Object doc = parseBlock(0);
                if (doc != null || !docs.isEmpty() || line < rawLines.length) {
                    docs.add(doc == null ? new LinkedHashMap<>() : doc);
                }
                skipEmptyAndComments();
                if (line < rawLines.length && (isDocStart(currentRaw()) || isDocEnd(currentRaw()))) {
                    line++;
                    skipEmptyAndCommentsAndDocMarkers();
                } else if (line < rawLines.length) {
                    skipEmptyAndCommentsAndDocMarkers();
                }
            }
            return docs;
        }

        Object parseBlock(int minIndent) throws IOException {
            skipEmptyAndComments();
            if (line >= rawLines.length) return null;
            Line L = peekLine();
            if (L == null) return null;
            if (L.indent < minIndent) return null;

            if (L.content.startsWith("- ") || L.content.equals("-")) {
                return parseList(L.indent);
            }
            if (L.content.startsWith("{") || L.content.startsWith("[")) {
                Object v = parseFlow(L.content);
                line++;
                return v;
            }
            if (looksLikeMapEntry(L.content)) {
                return parseMap(L.indent);
            }
            Object v = parseScalar(L.content);
            line++;
            return v;
        }

        Map<String, Object> parseMap(int indent) throws IOException {
            Map<String, Object> map = new LinkedHashMap<>();
            while (line < rawLines.length) {
                skipEmptyAndComments();
                if (line >= rawLines.length) break;
                if (isDocStart(currentRaw()) || isDocEnd(currentRaw())) break;
                Line L = peekLine();
                if (L == null) break;
                if (L.indent < indent) break;
                if (L.indent > indent) {
                    throw new IOException("bad indent at line " + (line + 1) + ": " + L.content);
                }
                if (L.content.startsWith("- ") || L.content.equals("-")) break;
                if (!looksLikeMapEntry(L.content)) break;

                KeyVal kv = splitKeyVal(L.content);
                line++;
                Object value;
                if (kv.valuePart != null) {
                    String vp = kv.valuePart.trim();
                    if (vp.isEmpty()) {
                        value = parseNestedAfterKey(indent);
                    } else if (vp.startsWith("|") || vp.startsWith(">")) {
                        value = parseLiteralBlock(indent, vp.startsWith("|"));
                    } else if (vp.startsWith("{") || vp.startsWith("[")) {
                        value = parseFlow(vp);
                    } else {
                        value = parseScalar(vp);
                    }
                } else {
                    value = parseNestedAfterKey(indent);
                }
                map.put(kv.key, value);
            }
            return map;
        }

        Object parseNestedAfterKey(int keyIndent) throws IOException {
            skipEmptyAndComments();
            Line next = peekLine();
            if (next == null) return null;
            if (next.indent > keyIndent) {
                return parseBlock(keyIndent + 1);
            }
            if (next.indent == keyIndent && (next.content.startsWith("- ") || next.content.equals("-"))) {
                return parseList(keyIndent);
            }
            return null;
        }

        List<Object> parseList(int indent) throws IOException {
            List<Object> list = new ArrayList<>();
            while (line < rawLines.length) {
                skipEmptyAndComments();
                if (line >= rawLines.length) break;
                if (isDocStart(currentRaw()) || isDocEnd(currentRaw())) break;
                Line L = peekLine();
                if (L == null) break;
                if (L.indent < indent) break;
                if (L.indent > indent) {
                    throw new IOException("list item indent mismatch at line " + (line + 1));
                }
                if (!(L.content.startsWith("- ") || L.content.equals("-"))) break;

                String rest = L.content.equals("-") ? "" : L.content.substring(2);
                line++;
                Object item;
                if (rest.isBlank()) {
                    skipEmptyAndComments();
                    Line next = peekLine();
                    if (next != null && next.indent > indent) {
                        item = parseBlock(indent + 1);
                    } else {
                        item = null;
                    }
                } else if (looksLikeMapEntry(rest)) {
                    KeyVal kv = splitKeyVal(rest);
                    Map<String, Object> m = new LinkedHashMap<>();
                    if (kv.valuePart != null && !kv.valuePart.trim().isEmpty()) {
                        String vp = kv.valuePart.trim();
                        if (vp.startsWith("|") || vp.startsWith(">")) {
                            m.put(kv.key, parseLiteralBlock(indent, vp.startsWith("|")));
                        } else if (vp.startsWith("{") || vp.startsWith("[")) {
                            m.put(kv.key, parseFlow(vp));
                        } else {
                            m.put(kv.key, parseScalar(vp));
                        }
                    } else {
                        skipEmptyAndComments();
                        Line next = peekLine();
                        if (next != null && next.indent > indent) {
                            m.put(kv.key, parseBlock(indent + 1));
                        } else {
                            m.put(kv.key, null);
                        }
                    }
                    while (true) {
                        skipEmptyAndComments();
                        Line n = peekLine();
                        if (n == null) break;
                        if (n.indent <= indent) break;
                        if (n.content.startsWith("- ") || n.content.equals("-")) break;
                        if (!looksLikeMapEntry(n.content)) break;
                        Map<String, Object> more = parseMap(n.indent);
                        m.putAll(more);
                        break;
                    }
                    item = m;
                } else if (rest.startsWith("{") || rest.startsWith("[")) {
                    item = parseFlow(rest);
                } else if (rest.startsWith("|") || rest.startsWith(">")) {
                    item = parseLiteralBlock(indent, rest.startsWith("|"));
                } else {
                    item = parseScalar(rest);
                }
                list.add(item);
            }
            return list;
        }

        String parseLiteralBlock(int parentIndent, boolean literal) {
            StringBuilder sb = new StringBuilder();
            int blockIndent = -1;
            while (line < rawLines.length) {
                String raw = rawLines[line];
                if (isDocStart(raw) || isDocEnd(raw)) break;
                if (raw.isEmpty()) { sb.append('\n'); line++; continue; }
                int ind = countIndent(raw);
                String content = raw.substring(ind);
                if (ind <= parentIndent && !raw.isBlank()) break;
                if (blockIndent < 0 && !raw.isBlank()) blockIndent = ind;
                if (blockIndent >= 0 && ind >= blockIndent) {
                    String piece = raw.substring(blockIndent);
                    sb.append(piece).append('\n');
                    line++;
                } else if (raw.isBlank()) {
                    sb.append('\n');
                    line++;
                } else {
                    break;
                }
            }
            String s = sb.toString();
            if (!literal) {
                s = s.replaceAll("(?<!\n)\n(?!\n)", " ").replaceAll("\n+", "\n").trim() + "\n";
            }
            while (s.endsWith("\n\n")) s = s.substring(0, s.length() - 1);
            if (s.endsWith("\n")) s = s.substring(0, s.length() - 1);
            return s;
        }

        Object parseFlow(String text) throws IOException {
            FlowParser fp = new FlowParser(text.trim());
            Object v = fp.parseValue();
            fp.skipWs();
            return v;
        }

        static Object parseScalar(String raw) {
            if (raw == null) return null;
            String s = raw.trim();
            if (s.isEmpty()) return "";
            if (s.charAt(0) != '"' && s.charAt(0) != '\'') {
                int hash = indexOfUnquotedComment(s);
                if (hash >= 0) s = s.substring(0, hash).trim();
            }
            if (s.isEmpty()) return "";
            char c0 = s.charAt(0);
            if (c0 == '"' && s.endsWith("\"") && s.length() >= 2) {
                return unescapeDouble(s.substring(1, s.length() - 1));
            }
            if (c0 == '\'' && s.endsWith("'") && s.length() >= 2) {
                return s.substring(1, s.length() - 1).replace("''", "'");
            }
            String lower = s.toLowerCase(Locale.ROOT);
            if ("null".equals(lower) || "~".equals(s) || "null".equals(s)) return null;
            if ("true".equals(lower) || "yes".equals(lower) || "on".equals(lower)) return Boolean.TRUE;
            if ("false".equals(lower) || "no".equals(lower) || "off".equals(lower)) return Boolean.FALSE;
            if (isNumber(s)) {
                try {
                    if (s.contains(".") || s.contains("e") || s.contains("E")) {
                        return Double.parseDouble(s);
                    }
                    long lv = Long.parseLong(s);
                    if (lv >= Integer.MIN_VALUE && lv <= Integer.MAX_VALUE) return (int) lv;
                    return lv;
                } catch (NumberFormatException ignored) {}
            }
            return s;
        }

        static boolean isNumber(String s) {
            if (s == null || s.isEmpty()) return false;
            int i = 0;
            if (s.charAt(0) == '-' || s.charAt(0) == '+') i++;
            if (i >= s.length()) return false;
            boolean digit = false, dot = false, exp = false;
            for (; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c >= '0' && c <= '9') digit = true;
                else if (c == '.' && !dot && !exp) dot = true;
                else if ((c == 'e' || c == 'E') && digit && !exp) {
                    exp = true;
                    digit = false;
                    if (i + 1 < s.length() && (s.charAt(i + 1) == '+' || s.charAt(i + 1) == '-')) i++;
                } else return false;
            }
            return digit;
        }

        static int indexOfUnquotedComment(String s) {
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c == '#' && (i == 0 || s.charAt(i - 1) == ' ' || s.charAt(i - 1) == '\t')) return i;
            }
            return -1;
        }

        static String unescapeDouble(String s) {
            StringBuilder sb = new StringBuilder(s.length());
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c == '\\' && i + 1 < s.length()) {
                    char n = s.charAt(++i);
                    switch (n) {
                        case 'n' -> sb.append('\n');
                        case 't' -> sb.append('\t');
                        case 'r' -> sb.append('\r');
                        case '\\' -> sb.append('\\');
                        case '"' -> sb.append('"');
                        case '0' -> sb.append('\0');
                        default -> sb.append(n);
                    }
                } else sb.append(c);
            }
            return sb.toString();
        }

        static boolean looksLikeMapEntry(String content) {
            if (content == null || content.isEmpty()) return false;
            if (content.startsWith("- ") || content.equals("-")) return false;
            boolean inSingle = false, inDouble = false;
            for (int i = 0; i < content.length(); i++) {
                char c = content.charAt(i);
                if (c == '\'' && !inDouble) inSingle = !inSingle;
                else if (c == '"' && !inSingle) inDouble = !inDouble;
                else if (c == ':' && !inSingle && !inDouble) {
                    if (i == 0) return false;
                    return true;
                }
            }
            return false;
        }

        static final class KeyVal {
            final String key;
            final String valuePart;
            KeyVal(String key, String valuePart) {
                this.key = key;
                this.valuePart = valuePart;
            }
        }

        static KeyVal splitKeyVal(String content) throws IOException {
            boolean inSingle = false, inDouble = false;
            int colon = -1;
            for (int i = 0; i < content.length(); i++) {
                char c = content.charAt(i);
                if (c == '\'' && !inDouble) inSingle = !inSingle;
                else if (c == '"' && !inSingle) inDouble = !inDouble;
                else if (c == ':' && !inSingle && !inDouble) {
                    colon = i;
                    break;
                }
            }
            if (colon < 0) throw new IOException("expected key: value in: " + content);
            String keyRaw = content.substring(0, colon).trim();
            String valRaw = content.substring(colon + 1);
            String key;
            if ((keyRaw.startsWith("\"") && keyRaw.endsWith("\""))
                    || (keyRaw.startsWith("'") && keyRaw.endsWith("'"))) {
                Object k = parseScalar(keyRaw);
                key = k == null ? "null" : String.valueOf(k);
            } else {
                key = keyRaw;
            }
            if (valRaw.isEmpty()) return new KeyVal(key, null);
            return new KeyVal(key, valRaw);
        }

        void skipEmptyAndComments() {
            while (line < rawLines.length) {
                String raw = rawLines[line];
                if (raw.isBlank()) { line++; continue; }
                int ind = countIndent(raw);
                String c = raw.substring(ind);
                if (c.startsWith("#")) { line++; continue; }
                break;
            }
        }

        void skipEmptyAndCommentsAndDocMarkers() {
            while (line < rawLines.length) {
                String raw = rawLines[line];
                if (raw.isBlank()) { line++; continue; }
                int ind = countIndent(raw);
                String c = raw.substring(ind);
                if (c.startsWith("#")) { line++; continue; }
                if (isDocStart(raw) || isDocEnd(raw)) { line++; continue; }
                break;
            }
        }

        String currentRaw() {
            return line < rawLines.length ? rawLines[line] : "";
        }

        Line peekLine() {
            if (line >= rawLines.length) return null;
            String raw = rawLines[line];
            if (raw.isBlank()) return null;
            int ind = countIndent(raw);
            String content = raw.substring(ind);
            if (content.startsWith("#")) return null;
            return new Line(ind, content, raw);
        }

        static int countIndent(String raw) {
            int i = 0;
            while (i < raw.length()) {
                char c = raw.charAt(i);
                if (c == ' ') i++;
                else if (c == '\t') i += 2;
                else break;
            }
            return i;
        }

        static boolean isDocStart(String raw) {
            String t = raw.trim();
            return t.equals("---") || t.startsWith("--- ") || t.startsWith("---\t");
        }

        static boolean isDocEnd(String raw) {
            String t = raw.trim();
            return t.equals("...") || t.startsWith("... ") || t.startsWith("...\t");
        }

        static final class Line {
            final int indent;
            final String content;
            final String raw;
            Line(int indent, String content, String raw) {
                this.indent = indent;
                this.content = content;
                this.raw = raw;
            }
        }
    }

    // ---- flow style [a, b] {k: v} ----
    private static final class FlowParser {
        final String s;
        int i;

        FlowParser(String s) { this.s = s; }

        void skipWs() {
            while (i < s.length()) {
                char c = s.charAt(i);
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') i++;
                else break;
            }
        }

        char peek() throws IOException {
            skipWs();
            if (i >= s.length()) throw new IOException("unexpected end of flow YAML");
            return s.charAt(i);
        }

        char next() throws IOException {
            skipWs();
            if (i >= s.length()) throw new IOException("unexpected end of flow YAML");
            return s.charAt(i++);
        }

        Object parseValue() throws IOException {
            char c = peek();
            if (c == '{') return parseObject();
            if (c == '[') return parseArray();
            if (c == '"' || c == '\'') return parseQuoted();
            return parsePlain();
        }

        Map<String, Object> parseObject() throws IOException {
            next();
            Map<String, Object> map = new LinkedHashMap<>();
            skipWs();
            if (peek() == '}') { next(); return map; }
            while (true) {
                Object keyObj = parseValue();
                String key = keyObj == null ? "null" : String.valueOf(keyObj);
                if (next() != ':') throw new IOException("expected ':' in flow map");
                Object val = parseValue();
                map.put(key, val);
                skipWs();
                char c = peek();
                if (c == ',') { next(); continue; }
                if (c == '}') { next(); break; }
                throw new IOException("expected ',' or '}' in flow map");
            }
            return map;
        }

        List<Object> parseArray() throws IOException {
            next();
            List<Object> list = new ArrayList<>();
            skipWs();
            if (peek() == ']') { next(); return list; }
            while (true) {
                list.add(parseValue());
                skipWs();
                char c = peek();
                if (c == ',') { next(); continue; }
                if (c == ']') { next(); break; }
                throw new IOException("expected ',' or ']' in flow list");
            }
            return list;
        }

        Object parseQuoted() throws IOException {
            char q = next();
            StringBuilder sb = new StringBuilder();
            while (i < s.length()) {
                char c = s.charAt(i++);
                if (q == '"' && c == '\\' && i < s.length()) {
                    char n = s.charAt(i++);
                    switch (n) {
                        case 'n' -> sb.append('\n');
                        case 't' -> sb.append('\t');
                        case 'r' -> sb.append('\r');
                        case '"' -> sb.append('"');
                        case '\\' -> sb.append('\\');
                        default -> sb.append(n);
                    }
                } else if (q == '\'' && c == '\'' && i < s.length() && s.charAt(i) == '\'') {
                    sb.append('\'');
                    i++;
                } else if (c == q) {
                    break;
                } else {
                    sb.append(c);
                }
            }
            return sb.toString();
        }

        Object parsePlain() {
            skipWs();
            int start = i;
            while (i < s.length()) {
                char c = s.charAt(i);
                if (c == ',' || c == ']' || c == '}' || c == ':' || c == '#') break;
                if (c == ' ' || c == '\t') {
                    int j = i;
                    while (j < s.length() && (s.charAt(j) == ' ' || s.charAt(j) == '\t')) j++;
                    if (j >= s.length()) break;
                    char n = s.charAt(j);
                    if (n == ',' || n == ']' || n == '}' || n == ':' || n == '#') break;
                }
                i++;
            }
            String raw = s.substring(start, i).trim();
            return Parser.parseScalar(raw);
        }
    }

    // =========================================================================
    // Dumper
    // =========================================================================

    private static final class Dumper {
        final StringBuilder sb;

        Dumper(StringBuilder sb) { this.sb = sb; }

        void dumpDocument(Object value) {
            if (value == null) {
                sb.append("null\n");
                return;
            }
            dumpNode(value, 0, true);
            if (sb.length() == 0 || sb.charAt(sb.length() - 1) != '\n') sb.append('\n');
        }

        void dumpNode(Object value, int indent, boolean atLineStart) {
            if (value == null) {
                if (atLineStart) indentWrite(indent);
                sb.append("null");
                return;
            }
            if (value instanceof Map) {
                Map<?, ?> map = (Map<?, ?>) value;
                if (map.isEmpty()) {
                    if (atLineStart) indentWrite(indent);
                    sb.append("{}");
                    return;
                }
                boolean first = true;
                for (Map.Entry<?, ?> e : map.entrySet()) {
                    if (!first) { sb.append('\n'); indentWrite(indent); }
                    else if (atLineStart) { indentWrite(indent); }
                    else { sb.append('\n'); indentWrite(indent); }
                    first = false;
                    sb.append(formatKey(String.valueOf(e.getKey()))).append(':');
                    writeMapValue(e.getValue(), indent);
                }
                return;
            }
            if (value instanceof Collection) {
                Collection<?> col = (Collection<?>) value;
                if (col.isEmpty()) {
                    if (atLineStart) indentWrite(indent);
                    sb.append("[]");
                    return;
                }
                boolean first = true;
                for (Object item : col) {
                    if (!first) { sb.append('\n'); indentWrite(indent); }
                    else if (atLineStart) { indentWrite(indent); }
                    else { sb.append('\n'); indentWrite(indent); }
                    first = false;
                    sb.append('-');
                    if (item == null) sb.append(' ').append("null");
                    else if (isScalar(item)) sb.append(' ').append(formatScalar(item));
                    else if (item instanceof Map) {
                        Map<?, ?> m = (Map<?, ?>) item;
                        if (m.isEmpty()) sb.append(' ').append("{}");
                        else {
                            boolean fk = true;
                            for (Map.Entry<?, ?> e : m.entrySet()) {
                                if (fk) sb.append(' ').append(formatKey(String.valueOf(e.getKey()))).append(':');
                                else { sb.append('\n'); indentWrite(indent + 2); sb.append(formatKey(String.valueOf(e.getKey()))).append(':'); }
                                writeMapValue(e.getValue(), indent + 2);
                                fk = false;
                            }
                        }
                    } else if (item instanceof Collection) {
                        sb.append('\n');
                        dumpNode(item, indent + 2, true);
                    } else {
                        sb.append(' ').append(formatScalar(item));
                    }
                }
                return;
            }
            if (atLineStart) indentWrite(indent);
            sb.append(formatScalar(value));
        }

        void writeMapValue(Object v, int keyIndent) {
            if (v == null) sb.append(' ').append("null");
            else if (isScalar(v)) sb.append(' ').append(formatScalar(v));
            else if (v instanceof Map) {
                Map<?, ?> m2 = (Map<?, ?>) v;
                if (m2.isEmpty()) sb.append(' ').append("{}");
                else { sb.append('\n'); dumpNode(v, keyIndent + 2, true); }
            }
            else if (v instanceof Collection) {
                Collection<?> c2 = (Collection<?>) v;
                if (c2.isEmpty()) sb.append(' ').append("[]");
                else { sb.append('\n'); dumpNode(v, keyIndent + 2, true); }
            }
            else { sb.append('\n'); dumpNode(v, keyIndent + 2, true); }
        }

        void indentWrite(int n) { for (int i = 0; i < n; i++) sb.append(' '); }

        static boolean isScalar(Object v) {
            return v instanceof String || v instanceof Number || v instanceof Boolean || v instanceof Character;
        }

        static String formatKey(String key) {
            if (key == null) return "\"null\"";
            if (needsQuoting(key) || key.contains(": ") || key.contains("#")
                    || key.isEmpty() || looksLikeNumber(key) || isBooleanish(key)) {
                return quote(key);
            }
            return key;
        }

        static String formatScalar(Object v) {
            if (v == null) return "null";
            if (v instanceof Boolean) return (Boolean) v ? "true" : "false";
            if (v instanceof Number) {
                double d = ((Number) v).doubleValue();
                if (Double.isNaN(d) || Double.isInfinite(d)) return "null";
                if (v instanceof Float || v instanceof Double) return Double.toString(d);
                return v.toString();
            }
            String s = String.valueOf(v);
            if (s.indexOf('\n') >= 0) {
                return quote(s);
            }
            if (needsQuoting(s) || s.isEmpty() || looksLikeNumber(s) || isBooleanish(s)
                    || "null".equalsIgnoreCase(s) || "~".equals(s)
                    || s.startsWith("{") || s.startsWith("[")
                    || s.contains(": ") || s.contains("#")) {
                return quote(s);
            }
            return s;
        }

        static boolean isBooleanish(String s) {
            String t = s.toLowerCase(Locale.ROOT);
            return "true".equals(t) || "false".equals(t) || "yes".equals(t) || "no".equals(t)
                    || "on".equals(t) || "off".equals(t);
        }

        static boolean looksLikeNumber(String s) {
            return Parser.isNumber(s);
        }

        static boolean needsQuoting(String s) {
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                if (c == ':' || c == '#' || c == '{' || c == '}' || c == '[' || c == ']'
                        || c == ',' || c == '&' || c == '*' || c == '!' || c == '|'
                        || c == '>' || c == '\'' || c == '"' || c == '%' || c == '@'
                        || c == '`' || c == '\t') {
                    return true;
                }
            }
            if (s.startsWith(" ") || s.endsWith(" ")) return true;
            return false;
        }

        static String quote(String s) {
            StringBuilder b = new StringBuilder(s.length() + 8);
            b.append('"');
            for (int i = 0; i < s.length(); i++) {
                char c = s.charAt(i);
                switch (c) {
                    case '\\' -> b.append("\\\\");
                    case '"' -> b.append("\\\"");
                    case '\n' -> b.append("\\n");
                    case '\t' -> b.append("\\t");
                    case '\r' -> b.append("\\r");
                    default -> b.append(c);
                }
            }
            b.append('"');
            return b.toString();
        }
    }
}
