/*
 * Minimal pure-Java HTTP/JSON client for sink backend calls.
 *
 * - Uses java.net.HttpURLConnection (no external deps).
 * - Supports bearer-token auth.
 * - JSON serialization is custom (no Jackson / Gson dependency).
 *
 * Production deployments may swap in OkHttp / Apache HttpClient for
 * connection pooling and TLS acceleration; this reference impl is
 * sufficient for control-plane traffic (low QPS).
 */
package org.bytedeco.pytorch.deploy.integrations.clearml;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Tiny HTTP/JSON client.
 */
public final class HttpJsonClient {

    private String authToken;
    private int connectTimeoutMs = 10_000;
    private int readTimeoutMs = 60_000;
    private boolean verbose;

    public HttpJsonClient setAuthToken(String t) { this.authToken = t; return this; }
    public HttpJsonClient setConnectTimeout(int ms) { this.connectTimeoutMs = ms; return this; }
    public HttpJsonClient setReadTimeout(int ms) { this.readTimeoutMs = ms; return this; }
    public HttpJsonClient setVerbose(boolean v) { this.verbose = v; return this; }

    public Map<String, Object> postJson(String url, Map<String, Object> body) {
        Objects.requireNonNull(url, "url");
        HttpURLConnection conn = null;
        try {
            conn = (HttpURLConnection) new URL(url).openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setRequestProperty("Accept", "application/json");
            if (authToken != null && !authToken.isEmpty()) {
                conn.setRequestProperty("Authorization", "Bearer " + authToken);
            }
            conn.setConnectTimeout(connectTimeoutMs);
            conn.setReadTimeout(readTimeoutMs);
            conn.setDoOutput(true);
            String json = JsonWriter.write(body);
            if (verbose) System.err.println("POST " + url + " body=" + json);
            try (OutputStream os = conn.getOutputStream()) {
                os.write(json.getBytes(StandardCharsets.UTF_8));
            }
            int code = conn.getResponseCode();
            if (code < 200 || code >= 300) {
                String err = readStream(conn.getErrorStream());
                throw new IOException("HTTP " + code + " from " + url + ": " + err);
            }
            String resp = readStream(conn.getInputStream());
            if (verbose) System.err.println("RESP " + resp);
            return JsonParser.parseObject(resp);
        } catch (IOException e) {
            throw new RuntimeException("HTTP POST failed: " + e.getMessage(), e);
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    private static String readStream(InputStream in) {
        if (in == null) return "";
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) sb.append(line);
        } catch (IOException ignored) {}
        return sb.toString();
    }
}

/**
 * Minimal JSON serializer (no external deps).
 */
final class JsonWriter {
    private JsonWriter() {}

    static String write(Object v) {
        StringBuilder sb = new StringBuilder();
        writeValue(sb, v);
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    static void writeValue(StringBuilder sb, Object v) {
        if (v == null) sb.append("null");
        else if (v instanceof Boolean) sb.append(v.toString());
        else if (v instanceof Number) sb.append(v.toString());
        else if (v instanceof CharSequence) writeString(sb, v.toString());
        else if (v instanceof Map) writeObject(sb, (Map<String, Object>) v);
        else if (v instanceof List) writeArray(sb, (List<Object>) v);
        else if (v instanceof Object[]) writeArray(sb, java.util.Arrays.asList((Object[]) v));
        else writeString(sb, v.toString());
    }

    @SuppressWarnings("unchecked")
    static void writeObject(StringBuilder sb, Map<String, Object> obj) {
        sb.append('{');
        boolean first = true;
        for (Map.Entry<String, Object> e : obj.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            writeString(sb, e.getKey());
            sb.append(':');
            writeValue(sb, e.getValue());
        }
        sb.append('}');
    }

    static void writeArray(StringBuilder sb, List<Object> arr) {
        sb.append('[');
        boolean first = true;
        for (Object x : arr) {
            if (!first) sb.append(',');
            first = false;
            writeValue(sb, x);
        }
        sb.append(']');
    }

    static void writeString(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                case '\b': sb.append("\\b"); break;
                case '\f': sb.append("\\f"); break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format(Locale.ROOT, "\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        sb.append('"');
    }
}

/**
 * Tiny JSON parser (recursive descent, supports object / array / string /
 * number / boolean / null).
 */
final class JsonParser {

    private final String src;
    private int pos;

    private JsonParser(String src) { this.src = src; }

    static Map<String, Object> parseObject(String s) {
        JsonParser p = new JsonParser(s == null ? "" : s);
        p.skipWs();
        Object v = p.parseValue();
        if (!(v instanceof Map)) throw new IllegalArgumentException("expected JSON object");
        @SuppressWarnings("unchecked")
        Map<String, Object> m = (Map<String, Object>) v;
        return m;
    }

    private Object parseValue() {
        skipWs();
        if (pos >= src.length()) return null;
        char c = src.charAt(pos);
        if (c == '{') return parseObj();
        if (c == '[') return parseArr();
        if (c == '"') return parseString();
        if (c == 't' || c == 'f') return parseBoolean();
        if (c == 'n') return parseNull();
        return parseNumber();
    }

    private Map<String, Object> parseObj() {
        Map<String, Object> m = new LinkedHashMap<>();
        expect('{');
        skipWs();
        if (peek() == '}') { pos++; return m; }
        while (true) {
            skipWs();
            String key = parseString();
            skipWs();
            expect(':');
            Object v = parseValue();
            m.put(key, v);
            skipWs();
            char c = peek();
            if (c == ',') { pos++; continue; }
            if (c == '}') { pos++; return m; }
            throw new IllegalArgumentException("unexpected '" + c + "' at " + pos);
        }
    }

    private List<Object> parseArr() {
        List<Object> arr = new ArrayList<>();
        expect('[');
        skipWs();
        if (peek() == ']') { pos++; return arr; }
        while (true) {
            arr.add(parseValue());
            skipWs();
            char c = peek();
            if (c == ',') { pos++; continue; }
            if (c == ']') { pos++; return arr; }
            throw new IllegalArgumentException("unexpected '" + c + "' at " + pos);
        }
    }

    private String parseString() {
        expect('"');
        StringBuilder sb = new StringBuilder();
        while (pos < src.length()) {
            char c = src.charAt(pos++);
            if (c == '"') return sb.toString();
            if (c == '\\' && pos < src.length()) {
                char e = src.charAt(pos++);
                switch (e) {
                    case '"': sb.append('"'); break;
                    case '\\': sb.append('\\'); break;
                    case '/': sb.append('/'); break;
                    case 'n': sb.append('\n'); break;
                    case 'r': sb.append('\r'); break;
                    case 't': sb.append('\t'); break;
                    case 'b': sb.append('\b'); break;
                    case 'f': sb.append('\f'); break;
                    case 'u': {
                        if (pos + 4 > src.length()) throw new IllegalArgumentException("bad \\u escape");
                        sb.append((char) Integer.parseInt(src.substring(pos, pos + 4), 16));
                        pos += 4;
                        break;
                    }
                    default: sb.append(e);
                }
            } else {
                sb.append(c);
            }
        }
        throw new IllegalArgumentException("unterminated string");
    }

    private Boolean parseBoolean() {
        if (src.startsWith("true", pos)) { pos += 4; return Boolean.TRUE; }
        if (src.startsWith("false", pos)) { pos += 5; return Boolean.FALSE; }
        throw new IllegalArgumentException("expected boolean at " + pos);
    }

    private Object parseNull() {
        if (src.startsWith("null", pos)) { pos += 4; return null; }
        throw new IllegalArgumentException("expected null at " + pos);
    }

    private Number parseNumber() {
        int start = pos;
        if (peek() == '-') pos++;
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                pos++;
            } else break;
        }
        String s = src.substring(start, pos);
        if (s.contains(".") || s.contains("e") || s.contains("E")) return Double.parseDouble(s);
        return Long.parseLong(s);
    }

    private void expect(char c) {
        skipWs();
        if (pos >= src.length() || src.charAt(pos) != c) {
            throw new IllegalArgumentException("expected '" + c + "' at " + pos);
        }
        pos++;
    }

    private char peek() {
        return pos < src.length() ? src.charAt(pos) : '\0';
    }

    private void skipWs() {
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') pos++;
            else break;
        }
    }
}