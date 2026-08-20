/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Common HTTP helpers — JSON IO, cookie/header parsing, path matching, error responses.
 * Pure JDK (no third-party deps).
 */
public final class ServerUtil {

    private ServerUtil() {}

    public static String sessionIdFromHeaders(Headers h) {
        String cookie = h.getFirst("Cookie");
        return parseCookieValue(cookie, "JSESSION_ID");
    }

    /** Parse a cookie header into a map. Null-safe. */
    public static Map<String, String> parseCookies(String cookieHeader) {
        Map<String, String> m = new HashMap<>();
        if (cookieHeader == null || cookieHeader.isEmpty()) return m;
        for (String pair : cookieHeader.split(";")) {
            int eq = pair.indexOf('=');
            if (eq < 0) continue;
            String k = pair.substring(0, eq).trim();
            String v = pair.substring(eq + 1).trim();
            if (!k.isEmpty()) m.put(k, v);
        }
        return m;
    }

    public static String parseCookieValue(String cookieHeader, String name) {
        if (cookieHeader == null || name == null) return null;
        for (String pair : cookieHeader.split(";")) {
            int eq = pair.indexOf('=');
            if (eq < 0) continue;
            String k = pair.substring(0, eq).trim();
            if (k.equals(name)) {
                return pair.substring(eq + 1).trim();
            }
        }
        return null;
    }

    public static String urlDecode(String s) {
        return s == null ? null : URLDecoder.decode(s, StandardCharsets.UTF_8);
    }

    public static String urlEncode(String s) {
        return s == null ? "" : URLEncoder.encode(s, StandardCharsets.UTF_8);
    }

    public static byte[] readAllBytes(InputStream is) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[4096];
        int n;
        while ((n = is.read(buf)) > 0) bos.write(buf, 0, n);
        return bos.toByteArray();
    }

    public static String readBody(HttpExchange ex) throws IOException {
        return new String(readAllBytes(ex.getRequestBody()), StandardCharsets.UTF_8);
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> parseJson(String body) {
        if (body == null || body.isEmpty()) return new LinkedHashMap<>();
        try {
            return Json.decodeObject(body);
        } catch (IOException ioe) {
            throw new RuntimeException("Invalid JSON body", ioe);
        }
    }

    @SuppressWarnings("unchecked")
    public static List<Object> parseJsonArray(String body) {
        if (body == null || body.isEmpty()) return new ArrayList<>();
        try {
            return Json.decodeArray(body);
        } catch (IOException ioe) {
            throw new RuntimeException("Invalid JSON array body", ioe);
        }
    }

    public static String toJson(Object o) {
        return Json.encode(o);
    }

    public static void setCors(Headers h, boolean allowAny) {
        if (allowAny) {
            h.set("Access-Control-Allow-Origin", "*");
        }
        h.set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        h.set("Access-Control-Allow-Headers", "Content-Type, Cookie");
        h.set("Access-Control-Allow-Credentials", "true");
    }

    public static void sendJson(HttpExchange ex, int status, Object body) throws IOException {
        byte[] bytes = toJson(body).getBytes(StandardCharsets.UTF_8);
        Headers h = ex.getResponseHeaders();
        h.set("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    public static void sendText(HttpExchange ex, int status, String contentType, String body) throws IOException {
        byte[] bytes = body == null ? new byte[0] : body.getBytes(StandardCharsets.UTF_8);
        Headers h = ex.getResponseHeaders();
        h.set("Content-Type", contentType + "; charset=utf-8");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    public static void sendNoContent(HttpExchange ex) throws IOException {
        ex.sendResponseHeaders(204, -1);
        ex.getResponseBody().close();
    }

    public static void sendError(HttpExchange ex, int status, String message) throws IOException {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("error", message);
        m.put("status", status);
        sendJson(ex, status, m);
    }

    public static void sendOptions(HttpExchange ex) throws IOException {
        Headers h = ex.getResponseHeaders();
        h.set("Allow", "GET, POST, DELETE, OPTIONS");
        ex.sendResponseHeaders(204, -1);
        ex.getResponseBody().close();
    }

    /**
     * Extract path variables from a URI template.
     * Example: {@code match("/v1/chat/{id}/turn", "/v1/chat/abc/turn")} -> {@code {id=abc}}.
     * Returns {@code null} if no match.
     */
    public static Map<String, String> matchPath(String template, String uri) {
        if (template == null || uri == null) return null;
        String[] tParts = template.split("/");
        String[] uParts = uri.split("/");
        if (tParts.length != uParts.length) return null;
        Map<String, String> vars = new LinkedHashMap<>();
        for (int i = 0; i < tParts.length; i++) {
            String t = tParts[i];
            String u = uParts[i];
            if (t.startsWith("{") && t.endsWith("}")) {
                vars.put(t.substring(1, t.length() - 1), u);
            } else if (!t.equals(u)) {
                return null;
            }
        }
        return vars;
    }

    /** Strip query string. */
    public static String pathOnly(String uri) {
        if (uri == null) return null;
        int q = uri.indexOf('?');
        return q < 0 ? uri : uri.substring(0, q);
    }

    public static Map<String, String> parseQuery(String uri) {
        Map<String, String> q = new LinkedHashMap<>();
        if (uri == null) return q;
        int hash = uri.indexOf('#');
        if (hash >= 0) uri = uri.substring(0, hash);
        int q2 = uri.indexOf('?');
        if (q2 < 0) return q;
        String qs = uri.substring(q2 + 1);
        if (qs.isEmpty()) return q;
        for (String pair : qs.split("&")) {
            int eq = pair.indexOf('=');
            if (eq < 0) {
                q.put(urlDecode(pair), "");
            } else {
                q.put(urlDecode(pair.substring(0, eq)), urlDecode(pair.substring(eq + 1)));
            }
        }
        return q;
    }

    public static String contentTypeFor(Path p) {
        String name = p.getFileName().toString().toLowerCase();
        if (name.endsWith(".html") || name.endsWith(".htm")) return "text/html";
        if (name.endsWith(".css")) return "text/css";
        if (name.endsWith(".js")) return "application/javascript";
        if (name.endsWith(".json")) return "application/json";
        if (name.endsWith(".png")) return "image/png";
        if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image/jpeg";
        if (name.endsWith(".gif")) return "image/gif";
        if (name.endsWith(".svg")) return "image/svg+xml";
        if (name.endsWith(".ico")) return "image/x-icon";
        if (name.endsWith(".txt")) return "text/plain";
        if (name.endsWith(".woff")) return "font/woff";
        if (name.endsWith(".woff2")) return "font/woff2";
        return "application/octet-stream";
    }
}