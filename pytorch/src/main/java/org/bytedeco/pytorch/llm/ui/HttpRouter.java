/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.ui;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;

/**
 * Tiny request router for the JDK {@link HttpServer}. The handler list is scanned in
 * registration order; the first match wins.
 *
 * <p>Paths may contain {@code {var}} placeholders. The matched values are exposed to the
 * handler via the second {@code Map} parameter.
 */
public final class HttpRouter {

    public interface PathHandler {
        void handle(HttpExchange ex, Map<String, String> pathVars) throws IOException;
    }

    /** Convenience: register a handler that ignores path variables. */
    public HttpRouter add(String method, String path, HttpHandler handler) {
        return add(method, path, (ex, vars) -> handler.handle(ex));
    }

    private final HttpServer server;
    private final List<Route> routes = new ArrayList<>();

    public HttpRouter(HttpServer server) {
        this.server = server;
    }

    public HttpRouter GET(String path, PathHandler handler) { return add("GET", path, handler); }
    public HttpRouter POST(String path, PathHandler handler) { return add("POST", path, handler); }
    public HttpRouter DELETE(String path, PathHandler handler) { return add("DELETE", path, handler); }
    public HttpRouter OPTIONS(String path, PathHandler handler) { return add("OPTIONS", path, handler); }

    /** Variant that accepts a vanilla {@link HttpHandler} for routes without path vars. */
    public HttpRouter GET(String path, HttpHandler handler) { return add("GET", path, handler); }
    public HttpRouter POST(String path, HttpHandler handler) { return add("POST", path, handler); }
    public HttpRouter DELETE(String path, HttpHandler handler) { return add("DELETE", path, handler); }
    public HttpRouter OPTIONS(String path, HttpHandler handler) { return add("OPTIONS", path, handler); }

    public HttpRouter add(String method, String path, PathHandler handler) {
        routes.add(new Route(method, path, handler));
        return this;
    }

    public void mount() {
        // Single dispatch context — the RouterDispatch reads the method + path and forwards.
        server.createContext("/", new DispatchHandler());
    }

    /** Apply CORS + method dispatching to a single context handler. */
    public final class DispatchHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange ex) throws IOException {
            String method = ex.getRequestMethod();
            String path = ServerUtil.pathOnly(ex.getRequestURI().toString());

            // CORS preflight
            if ("OPTIONS".equalsIgnoreCase(method)) {
                ServerUtil.sendOptions(ex);
                return;
            }
            ServerUtil.setCors(ex.getResponseHeaders(), true);

            // 1. Exact-match routes
            for (Route r : routes) {
                if (!r.method.equalsIgnoreCase(method)) continue;
                if (r.path.equals(path)) {
                    r.handler.handle(ex, new LinkedHashMap<>());
                    return;
                }
            }
            // 2. Templated routes
            for (Route r : routes) {
                if (!r.method.equalsIgnoreCase(method)) continue;
                Map<String, String> vars = ServerUtil.matchPath(r.path, path);
                if (vars != null) {
                    r.handler.handle(ex, vars);
                    return;
                }
            }
            // 3. 404
            ServerUtil.sendError(ex, 404, "No route for " + method + " " + path);
        }
    }

    private static final class Route {
        final String method;
        final String path;
        final PathHandler handler;
        Route(String method, String path, PathHandler handler) {
            this.method = method;
            this.path = path;
            this.handler = handler;
        }
    }
}