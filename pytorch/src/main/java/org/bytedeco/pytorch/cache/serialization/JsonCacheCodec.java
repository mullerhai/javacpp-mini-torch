/* JsonCacheCodec — JSON-based codec (universal fallback). */
package org.bytedeco.pytorch.cache.serialization;

import org.bytedeco.pytorch.cache.CacheException;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

public final class JsonCacheCodec implements CacheCodec {

    public static final JsonCacheCodec INSTANCE = new JsonCacheCodec();

    @Override
    public String name() { return "json"; }

    @Override
    public byte[] encode(Object value) {
        if (value == null) return new byte[0];
        if (value instanceof byte[]) return (byte[]) value;
        if (value instanceof CharSequence) return value.toString().getBytes(StandardCharsets.UTF_8);
        if (value instanceof Number || value instanceof Boolean) return value.toString().getBytes(StandardCharsets.UTF_8);
        return Json.encode(value).getBytes(StandardCharsets.UTF_8);
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> T decode(byte[] payload, Class<T> type) {
        if (payload == null || payload.length == 0) return null;
        if (type == byte[].class) return (T) payload;
        if (type == String.class) return (T) new String(payload, StandardCharsets.UTF_8);
        String s = new String(payload, StandardCharsets.UTF_8);
        try {
            Object parsed = Json.decode(s);
            if (type == Map.class) return (T) Json.decodeObject(s);
            if (type == List.class) return (T) Json.decodeArray(s);
            if (type.isInstance(parsed)) return (T) parsed;
            return (T) parsed;
        } catch (IOException e) {
            throw CacheException.serialization("json decode failed: " + e.getMessage(), e);
        }
    }

    @Override
    public boolean supports(Class<?> type) {
        if (type == null) return false;
        if (type == byte[].class || type == String.class) return true;
        if (Number.class.isAssignableFrom(type) || type == Boolean.class) return true;
        return true;
    }

    public Map<String, Object> decodeMap(byte[] payload) {
        if (payload == null || payload.length == 0) return null;
        try {
            return Json.decodeObject(new String(payload, StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw CacheException.serialization("json decodeMap failed: " + e.getMessage(), e);
        }
    }
}
