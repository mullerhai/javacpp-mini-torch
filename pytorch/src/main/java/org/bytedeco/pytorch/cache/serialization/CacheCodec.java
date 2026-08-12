/* Codec SPI — pluggable serialization for cached values. */
package org.bytedeco.pytorch.cache.serialization;

public interface CacheCodec {

    String name();

    /** Encode to a backend-friendly byte payload (snappy/zstd compressed if available). */
    byte[] encode(Object value);

    /** Decode a payload previously produced by {@link #encode(Object)}. */
    <T> T decode(byte[] payload, Class<T> type);

    /** Whether this codec can handle the given type natively. */
    boolean supports(Class<?> type);
}
