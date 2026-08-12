/* CacheException — recoverable / non-recoverable classification. */
package org.bytedeco.pytorch.cache;

import java.io.IOException;

public class CacheException extends RuntimeException {

    public enum Kind { BACKEND, SERIALIZATION, SHARDING, INVALIDATION, TIMEOUT, OVERLOAD, INVARIANT }

    private final Kind kind;

    public CacheException(Kind kind, String message) {
        super(message);
        this.kind = kind;
    }

    public CacheException(Kind kind, String message, Throwable cause) {
        super(message, cause);
        this.kind = kind;
    }

    public Kind kind() { return kind; }

    public static CacheException backend(String msg)                   { return new CacheException(Kind.BACKEND, msg); }
    public static CacheException backend(String msg, Throwable cause) { return new CacheException(Kind.BACKEND, msg, cause); }
    public static CacheException serialization(String msg, IOException e)            { return new CacheException(Kind.SERIALIZATION, msg); }
    public static CacheException sharding(String msg)                  { return new CacheException(Kind.SHARDING, msg); }
    public static CacheException invalidation(String msg)               { return new CacheException(Kind.INVALIDATION, msg); }
    public static CacheException timeout(String msg)                    { return new CacheException(Kind.TIMEOUT, msg); }
    public static CacheException overload(String msg)                  { return new CacheException(Kind.OVERLOAD, msg); }
    public static CacheException invariant(String msg)                 { return new CacheException(Kind.INVARIANT, msg); }
}
