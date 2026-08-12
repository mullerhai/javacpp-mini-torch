/*
 * SecureSerde -- value <-> byte[] codec used by SecureCacheWriter.
 *
 * <p>Default implementation serialises via Java serialization (so the cached
 * value type is preserved after decrypt). Production deployments should
 * replace this with a JSON or Protobuf codec once the type system is stable.
 */
package org.bytedeco.pytorch.cache.security;

import org.bytedeco.pytorch.cache.CacheValue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

final class SecureSerde {

    byte[] encode(CacheValue<Object> v) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(v);
        }
        return baos.toByteArray();
    }

    @SuppressWarnings("unchecked")
    CacheValue<Object> decode(byte[] buf, CacheValue<Object> template) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(buf))) {
            return (CacheValue<Object>) ois.readObject();
        }
    }
}
