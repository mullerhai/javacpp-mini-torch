/*
 * EncryptedEnvelope -- versioned ciphertext container.
 *
 * <p>Forward-compat: include a 1-byte version prefix so the format can evolve
 * without breaking live envelopes. Each envelope declares the key version it
 * was sealed with, allowing old data to remain decryptable after a rotation.
 */
package org.bytedeco.pytorch.cache.security;

import java.util.Arrays;

public final class EncryptedEnvelope {

    public static final byte VERSION_1 = 1;

    private final byte version;
    private final String keyVersion;
    private final byte[] iv;
    private final byte[] ciphertext;
    private final byte[] tag;

    public EncryptedEnvelope(byte version, String keyVersion, byte[] iv, byte[] ciphertext, byte[] tag) {
        if (ciphertext == null) throw new IllegalArgumentException("ciphertext==null");
        if (iv == null) throw new IllegalArgumentException("iv==null");
        this.version = version;
        this.keyVersion = keyVersion == null ? "0" : keyVersion;
        this.iv = iv.clone();
        this.ciphertext = ciphertext.clone();
        this.tag = tag == null ? new byte[0] : tag.clone();
    }

    public byte version() { return version; }
    public String keyVersion() { return keyVersion; }
    public byte[] iv() { return iv.clone(); }
    public byte[] ciphertext() { return ciphertext.clone(); }
    public byte[] tag() { return tag.clone(); }

    public void clear() {
        Arrays.fill(iv, (byte) 0);
        Arrays.fill(ciphertext, (byte) 0);
        Arrays.fill(tag, (byte) 0);
    }

    public byte[] serialize() {
        byte[] v = keyVersion.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        int size = 1 + 4 + v.length + 4 + iv.length + 4 + ciphertext.length + 4 + tag.length;
        byte[] out = new byte[size];
        int o = 0;
        out[o++] = version;
        out[o++] = (byte) ((v.length >>> 24) & 0xFF);
        out[o++] = (byte) ((v.length >>> 16) & 0xFF);
        out[o++] = (byte) ((v.length >>> 8) & 0xFF);
        out[o++] = (byte) (v.length & 0xFF);
        System.arraycopy(v, 0, out, o, v.length); o += v.length;
        out[o++] = (byte) ((iv.length >>> 24) & 0xFF);
        out[o++] = (byte) ((iv.length >>> 16) & 0xFF);
        out[o++] = (byte) ((iv.length >>> 8) & 0xFF);
        out[o++] = (byte) (iv.length & 0xFF);
        System.arraycopy(iv, 0, out, o, iv.length); o += iv.length;
        out[o++] = (byte) ((ciphertext.length >>> 24) & 0xFF);
        out[o++] = (byte) ((ciphertext.length >>> 16) & 0xFF);
        out[o++] = (byte) ((ciphertext.length >>> 8) & 0xFF);
        out[o++] = (byte) (ciphertext.length & 0xFF);
        System.arraycopy(ciphertext, 0, out, o, ciphertext.length); o += ciphertext.length;
        out[o++] = (byte) ((tag.length >>> 24) & 0xFF);
        out[o++] = (byte) ((tag.length >>> 16) & 0xFF);
        out[o++] = (byte) ((tag.length >>> 8) & 0xFF);
        out[o++] = (byte) (tag.length & 0xFF);
        System.arraycopy(tag, 0, out, o, tag.length);
        return out;
    }

    public static EncryptedEnvelope deserialize(byte[] buf) {
        if (buf == null || buf.length < 5) throw new IllegalArgumentException("bad envelope");
        int o = 0;
        byte version = buf[o++];
        int vlen = ((buf[o++] & 0xFF) << 24) | ((buf[o++] & 0xFF) << 16) | ((buf[o++] & 0xFF) << 8) | (buf[o++] & 0xFF);
        String keyVersion = new String(buf, o, vlen, java.nio.charset.StandardCharsets.UTF_8);
        o += vlen;
        int ivLen = ((buf[o++] & 0xFF) << 24) | ((buf[o++] & 0xFF) << 16) | ((buf[o++] & 0xFF) << 8) | (buf[o++] & 0xFF);
        byte[] iv = new byte[ivLen];
        System.arraycopy(buf, o, iv, 0, ivLen); o += ivLen;
        int ctLen = ((buf[o++] & 0xFF) << 24) | ((buf[o++] & 0xFF) << 16) | ((buf[o++] & 0xFF) << 8) | (buf[o++] & 0xFF);
        byte[] ct = new byte[ctLen];
        System.arraycopy(buf, o, ct, 0, ctLen); o += ctLen;
        int tagLen = ((buf[o++] & 0xFF) << 24) | ((buf[o++] & 0xFF) << 16) | ((buf[o++] & 0xFF) << 8) | (buf[o++] & 0xFF);
        byte[] tag = new byte[tagLen];
        if (tagLen > 0) System.arraycopy(buf, o, tag, 0, tagLen);
        return new EncryptedEnvelope(version, keyVersion, iv, ct, tag);
    }
}
