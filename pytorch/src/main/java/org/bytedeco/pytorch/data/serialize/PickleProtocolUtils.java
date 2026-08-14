package org.bytedeco.pytorch.data.serialize;

import java.io.IOException;
import java.io.InputStream;
import java.math.BigInteger;

/**
 * Pure-Java pickle protocol utilities.
 * 
 * <p>This provides the low-level byte manipulation and conversion functions
 * needed to parse Python pickle protocol data.</p>
 */
public abstract class PickleProtocolUtils {

    /**
     * Read a line of text (excluding LF).
     */
    public static String readline(InputStream input) throws IOException {
        return readline(input, false);
    }

    /**
     * Read a line of text, optionally including the LF character.
     */
    public static String readline(InputStream input, boolean includeLF) throws IOException {
        StringBuilder sb = new StringBuilder();
        while (true) {
            int c = input.read();
            if (c == -1) {
                if (sb.length() == 0) {
                    throw new IOException("premature end of file");
                }
                break;
            }
            if (c != '\n' || includeLF) {
                sb.append((char) c);
            }
            if (c == '\n') {
                break;
            }
        }
        return sb.toString();
    }

    /**
     * Read a single unsigned byte.
     */
    public static int readUnsignedByte(InputStream input) throws IOException {
        int b = input.read();
        if (b < 0) throw new IOException("unexpected EOF");
        return b;
    }

    /**
     * Read n bytes into a new byte array.
     */
    public static byte[] readBytes(InputStream input, int n) throws IOException {
        byte[] buffer = new byte[n];
        readBytesInto(input, buffer, 0, n);
        return buffer;
    }

    /**
     * Read exactly len bytes into buffer at offset.
     */
    public static void readBytesInto(InputStream input, byte[] buffer, int offset, int length) throws IOException {
        int totalRead = 0;
        while (totalRead < length) {
            int read = input.read(buffer, offset + totalRead, length - totalRead);
            if (read == -1) {
                throw new IOException("expected " + length + " bytes but got " + totalRead);
            }
            totalRead += read;
        }
    }

    /**
     * Convert 2 little-endian bytes to unsigned int.
     */
    public static int bytesToUnsignedShort(byte[] bytes, int offset) {
        int i = bytes[1 + offset] & 0xff;
        i <<= 8;
        i |= bytes[0 + offset] & 0xff;
        return i;
    }

    /**
     * Convert 4 little-endian bytes to signed int.
     */
    public static int bytesToInt(byte[] bytes, int offset) {
        int i = bytes[3 + offset];
        i <<= 8;
        i |= bytes[2 + offset] & 0xff;
        i <<= 8;
        i |= bytes[1 + offset] & 0xff;
        i <<= 8;
        i |= bytes[0 + offset] & 0xff;
        return i;
    }

    /**
     * Convert 4 big-endian bytes to signed int.
     */
    public static int bytesToIntBE(byte[] bytes, int offset) {
        int i = bytes[offset] & 0xff;
        i <<= 8;
        i |= bytes[1 + offset] & 0xff;
        i <<= 8;
        i |= bytes[2 + offset] & 0xff;
        i <<= 8;
        i |= bytes[3 + offset] & 0xff;
        return i;
    }

    /**
     * Convert 8 little-endian bytes to long.
     */
    public static long bytesToLong(byte[] bytes, int offset) {
        long i = bytes[7 + offset] & 0xff;
        i <<= 8;
        i |= bytes[6 + offset] & 0xff;
        i <<= 8;
        i |= bytes[5 + offset] & 0xff;
        i <<= 8;
        i |= bytes[4 + offset] & 0xff;
        i <<= 8;
        i |= bytes[3 + offset] & 0xff;
        i <<= 8;
        i |= bytes[2 + offset] & 0xff;
        i <<= 8;
        i |= bytes[1 + offset] & 0xff;
        i <<= 8;
        i |= bytes[offset] & 0xff;
        return i;
    }

    /**
     * Convert 8 big-endian bytes to double.
     */
    public static double bytesToDoubleBE(byte[] bytes, int offset) {
        long bits = bytesToLongBE(bytes, offset);
        return Double.longBitsToDouble(bits);
    }

    /**
     * Convert 8 big-endian bytes to long.
     */
    public static long bytesToLongBE(byte[] bytes, int offset) {
        long i = bytes[offset] & 0xff;
        i <<= 8;
        i |= bytes[1 + offset] & 0xff;
        i <<= 8;
        i |= bytes[2 + offset] & 0xff;
        i <<= 8;
        i |= bytes[3 + offset] & 0xff;
        i <<= 8;
        i |= bytes[4 + offset] & 0xff;
        i <<= 8;
        i |= bytes[5 + offset] & 0xff;
        i <<= 8;
        i |= bytes[6 + offset] & 0xff;
        i <<= 8;
        i |= bytes[7 + offset] & 0xff;
        return i;
    }

    /**
     * Decode a pickle long (little-endian signed integer bytes).
     */
    public static Number decodeLong(byte[] data) {
        if (data.length == 0) {
            return 0L;
        }
        // Reverse for little-endian
        byte[] reversed = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            reversed[i] = data[data.length - 1 - i];
        }
        BigInteger bigint = new BigInteger(1, reversed);
        return optimizeInteger(bigint);
    }

    /**
     * Optimize BigInteger to primitive if possible.
     */
    public static Number optimizeInteger(BigInteger bigint) {
        if (bigint.signum() == 0) return 0L;
        
        BigInteger MAX_LONG = BigInteger.valueOf(Long.MAX_VALUE);
        BigInteger MIN_LONG = BigInteger.valueOf(Long.MIN_VALUE);
        
        if (bigint.signum() > 0) {
            if (bigint.compareTo(MAX_LONG) <= 0) {
                return bigint.longValue();
            }
        } else {
            if (bigint.compareTo(MIN_LONG) >= 0) {
                return bigint.longValue();
            }
        }
        return bigint;
    }

    /**
     * Convert raw bytes to string using ISO-8859-1 (latin1).
     */
    public static String rawStringFromBytes(byte[] data) {
        StringBuilder sb = new StringBuilder(data.length);
        for (byte b : data) {
            sb.append((char) (b & 0xff));
        }
        return sb.toString();
    }

    /**
     * Decode escaped string sequences like \xNN, \n, \r, \t.
     */
    public static String decodeEscaped(String str) {
        if (str == null || str.isEmpty() || str.indexOf('\\') == -1) {
            return str;
        }
        StringBuilder sb = new StringBuilder(str.length());
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (c == '\\' && i + 1 < str.length()) {
                char c2 = str.charAt(++i);
                switch (c2) {
                    case '\\': sb.append('\\'); break;
                    case 'x': {
                        if (i + 2 < str.length()) {
                            char h1 = str.charAt(++i);
                            char h2 = str.charAt(++i);
                            c = (char) Integer.parseInt("" + h1 + h2, 16);
                        }
                        sb.append(c);
                        break;
                    }
                    case 'n': sb.append('\n'); break;
                    case 'r': sb.append('\r'); break;
                    case 't': sb.append('\t'); break;
                    default: sb.append(c2); break;
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /**
     * Decode Unicode escape sequences.
     */
    public static String decodeUnicodeEscaped(String str) {
        if (str == null || str.isEmpty() || str.indexOf('\\') == -1) {
            return str;
        }
        StringBuilder sb = new StringBuilder(str.length());
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            if (c == '\\' && i + 1 < str.length()) {
                char c2 = str.charAt(++i);
                switch (c2) {
                    case '\\': sb.append('\\'); break;
                    case 'u': {
                        if (i + 4 < str.length()) {
                            char h1 = str.charAt(++i);
                            char h2 = str.charAt(++i);
                            char h3 = str.charAt(++i);
                            char h4 = str.charAt(++i);
                            c = (char) Integer.parseInt("" + h1 + h2 + h3 + h4, 16);
                        }
                        sb.append(c);
                        break;
                    }
                    case 'U': {
                        if (i + 8 < str.length()) {
                            String hex = "" + str.charAt(++i) + str.charAt(++i) 
                                       + str.charAt(++i) + str.charAt(++i)
                                       + str.charAt(++i) + str.charAt(++i)
                                       + str.charAt(++i) + str.charAt(++i);
                            c = (char) Integer.parseInt(hex, 16);
                        }
                        sb.append(c);
                        break;
                    }
                    case 'n': sb.append('\n'); break;
                    case 'r': sb.append('\r'); break;
                    case 't': sb.append('\t'); break;
                    default: sb.append(c2); break;
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /**
     * Swap byte order in-place for given element size.
     */
    public static void swapEndianness(byte[] a, int elemSize) {
        for (int i = 0; i + elemSize <= a.length; i += elemSize) {
            for (int j = 0; j < elemSize / 2; j++) {
                byte tmp = a[i + j];
                a[i + j] = a[i + elemSize - 1 - j];
                a[i + elemSize - 1 - j] = tmp;
            }
        }
    }
}
