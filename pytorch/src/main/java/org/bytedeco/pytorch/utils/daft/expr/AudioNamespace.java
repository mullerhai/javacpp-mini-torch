/*
 * Audio namespace.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.dataframe.media.MediaBridge;

import java.util.Objects;

public final class AudioNamespace {

    private final Expression inner;
    public AudioNamespace(Expression inner) { this.inner = Objects.requireNonNull(inner); }

    public Expression decode() {
        return new MediaFn(this, "decode", bytes -> {
            if (bytes == null) return null;
            if (bytes instanceof byte[]) return new AudioData((byte[]) bytes, 44100);
            return new AudioData(bytes.toString());
        });
    }

    public Expression decode(int targetSampleRate, boolean mono) {
        return new MediaFn(this, "decode", bytes -> {
            if (bytes == null) return null;
            AudioData a;
            if (bytes instanceof byte[]) a = new AudioData((byte[]) bytes, targetSampleRate);
            else a = new AudioData(bytes.toString());
            if (a.getSampleRate() != targetSampleRate) {
                a = MediaBridge.resample(a, targetSampleRate);
            }
            if (mono) a = MediaBridge.toMono(a);
            return a;
        });
    }

    public Expression resample(int targetSampleRate) {
        return new MediaFn(this, "resample_" + targetSampleRate, value -> {
            if (value == null) return null;
            AudioData a = asAudio(value);
            return a == null ? null : MediaBridge.resample(a, targetSampleRate);
        });
    }

    public Expression toMono() {
        return new MediaFn(this, "mono", value -> {
            if (value == null) return null;
            AudioData a = asAudio(value);
            return a == null ? null : MediaBridge.toMono(a);
        });
    }

    public Expression sampleRate() {
        return new MediaFn(this, "sampleRate", value -> {
            if (value == null) return null;
            AudioData a = asAudio(value);
            return a == null ? null : (long) a.getSampleRate();
        });
    }

    public Expression duration() {
        return new MediaFn(this, "duration", value -> {
            if (value == null) return null;
            AudioData a = asAudio(value);
            return a == null ? null : (double) a.getSamples().length / a.getSampleRate();
        });
    }

    public Expression toTensor() {
        return new MediaFn(this, "tensor", value -> {
            if (value == null) return null;
            AudioData a = asAudio(value);
            return a == null ? null : MediaBridge.audioToTensor(a);
        });
    }

    private static AudioData asAudio(Object v) {
        if (v instanceof AudioData) return (AudioData) v;
        if (v instanceof byte[]) return new AudioData((byte[]) v, 44100);
        if (v instanceof String) return new AudioData((String) v);
        return null;
    }
}
