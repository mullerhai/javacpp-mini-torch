/*
 * Video namespace.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.dtype.VideoData;
import org.bytedeco.pytorch.dataframe.media.MediaBridge;

import java.util.Objects;

public final class VideoNamespace {

    private final Expression inner;
    public VideoNamespace(Expression inner) { this.inner = Objects.requireNonNull(inner); }

    public Expression decode() {
        return new MediaFn(this, "decode", bytes -> {
            if (bytes == null) return null;
            if (bytes instanceof byte[]) return new VideoData((byte[]) bytes, "h264");
            return new VideoData(bytes.toString());
        });
    }

    public Expression frameCount() {
        return new MediaFn(this, "frameCount", value -> {
            if (value == null) return null;
            VideoData v = asVideo(value);
            return v == null ? null : (long) v.getFrameCount();
        });
    }

    public Expression frameRate() {
        return new MediaFn(this, "fps", value -> {
            if (value == null) return null;
            VideoData v = asVideo(value);
            return v == null ? null : v.getFps();
        });
    }

    public Expression duration() {
        return new MediaFn(this, "duration", value -> {
            if (value == null) return null;
            VideoData v = asVideo(value);
            return v == null ? null : (double) v.getDuration();
        });
    }

    public Expression width() {
        return new MediaFn(this, "width", value -> {
            if (value == null) return null;
            VideoData v = asVideo(value);
            return v == null ? null : (long) v.getWidth();
        });
    }

    public Expression height() {
        return new MediaFn(this, "height", value -> {
            if (value == null) return null;
            VideoData v = asVideo(value);
            return v == null ? null : (long) v.getHeight();
        });
    }

    public Expression frameAt(int frameIndex) {
        return new MediaFn(this, "frame_" + frameIndex, value -> {
            if (value == null) return null;
            VideoData v = asVideo(value);
            return v == null ? null : MediaBridge.frameAt(v, (double) frameIndex);
        });
    }

    public Expression toTensor() {
        return new MediaFn(this, "tensor", value -> {
            if (value == null) return null;
            VideoData v = asVideo(value);
            return v == null ? null : MediaBridge.videoToTensor(v);
        });
    }

    private static VideoData asVideo(Object v) {
        if (v instanceof VideoData) return (VideoData) v;
        if (v instanceof byte[]) return new VideoData((byte[]) v, "h264");
        if (v instanceof String) return new VideoData((String) v);
        return null;
    }
}
