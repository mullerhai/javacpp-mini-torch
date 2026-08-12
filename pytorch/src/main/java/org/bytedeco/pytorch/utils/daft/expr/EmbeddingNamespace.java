/*
 * Embedding namespace - produce embedding vectors for text / image / audio columns.
 *
 * <p>Backed by the {@link org.bytedeco.pytorch.dataframe.ai.EmbeddingModel} registry.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import org.bytedeco.pytorch.dataframe.ai.EmbeddingModel;
import org.bytedeco.pytorch.dataframe.ai.EmbeddingRegistry;
import org.bytedeco.pytorch.dataframe.ai.Modality;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;

import java.util.Objects;

public final class EmbeddingNamespace {

    private final Expression inner;
    public EmbeddingNamespace(Expression inner) { this.inner = Objects.requireNonNull(inner); }

    public Expression encodeText() {
        return encodeText("default-text-model");
    }

    public Expression encodeText(String modelId) {
        return new MediaFn(this, "embed_text_" + modelId, value -> {
            if (value == null) return null;
            EmbeddingModel model = EmbeddingRegistry.get(modelId);
            if (model == null) return null;
            float[] vec = model.embed(value.toString(), Modality.TEXT);
            return model.toEmbeddingData(vec);
        });
    }

    public Expression encodeImage() {
        return encodeImage("default-image-model");
    }

    public Expression encodeImage(String modelId) {
        return new MediaFn(this, "embed_image_" + modelId, value -> {
            if (value == null) return null;
            EmbeddingModel model = EmbeddingRegistry.get(modelId);
            if (model == null) return null;
            byte[] bytes = null;
            if (value instanceof ImageData img) {
                bytes = img.getRawBytes();
            }
            if (bytes == null) {
                bytes = ImageNamespace.bytesOf(value);
            }
            if (bytes == null) return null;
            float[] vec = model.embed(bytes, Modality.IMAGE);
            return model.toEmbeddingData(vec);
        });
    }

    public Expression encodeAudio() {
        return encodeAudio("default-audio-model");
    }

    public Expression encodeAudio(String modelId) {
        return new MediaFn(this, "embed_audio_" + modelId, value -> {
            if (value == null) return null;
            EmbeddingModel model = EmbeddingRegistry.get(modelId);
            if (model == null) return null;
            byte[] bytes = null;
            if (value instanceof AudioData aud) {
                bytes = aud.getRawBytes();
            }
            if (bytes == null) {
                bytes = ImageNamespace.bytesOf(value);
            }
            if (bytes == null) return null;
            float[] vec = model.embed(bytes, Modality.AUDIO);
            return model.toEmbeddingData(vec);
        });
    }
}
