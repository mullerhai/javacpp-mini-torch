/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option)
 * any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/licenses/
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.diffusion.modeling;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * CLIPTextEmbeddings — CLIP's text encoder producing [B, seq_len, hidden_dim]
 * text embeddings for conditioning diffusion models.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class CLIPTextEmbeddings extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    // ── Config ─────────────────────────────────────────────────────

    public static class CLIPTextConfig {
        private int vocabSize = 49408;
        private int hiddenSize = 768;
        private int intermediateSize = 3072;
        private int numHiddenLayers = 12;
        private int numAttentionHeads = 12;
        private int maxPositionEmbeddings = 77;

        public int vocabSize() { return vocabSize; }
        public void vocabSize(int v) { this.vocabSize = v; }
        public int hiddenSize() { return hiddenSize; }
        public void hiddenSize(int v) { this.hiddenSize = v; }
        public int intermediateSize() { return intermediateSize; }
        public void intermediateSize(int v) { this.intermediateSize = v; }
        public int numHiddenLayers() { return numHiddenLayers; }
        public void numHiddenLayers(int v) { this.numHiddenLayers = v; }
        public int numAttentionHeads() { return numAttentionHeads; }
        public void numAttentionHeads(int v) { this.numAttentionHeads = v; }
        public int maxPositionEmbeddings() { return maxPositionEmbeddings; }
        public void maxPositionEmbeddings(int v) { this.maxPositionEmbeddings = v; }
    }

    // ── Inner Classes ─────────────────────────────────────────────

    public static class MultiHeadAttention extends Module {
        private final int hiddenSize;
        private final int numHeads;
        private final int headDim;
        private final Module qProj;
        private final Module kProj;
        private final Module vProj;
        private final Module outProj;

        public MultiHeadAttention(int hiddenSize, int numHeads) {
            super("MultiHeadAttention");
            this.hiddenSize = hiddenSize;
            this.numHeads = numHeads;
            this.headDim = hiddenSize / numHeads;

            this.qProj = register_module("q_proj", new LinearImpl(new LinearOptions(hiddenSize, hiddenSize).bias(false)));
            this.kProj = register_module("k_proj", new LinearImpl(new LinearOptions(hiddenSize, hiddenSize).bias(false)));
            this.vProj = register_module("v_proj", new LinearImpl(new LinearOptions(hiddenSize, hiddenSize).bias(false)));
            this.outProj = register_module("out_proj", new LinearImpl(new LinearOptions(hiddenSize, hiddenSize).bias(false)));
        }

        @Override
        public Tensor forward(Tensor x, Tensor k, Tensor v) {
            long b = x.size(0), seqLen = x.size(1);

            Tensor q = ((LinearImpl) qProj).forward(x);
            Tensor kt = ((LinearImpl) kProj).forward(k);
            Tensor vt = ((LinearImpl) vProj).forward(v);

            q = q.reshape(new long[]{b, seqLen, numHeads, headDim}).transpose(1, 2);
            kt = kt.reshape(new long[]{b, k.size(1), numHeads, headDim}).transpose(1, 2);
            vt = vt.reshape(new long[]{b, v.size(1), numHeads, headDim}).transpose(1, 2);

            Tensor scale = tensor((float) (1.0 / Math.sqrt(headDim)));
            Tensor attn = softmax(matmul(q, kt.transpose(-2, -1)).mul(scale), -1);
            Tensor out = matmul(attn, vt).transpose(1, 2).reshape(new long[]{b, seqLen, hiddenSize});
            return ((LinearImpl) outProj).forward(out);
        }
    }

    public static class QuickGELU extends Module {
        @Override
        public Tensor forward(Tensor x) {
            return x.mul(tanh(x.add(x.mul(x).mul(x).mul(new Scalar(0.044715))).mul(new Scalar(Math.sqrt(2.0 / Math.PI)))).add(new Scalar(1.0))).mul(new Scalar(0.5));
        }
    }

    public static class MLPBlock extends Module {
        private final Module fc1;
        private final Module fc2;
        private final Module act;

        public MLPBlock(int hiddenSize, int intermediateSize) {
            super("MLPBlock");
            this.fc1 = register_module("fc1", new LinearImpl(new LinearOptions(hiddenSize, intermediateSize)));
            this.fc2 = register_module("fc2", new LinearImpl(new LinearOptions(intermediateSize, hiddenSize)));
            this.act = register_module("act", new QuickGELU());
        }

        @Override
        public Tensor forward(Tensor x) {
            return ((LinearImpl) fc2).forward(((Module) act).forward(((LinearImpl) fc1).forward(x)));
        }
    }

    public static class CLIPEncoderLayer extends Module {
        private final Module selfAttn;
        private final Module mlp;
        private final Module layerNorm1;
        private final Module layerNorm2;

        public CLIPEncoderLayer(CLIPTextConfig config, int layerIdx) {
            super("CLIPEncoderLayer." + layerIdx);
            int h = config.hiddenSize();

            this.selfAttn = register_module("self_attn",
                new MultiHeadAttention(h, config.numAttentionHeads()));
            this.mlp = register_module("mlp", new MLPBlock(h, config.intermediateSize()));
        this.layerNorm1 = register_module("layer_norm1", new LayerNormImpl(new LongVector().put((long) h)));
        this.layerNorm2 = register_module("layer_norm2", new LayerNormImpl(new LongVector().put((long) h)));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor residual = x;
            x = ((LayerNormImpl) layerNorm1).forward(x);
            x = ((MultiHeadAttention) selfAttn).forward(x, x, x);
            x = x.add(residual);
            residual = x;
            x = ((LayerNormImpl) layerNorm2).forward(x);
            x = ((MLPBlock) mlp).forward(x);
            return x.add(residual);
        }
    }

    public static class CLIPEncoder extends Module {
        private final List<CLIPEncoderLayer> layers = new ArrayList<>();

        public CLIPEncoder(CLIPTextConfig config) {
            super("CLIPEncoder");
            for (int i = 0; i < config.numHiddenLayers(); i++) {
                layers.add(register_module("layers_" + i,
                    new CLIPEncoderLayer(config, i)));
            }
        }

        @Override
        public Tensor forward(Tensor x) {
            for (CLIPEncoderLayer layer : layers) {
                x = layer.forward(x);
            }
            return x;
        }
    }

    // ── Main Class ─────────────────────────────────────────────────

    private final CLIPTextConfig config;
    private final Module tokenEmbedding;
    private final Module positionEmbedding;
    private final CLIPEncoder encoder;
    private final Module finalLayerNorm;

    public CLIPTextEmbeddings(CLIPTextConfig config) {
        super("CLIPTextEmbeddings");
        this.config = Objects.requireNonNull(config);

        this.tokenEmbedding = register_module("token_embedding",
            new EmbeddingImpl(config.vocabSize(), config.hiddenSize()));
        this.positionEmbedding = register_module("position_embedding",
            new EmbeddingImpl(config.maxPositionEmbeddings(), config.hiddenSize()));
        this.encoder = register_module("encoder", new CLIPEncoder(config));
        this.finalLayerNorm = register_module("final_layer_norm",
            new LayerNormImpl(new LongVector().put((long) config.hiddenSize())));
    }

    public Tensor forward(Tensor inputIds) {
        long b = inputIds.size(0);
        long seqLen = Math.min(inputIds.size(1), config.maxPositionEmbeddings());

        if (inputIds.size(1) > seqLen) {
            inputIds = inputIds.narrow(1, 0, seqLen);
        }

        // Clamp input IDs to the embedding table size to avoid index errors
        // when used with arbitrary tokenizers (e.g., tiktoken with vocab > 49k).
        int vocabSize = config.vocabSize();
        // Build clamp value as a same-dtype Long tensor
        Tensor clampVal = torch.full(inputIds.sizes(),
            new Scalar(vocabSize - 1L),
            new org.bytedeco.pytorch.TensorOptions(ScalarType.Long));
        inputIds = inputIds.minimum(clampVal);
        inputIds = inputIds.maximum(torch.zeros_like(inputIds));

        Tensor embeddings = ((EmbeddingImpl) tokenEmbedding).forward(inputIds);

        // Ensure float32 (Embedding weight is float32)
        // Embedding weights are float32; explicitly cast to be safe.
        embeddings = embeddings.to(torch.ScalarType.Float);

        Tensor posIds = arange(new Scalar(seqLen),
            new org.bytedeco.pytorch.TensorOptions(ScalarType.Long));
        if (b > 1) {
            posIds = posIds.unsqueeze(0).expand(new long[]{b, seqLen});
        }
        Tensor posEmb = ((EmbeddingImpl) positionEmbedding).forward(posIds);
        posEmb = posEmb.to(torch.ScalarType.Float);
        embeddings = embeddings.add(posEmb);
        embeddings = ((CLIPEncoder) encoder).forward(embeddings);
        return ((LayerNormImpl) finalLayerNorm).forward(embeddings);
    }

    public Tensor forward(int[][] inputIds) {
        long[][] longs = new long[inputIds.length][];
        for (int i = 0; i < inputIds.length; i++) {
            longs[i] = new long[inputIds[i].length];
            for (int j = 0; j < inputIds[i].length; j++) longs[i][j] = inputIds[i][j];
        }
        return forward(torch.tensor(longs, new org.bytedeco.pytorch.TensorOptions(ScalarType.Long)));
    }

    public CLIPTextConfig config() { return config; }

    /**
     * Convenience overload that tokenises a single prompt using a simple
     * whitespace split into a BPE-style id sequence. Used by pipelines
     * (e.g. {@code FluxPipeline}) that take a {@link String} prompt.
     *
     * <p>Tokens are mapped via {@code ch % vocabSize} so any reasonable
     * UTF-8 string becomes a valid id sequence without an external
     * tokenizer. For production-quality tokenisation use a {@code
     * org.bytedeco.pytorch.llm.tokenizers.Tokenizer} instance instead and
     * call {@link #forward(int[][])} directly.
     */
    public Tensor forward(String prompt) {
        Objects.requireNonNull(prompt, "prompt");
        int maxPos = config.maxPositionEmbeddings();
        int vocabSize = config.vocabSize();
        String[] toks = prompt.trim().isEmpty() ? new String[0] : prompt.trim().split("\\s+");
        int n = Math.min(toks.length, maxPos);
        int[][] ids = new int[1][n];
        for (int i = 0; i < n; i++) {
            int h = 0;
            for (int k = 0; k < toks[i].length(); k++) {
                h = h * 31 + toks[i].charAt(k);
            }
            ids[0][i] = Math.floorMod(h, vocabSize);
        }
        return forward(ids);
    }
}
