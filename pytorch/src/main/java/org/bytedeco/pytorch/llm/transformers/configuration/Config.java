/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.transformers.configuration;

import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;
import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;

/**
 * Type-safe wrapper around {@link PretrainedConfig} for a single architecture.
 *
 * <p>Each subclass exposes a {@link #modelType()} string that maps to
 * HuggingFace's {@code config.json:model_type} value (e.g. {@code "llama"},
 * {@code "qwen2"}, {@code "bert"}). {@link AutoConfig} uses this dispatch
 * key to load the right class.
 *
 * <p>Subclasses typically:
 * <ul>
 *   <li>Expose typed accessors (e.g. {@code LlamaConfig.ropeTheta()}).</li>
 *   <li>Provide a {@code fromMap(Map)} static factory that augments
 *       {@link PretrainedConfig#fromMap(Map)} with model-specific fields.</li>
 *   <li>Carry the canonical HF default values as scratch fields.</li>
 * </ul>
 *
 * <p>Instances are immutable — wrap a {@link PretrainedConfig} once and
 * hand it out. The underlying fields stay aligned with the loaded JSON.
 */
public abstract class Config {

    private final PretrainedConfig base;

    protected Config(PretrainedConfig base) {
        this.base = base;
    }

    public final PretrainedConfig base() { return base; }

    /** Model type string used by HF {@code config.json} (e.g. {@code "llama"}). */
    public abstract String modelType();

    /** Concrete subclass. */
    public abstract Class<? extends Config> getClass_();

    /** Convenience: model type enum from the underlying base. */
    public final PretrainedConfig.ModelType baseModelType() {
        return base.modelType();
    }

    public int vocabSize() { return base.vocabSize(); }
    public int hiddenSize() { return base.hiddenSize(); }
    public int intermediateSize() { return base.intermediateSize(); }
    public int numHiddenLayers() { return base.numHiddenLayers(); }
    public int numAttentionHeads() { return base.numAttentionHeads(); }
    public int numKeyValueHeads() { return base.numKeyValueHeads(); }
    public int headDim() { return base.headDim(); }
    public int maxPositionEmbeddings() { return base.maxPositionEmbeddings(); }
    public int bosTokenId() { return base.bosTokenId(); }
    public int eosTokenId() { return base.eosTokenId(); }
    public int padTokenId() { return base.padTokenId(); }
    public double rmsNormEps() { return base.rmsNormEps(); }
    public double layerNormEps() { return base.layerNormEps(); }
    public double ropeTheta() { return base.ropeTheta(); }
    public boolean tieWordEmbeddings() { return base.tieWordEmbeddings(); }
    public boolean attentionBias() { return base.attentionBias(); }
    public String torchDtype() { return base.torchDtype(); }
    public Map<String, Object> extra() { return base.extra(); }

    /** Read {@code config.json} from a HF snapshot directory and dispatch to a subclass. */
    public static Config fromDirectory(Path dir) throws IOException {
        Path cfg = dir.resolve("config.json");
        if (!Files.isRegularFile(cfg)) {
            throw new IOException("No config.json found at " + cfg);
        }
        return fromJson(Files.readString(cfg, StandardCharsets.UTF_8));
    }

    public static Config fromJson(String json) throws IOException {
        Map<String, Object> m = Json.decodeObject(json);
        return fromMap(m);
    }

    public static Config fromMap(Map<String, Object> m) {
        String mt = m.containsKey("model_type") ? String.valueOf(m.get("model_type")).toLowerCase(Locale.ROOT) : "generic";
        PretrainedConfig base = PretrainedConfig.fromMap(m);
        return dispatch(mt, base, m);
    }

    /**
     * Resolve the right subclass for the given model_type. Defaults to
     * a {@link GenericConfig} wrapper for unknown types.
     */
    static Config dispatch(String modelType, PretrainedConfig base, Map<String, Object> raw) {
        switch (modelType) {
            // ---- Decoder-Only ----
            case "llama":          return new LlamaConfig(base);
            case "qwen2":          return new Qwen2Config(base);
            case "qwen3":          return new Qwen3Config(base);
            case "qwen3_vl":       return new Qwen3VLConfig(base);
            case "qwen3_vl_text":  return new Qwen3VLConfig(base);
            case "qwen2_vl":       return new Qwen2VLConfig(base);
            case "qwen2_5_vl":     return new Qwen2_5VLConfig(base);
            case "mistral":        return new MistralConfig(base);
            case "mixtral":        return new MixtralConfig(base);
            case "gemma":          return new GemmaConfig(base);
            case "gemma2":         return new Gemma2Config(base);
            case "gemma3":         return new Gemma3Config(base);
            case "gemma3_text":    return new Gemma3Config(base);
            case "gemma3_vl":      return new Gemma3VLConfig(base);
            case "phi":            return new PhiConfig(base);
            case "phi2":           return new Phi2Config(base);
            case "phi3":           return new Phi3Config(base);
            case "phi4":           return new Phi4Config(base);
            case "gpt2":           return new GPT2Config(base);
            case "gpt_neo":        return new GPTNeoConfig(base);
            case "gpt_neox":       return new GPTNeoXConfig(base);
            case "gptj":           return new GPTJConfig(base);
            case "deepseek":       return new DeepSeekConfig(base);
            case "deepseek_v2":    return new DeepSeekV2Config(base);
            case "deepseek_v3":    return new DeepSeekV3Config(base);
            case "falcon":         return new FalconConfig(base);
            case "starcoder2":     return new Starcoder2Config(base);
            case "bloom":          return new BloomConfig(base);
            case "mpt":            return new MptConfig(base);
            case "cohere":         return new CohereConfig(base);
            case "cohere2":        return new Cohere2Config(base);
            case "olmo":           return new OlmoConfig(base);
            case "olmo2":          return new Olmo2Config(base);
            case "dbrx":           return new DbrxConfig(base);
            case "stablelm":       return new StableLmConfig(base);
            case "grok":           return new GrokConfig(base);
            case "aya":            return new AyaConfig(base);
            case "smollm":         return new SmolLmConfig(base);
            case "recurrent_gemma":return new RecurrentGemmaConfig(base);
            case "mamba":          return new MambaConfig(base);
            case "rwkv":           return new RwkvConfig(base);
            case "granite":        return new GraniteConfig(base);
            case "chatglm":
            case "glm":            return new GlmConfig(base);

            // ---- Encoder-Only ----
            case "bert":           return new BertConfig(base);
            case "roberta":        return new RobertaConfig(base);
            case "deberta":        return new DebertaConfig(base);
            case "deberta-v2":     return new DebertaV2Config(base);
            case "albert":         return new AlbertConfig(base);
            case "electra":        return new ElectraConfig(base);

            // ---- Seq2Seq ----
            case "bart":           return new BartConfig(base);
            case "t5":             return new T5Config(base);

            // ---- Vision ----
            case "vit":            return new ViTConfig(base);
            case "deit":           return new DeiTConfig(base);
            case "clip":           return new CLIPConfig(base);
            case "clip_vision_model": return new CLIPVisionConfig(base);
            case "clip_text_model":   return new CLIPTextConfig(base);
            case "siglip_vision_model": return new SiglipVisionConfig(base);
            case "dinov2":         return new Dinov2Config(base);

            // ---- Multimodal ----
            case "llava":          return new LlavaConfig(base);
            case "llava_next":     return new LlavaNextConfig(base);
            case "mllama":         return new MllamaConfig(base);
            case "pixtral":        return new PixtralConfig(base);

            // ---- Audio ----
            case "whisper":        return new WhisperConfig(base);
            case "speech_to_text": return new Speech2TextConfig(base);
            case "musicgen":       return new MusicgenConfig(base);
            case "musicgen_small": return new MusicgenSmallConfig(base);
            case "encodec":        return new EncodecConfig(base);
            case "clap":           return new ClapConfig(base);

            default:
                return new GenericConfig(base);
        }
    }
}
