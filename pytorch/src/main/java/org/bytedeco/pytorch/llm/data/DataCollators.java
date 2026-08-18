/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.data;

import org.bytedeco.pytorch.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Mirrors Hugging Face's data collators that every training script in the parent
 * collections depends on.
 *
 * <p>A collator takes a list of feature-maps (as produced by
 * {@code dataset.map(preprocess_function)}) and returns a single {@code Map<String, Tensor>}
 * batch ready for the model forward pass. We deliberately keep tensors at
 * {@code long[]} level; the trainer step is responsible for turning them into
 * {@link org.bytedeco.pytorch.Tensor} right before the forward.
 */
public final class DataCollators {

    private DataCollators() {}

    public interface Collator {
        /** @return batch with keys input_ids / attention_mask / labels (Tensor-ready). */
        Map<String, Object> collate(List<Map<String, Object>> features);
    }

    /**
     * {@code transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)}.
     * Builds labels = input_ids, pads to the longest sequence, masks pad tokens with
     * {@code -100} in the loss labels.
     */
    public static final class LanguageModeling implements Collator {
        private final long padTokenId;
        private final boolean mlm;
        public LanguageModeling(long padTokenId) { this(padTokenId, false); }
        public LanguageModeling(long padTokenId, boolean mlm) { this.padTokenId = padTokenId; this.mlm = mlm; }

        @Override
        public Map<String, Object> collate(List<Map<String, Object>> features) {
            int bsz = features.size();
            int maxLen = 0;
            for (Map<String, Object> f : features) {
                int[] ids = (int[]) f.get("input_ids");
                if (ids.length > maxLen) maxLen = ids.length;
            }
            long[][] inputIds = new long[bsz][maxLen];
            long[][] labels = new long[bsz][maxLen];
            long[][] mask = new long[bsz][maxLen];
            for (int i = 0; i < bsz; i++) {
                int[] ids = (int[]) features.get(i).get("input_ids");
                for (int j = 0; j < maxLen; j++) {
                    if (j < ids.length) {
                        inputIds[i][j] = ids[j];
                        labels[i][j] = mlm ? 0 : ids[j];   // mlm mask would go here
                        mask[i][j] = 1;
                    } else {
                        inputIds[i][j] = padTokenId;
                        labels[i][j] = -100;
                        mask[i][j] = 0;
                    }
                }
            }
            Map<String, Object> out = new java.util.LinkedHashMap<>();
            out.put("input_ids", inputIds);
            out.put("attention_mask", mask);
            out.put("labels", labels);
            return out;
        }
    }

    /**
     * {@code transformers.DataCollatorForSeq2Seq(tokenizer, model, label_pad_token_id=-100, pad_to_multiple_of=8)}.
     * Pads input and labels separately to the longest sequence in the batch (not the dataset),
     * masking pad tokens in the labels.
     */
    public static final class Seq2Seq implements Collator {
        private final long padTokenId;
        private final long labelPadTokenId;
        private final int padToMultipleOf;
        public Seq2Seq(long padTokenId) { this(padTokenId, -100L, 8); }
        public Seq2Seq(long padTokenId, long labelPadTokenId, int padToMultipleOf) {
            this.padTokenId = padTokenId;
            this.labelPadTokenId = labelPadTokenId;
            this.padToMultipleOf = padToMultipleOf;
        }

        @Override
        public Map<String, Object> collate(List<Map<String, Object>> features) {
            int bsz = features.size();
            int maxIn = 0, maxLab = 0;
            for (Map<String, Object> f : features) {
                int[] in = (int[]) f.get("input_ids");
                int[] lab = (int[]) f.get("labels");
                if (in.length > maxIn) maxIn = in.length;
                if (lab.length > maxLab) maxLab = lab.length;
            }
            if (padToMultipleOf > 1) {
                while (maxIn % padToMultipleOf != 0) maxIn++;
                while (maxLab % padToMultipleOf != 0) maxLab++;
            }
            long[][] inputIds = new long[bsz][maxIn];
            long[][] attn = new long[bsz][maxIn];
            long[][] labels = new long[bsz][maxLab];
            long[][] decoderInputIds = new long[bsz][Math.max(1, maxLab - 1)];
            for (int i = 0; i < bsz; i++) {
                int[] in = (int[]) features.get(i).get("input_ids");
                int[] lab = (int[]) features.get(i).get("labels");
                for (int j = 0; j < maxIn; j++) {
                    if (j < in.length) { inputIds[i][j] = in[j]; attn[i][j] = 1; }
                    else { inputIds[i][j] = padTokenId; attn[i][j] = 0; }
                }
                for (int j = 0; j < maxLab; j++) {
                    if (j < lab.length) labels[i][j] = lab[j];
                    else labels[i][j] = labelPadTokenId;
                }
                // decoder_input_ids = labels shifted right (without the last token, pad with 0 at front)
                for (int j = 0; j < decoderInputIds[0].length; j++) {
                    decoderInputIds[i][j] = (j + 1 < maxLab) ? labels[i][j + 1] : 0L;
                }
            }
            Map<String, Object> out = new java.util.LinkedHashMap<>();
            out.put("input_ids", inputIds);
            out.put("attention_mask", attn);
            out.put("labels", labels);
            out.put("decoder_input_ids", decoderInputIds);
            return out;
        }
    }

    /**
     * {@code trl.DataCollatorForCompletionOnlyLM(response_template, tokenizer)} —
     * masks everything before the first occurrence of {@code response_template}
     * (instruction side) with -100.
     */
    public static final class CompletionOnlyLM implements Collator {
        private final String responseTemplate;
        private final long padTokenId;
        public CompletionOnlyLM(String responseTemplate, long padTokenId) {
            this.responseTemplate = responseTemplate;
            this.padTokenId = padTokenId;
        }
        @Override
        public Map<String, Object> collate(List<Map<String, Object>> features) {
            int bsz = features.size();
            int maxLen = 0;
            for (Map<String, Object> f : features) {
                int l = ((int[]) f.get("input_ids")).length;
                if (l > maxLen) maxLen = l;
            }
            long[][] inputIds = new long[bsz][maxLen];
            long[][] attn = new long[bsz][maxLen];
            long[][] labels = new long[bsz][maxLen];
            for (int i = 0; i < bsz; i++) {
                int[] ids = (int[]) features.get(i).get("input_ids");
                int[] templateIds = (int[]) features.get(i).getOrDefault("template_ids", new int[0]);
                int maskUntil = findResponseStart(ids, templateIds);
                for (int j = 0; j < maxLen; j++) {
                    if (j < ids.length) {
                        inputIds[i][j] = ids[j];
                        attn[i][j] = 1;
                        labels[i][j] = (j < maskUntil) ? -100 : ids[j];
                    } else {
                        inputIds[i][j] = padTokenId;
                        attn[i][j] = 0;
                        labels[i][j] = -100;
                    }
                }
            }
            Map<String, Object> out = new java.util.LinkedHashMap<>();
            out.put("input_ids", inputIds);
            out.put("attention_mask", attn);
            out.put("labels", labels);
            return out;
        }
        private static int findResponseStart(int[] ids, int[] template) {
            if (template.length == 0) return 0;
            outer: for (int i = 0; i + template.length <= ids.length; i++) {
                for (int k = 0; k < template.length; k++) if (ids[i + k] != template[k]) continue outer;
                return i + template.length;
            }
            return ids.length;
        }
    }

    /**
     * {@code RewardDataCollatorWithPadding} from beyondguo/RLHF/reward_modeling.py.
     * Splits each feature into {@code _j} / {@code _k} sub-features, then pads independently.
     */
    public static final class RewardPairwise implements Collator {
        private final long padTokenId;
        private final int maxLength;
        public RewardPairwise(long padTokenId, int maxLength) {
            this.padTokenId = padTokenId;
            this.maxLength = maxLength;
        }
        @Override
        public Map<String, Object> collate(List<Map<String, Object>> features) {
            int bsz = features.size();
            long[][] idsJ = new long[bsz][maxLength];
            long[][] maskJ = new long[bsz][maxLength];
            long[][] idsK = new long[bsz][maxLength];
            long[][] maskK = new long[bsz][maxLength];
            for (int i = 0; i < bsz; i++) {
                int[] jIds = (int[]) features.get(i).get("input_ids_j");
                int[] kIds = (int[]) features.get(i).get("input_ids_k");
                pack(jIds, idsJ[i], maskJ[i]);
                pack(kIds, idsK[i], maskK[i]);
            }
            Map<String, Object> out = new java.util.LinkedHashMap<>();
            out.put("input_ids_j", idsJ);
            out.put("attention_mask_j", maskJ);
            out.put("input_ids_k", idsK);
            out.put("attention_mask_k", maskK);
            out.put("return_loss", Boolean.TRUE);
            return out;
        }
        private void pack(int[] src, long[] dstIds, long[] dstMask) {
            int n = Math.min(src.length, dstIds.length);
            for (int j = 0; j < dstIds.length; j++) {
                if (j < n) { dstIds[j] = src[j]; dstMask[j] = 1; }
                else { dstIds[j] = padTokenId; dstMask[j] = 0; }
            }
        }
    }

    /** Lightweight PPOTrainer collator (just stacks lists). */
    public static final class PPO implements Collator {
        @Override public Map<String, Object> collate(List<Map<String, Object>> features) {
            Map<String, Object> out = new java.util.LinkedHashMap<>();
            if (features.isEmpty()) return out;
            for (String key : features.get(0).keySet()) {
                List<Object> vals = new ArrayList<>(features.size());
                for (Map<String, Object> f : features) vals.add(f.get(key));
                out.put(key, vals);
            }
            return out;
        }
    }
}