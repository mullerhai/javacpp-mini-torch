///*
// * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
// *
// * Apache License 2.0.
// */
//package org.bytedeco.pytorch.llm.tunning;
//
//import java.util.LinkedHashMap;
//import java.util.Map;
//import java.util.List;
//
////import org.bytedeco.pytorch.llm.hardware.HardwareSupport;
//import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;
//import org.bytedeco.pytorch.llm.rlhf.BaichuanForSequenceClassification;
//import org.bytedeco.pytorch.llm.rlhf.SequenceClassificationHead;
//import org.bytedeco.pytorch.llm.tokenizers.FastTokenizer;
//import org.bytedeco.pytorch.nn.Module;
//
///** Ex24 — Full fine-tuning for sequence classification (Baichuan). */
//public final class Ex24_FullFinetuningSequenceClassification {
//
//    public static final String NAME = "Ex24_FullFinetuningSequenceClassification";
//
//    public static void run(FastTokenizer tokenizer) {
//        TunningSupport.banner(24, "Full Finetuning Sequence Classification");
//
//        List<Map<String, Object>> raw = TunningSupport.alpacaSample(64);
//        List<Map<String, Object>> formatted = raw.stream().map(row -> {
//            Map<String, Object> r = new LinkedHashMap<>(row);
//            r.put("text", TunningSupport.alpacaPrompt(row));
//            return r;
//        }).toList();
//
//        BitsAndBytesConfig bnb = BitsAndBytesConfig.builder()
//                .loadIn4Bit(true)
//                .bnb4BitQuantType("nf4")
//                .build();
//        Module base = new Module("baichuan-base");
//        SequenceClassificationHead head = new SequenceClassificationHead(4096, 2);
//        BaichuanForSequenceClassification model = new BaichuanForSequenceClassification(base, head);
//
//        double[] losses = TunningSupport.simulateTrainingLoop(1, 0.96);
//        System.out.println("Initial loss: " + losses[0]);
//        System.out.println("Final loss: " + losses[losses.length - 1]);
////        System.out.println("Precision: " + HardwareSupport.pickPrecision(false));
//    }
//
//    public static void main(String[] args) {
//        try (FastTokenizer t = TunningSupport.tokenizerFor("baichuan")) {
//            run(t);
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }
//    }
//}
