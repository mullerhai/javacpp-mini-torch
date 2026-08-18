///*
// * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
// *
// * Licensed either under the Apache License, Version 2.0, or (at your option)
// * under the terms of the GNU General Public License as published by
// * the Free Software Foundation (subject to the "Classpath" exception),
// * either version 2, or any later version (collectively, the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *     http://www.gnu.org/licenses/
// *     http://www.gnu.org/software/classpath/license.html
// */
//package org.bytedeco.pytorch.llm.tunning;
//
///**
// * Driver / index for the Java port of several open-source LLM fine-tuning
// * collections. Each example demonstrates a specific fine-tuning pattern.
// */
//public final class TunningExamples {
//
//    private TunningExamples() {}
//
//    public static void main(String[] args) {
//        if (args.length == 0) {
//            System.out.println("Usage: TunningExamples <index 1..36 | all>");
//            System.out.println();
//            System.out.println("Examples:");
//            System.out.println("  1   LoRA fine-tuning");
//            System.out.println("  2   Pretrain WizardLM");
//            System.out.println("  3   Instruction tuning Llama-2");
//            System.out.println("  4   Guanaco LoRA");
//            System.out.println("  5   Bloom-560M QLoRA");
//            System.out.println("  6   OPT LoRA");
//            System.out.println("  7   Falcon-7B QLoRA");
//            System.out.println("  8   StableVicuna");
//            System.out.println("  9   GPT-NeoX 4-bit");
//            System.out.println(" 10   MPT-Instruct");
//            System.out.println(" 11   RLHF PPO");
//            System.out.println(" 12   Llama-2 SFT");
//            System.out.println(" 13   Samantha Assistant");
//            System.out.println(" 14   OpenAI Fine-tuning");
//            System.out.println(" 15   Phi-1.5 Fine-tuning");
//            System.out.println(" 16   Gemma LoRA + Neo4j");
//            System.out.println(" 17   Wikipedia Entity Link");
//            System.out.println(" 18   Neo4j GraphCypher");
//            System.out.println(" 19   MLflow Evaluate");
//            System.out.println(" 20   HQQ 1-bit Quantization");
//            System.out.println(" 21   LlamaFactory Config");
//            System.out.println(" 22   Gradio Chatbot");
//            System.out.println(" 23   Cache-Augmented Generation");
//            System.out.println(" 24   Full Finetuning Sequence Classification");
//            System.out.println(" 25   LoRA PEFT Tuning");
//            System.out.println(" 26   Prefix Tuning + BitFit");
//            System.out.println(" 27   Instruction Tuning Causal LM");
//            System.out.println(" 28   Custom Dataset Save/Load");
//            System.out.println(" 29   QLoRA Pipeline Beginner");
//            System.out.println(" 30   Instruction Data Tokenization");
//            System.out.println(" 31   ChatGLM LoRA Tuning");
//            System.out.println(" 32   Reward Modeling RLHF");
//            System.out.println(" 33   Unsloth LLaMA-3 Fine-tune");
//            System.out.println(" 34   DeepSpeed NeMo + Inference");
//            System.out.println(" 35   Gemma KULM Tuning");
//            System.out.println(" 36   Gemma Inference + Evaluation");
//            return;
//        }
//
//        String sel = args[0];
//        boolean all = sel.equalsIgnoreCase("all");
//
//        if (all || sel.equals("1"))  Ex01_LoraFineTuning.run();
//        if (all || sel.equals("2"))  Ex02_PretrainWizardLM.run();
//        if (all || sel.equals("3"))  Ex03_InstructionTuningLlama2.run();
//        if (all || sel.equals("4"))  Ex04_GuanacoLoRA.run();
//        if (all || sel.equals("5"))  Ex05_BloomQLoRA.run();
//        if (all || sel.equals("6"))  Ex06_OptLoRA.run();
//        if (all || sel.equals("7"))  Ex07_Falcon7BQLoRA.run();
//        if (all || sel.equals("8"))  Ex08_StableVicuna.run();
//        if (all || sel.equals("9"))  Ex09_GPTNeoX4bit.run();
//        if (all || sel.equals("10")) Ex10_MPTInstruct.run();
//        if (all || sel.equals("11")) Ex11_RLHFPPO.run();
//        if (all || sel.equals("12")) Ex12_Llama2SFT.run();
//        if (all || sel.equals("13")) Ex13_SamanthaAssistant.run();
//        if (all || sel.equals("14")) Ex14_OpenAIFineTuning.run();
////        if (all || sel.equals("15")) Ex15_Phi15FineTuning.run();
//        if (all || sel.equals("16")) Ex16_GemmaLoRAG.run();
////        if (all || sel.equals("17")) Ex17_WikipediaEntityLink.run();
//        if (all || sel.equals("18")) Ex18_Neo4jGraphCypher.run();
//        if (all || sel.equals("19")) Ex19_MLflowEvaluate.run();
//        if (all || sel.equals("20")) Ex20_LlamaHQQ1bit.run();
//        if (all || sel.equals("21")) Ex21_LlamaFactoryConfig.run();
//        if (all || sel.equals("22")) Ex22_GradioChatbot.run();
//        if (all || sel.equals("23")) Ex23_CacheAugmentedGeneration.run();
////        if (all || sel.equals("24")) Ex24_FullFinetuningSequenceClassification.run();
//        if (all || sel.equals("25")) Ex25_LoRAPeftTuning.run();
//        if (all || sel.equals("26")) Ex26_PrefixTuningAndBitFit.run();
//        if (all || sel.equals("27")) Ex27_InstructionTuningCausalLM.run();
//        if (all || sel.equals("28")) Ex28_CustomDatasetSaveLoad.run();
//        if (all || sel.equals("29")) Ex29_QLoRAPipelineBeginner.run();
//        if (all || sel.equals("30")) Ex30_InstructionDataTokenization.run();
//        if (all || sel.equals("31")) Ex31_ChatGMLoRATuning.run();
//        if (all || sel.equals("32")) Ex32_RewardModelingRLHF.run();
////        if (all || sel.equals("33")) Ex33_UnslothLLaMA3FineTune.run();
//        if (all || sel.equals("34")) Ex34_DeepSpeedNeMoAndInference.run();
//        if (all || sel.equals("35")) Ex35_GemmaKULLMTuning.run();
//        if (all || sel.equals("36")) Ex36_GemmaInferenceAndEvaluation.run();
//    }
//}
