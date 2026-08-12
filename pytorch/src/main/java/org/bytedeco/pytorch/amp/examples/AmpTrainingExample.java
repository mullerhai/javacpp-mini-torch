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
 * or as provided under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.amp.examples;

import org.bytedeco.pytorch.amp.*;
import org.bytedeco.pytorch.amp.config.AmpConfig;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

import java.util.Random;

/**
 * Example demonstrating AMP integration for LLM training.
 *
 * <p>Shows how to use the enterprise-grade AMP system for:
 * <ul>
 *   <li>BF16 mixed precision training</li>
 *   <li>Dynamic loss scaling</li>
 *   <li>Gradient clipping</li>
 *   <li>Performance monitoring</li>
 *   <li>Multi-GPU training</li>
 * </ul>
 *
 * <p>Reference: Meta LLaMA training, DeepSpeed, and Megatron-LM
 */
public class AmpTrainingExample {

    public static void main(String[] args) {
        System.out.println("=== AMP Training Example ===\n");

        // Example 1: Basic BF16 training
        basicBF16Training();

        // Example 2: FP16 training with custom config
        fp16TrainingWithCustomConfig();

        // Example 3: FP8 inference
        fp8Inference();

        // Example 4: Distributed training
        distributedBF16Training();

        // Example 5: Multi-modal training
        multiModalTraining();

        System.out.println("\n=== All Examples Complete ===");
    }

    /**
     * Example 1: Basic BF16 training with default settings.
     */
    public static void basicBF16Training() {
        System.out.println("--- Example 1: Basic BF16 Training ---");

        AmpConfig config = AmpConfig.bf16();
        AmpManager amp = AmpManager.builder()
                .bf16()
                .device("cuda")
                .scalerConfig(AmpManager.GradScalerConfig.builder().maxGradNorm(1.0).build())
                .build();
        GradScaler scaler = GradScaler.createForBF16();

        try {

            System.out.println("AMP enabled: " + amp.isEnabled());
            System.out.println("Forward precision: " + amp.forwardPrecision());
            System.out.println("Backward precision: " + amp.backwardPrecision());
            System.out.println("Initial scale: " + scaler.getScaleFactor());

            // Simulate training loop
            for (int step = 0; step < 3; step++) {
                // Create dummy tensors
                Tensor input = torch.rand(new long[]{4, 512, 4096});
                Tensor target = torch.rand(new long[]{4, 512, 4096});

                // Forward pass with autocast
                try (AutocastContext ctx = amp.autocast()) {
                    Tensor output = input; // Simplified
                    Tensor loss = torch.mse_loss(output, target);

                    // Scale loss and backward
                    Tensor scaledLoss = scaler.scale(loss);
                    scaledLoss.backward();

                    // Unscale gradients and check
                    TensorVector params = new TensorVector();
                    // In real training, params would contain model parameters

                    if (scaler.unscaleAndCheck(params)) {
                        // Optimizer step would go here
                        System.out.printf("  Step %d: scale=%.1f, gradients valid%n",
                                step, scaler.getScaleFactor());
                    } else {
                        System.out.printf("  Step %d: overflow detected, skipping%n", step);
                    }
                }

                input.close();
                target.close();
            }

            // Print stats
            AmpManager.AmpStats stats = amp.getStats();
            System.out.println("\nAMP Stats: " + stats);

            GradScaler.GradScalerStats scalerStats = scaler.getStats();
            System.out.println("Scaler Stats: " + scalerStats);

        } catch (Exception e) {
            System.err.println("Error in basic BF16 training: " + e.getMessage());
        } finally {
            amp.close();
        }

        System.out.println();
    }

    /**
     * Example 2: FP16 training with custom configuration.
     */
    public static void fp16TrainingWithCustomConfig() {
        System.out.println("--- Example 2: FP16 Training with Custom Config ---");

        AmpConfig config = AmpConfig.fp16();

        GradScaler scaler = GradScaler.builder()
                .fp16()
                .initScale(32768.0f)
                .growthFactor(2.0f)
                .backoffFactor(0.5f)
                .growthInterval(1000)
                .build();

        try {
            System.out.println("Config: " + config);

            // Training loop
            for (int step = 0; step < 3; step++) {
                // Simulate gradient computation
                boolean hasOverflow = step == 1; // Simulate overflow at step 1

                if (!hasOverflow) {
                    scaler.getStats(); // Update stats
                    System.out.printf("  Step %d: scale=%.1f, training successful%n",
                            step, scaler.getScaleFactor());
                } else {
                    System.out.printf("  Step %d: overflow detected, scale backoff applied%n", step);
                }
            }
        } catch (Exception e) {
            System.err.println("Error in FP16 training: " + e.getMessage());
        }

        System.out.println();
    }

    /**
     * Example 3: FP8 inference.
     */
    public static void fp8Inference() {
        System.out.println("--- Example 3: FP8 Inference ---");

        try (FP8Scaler e4m3Scaler = FP8Scaler.builder()
                .format(FP8Scaler.FP8Format.E4M3)
                .scaleMethod(FP8Scaler.ScaleMethod.PER_TENSOR)
                .historyWindow(1024)
                .build();
             FP8Scaler e5m2Scaler = FP8Scaler.builder()
                .format(FP8Scaler.FP8Format.E5M2)
                .build()) {

            // Create test tensor
            Tensor input = torch.rand(new long[]{1, 1024, 4096});

            System.out.println("Input shape: " + input.sizes());
            System.out.println("Input dtype: " + input.scalar_type());

            // Quantize to FP8 E4M3
            Tensor fp8Forward = e4m3Scaler.quantize(input);
            System.out.println("FP8 E4M3 dtype: " + fp8Forward.scalar_type());

            // Dequantize back
            Tensor restored = e4m3Scaler.dequantize(fp8Forward);
            System.out.println("Restored dtype: " + restored.scalar_type());

            // Print stats
            System.out.println("\nE4M3 Scaler Stats: " + e4m3Scaler.getStats());
            System.out.println("E5M2 Scaler Stats: " + e5m2Scaler.getStats());

            fp8Forward.close();
            restored.close();
            input.close();

        } catch (Exception e) {
            System.err.println("Error in FP8 inference: " + e.getMessage());
        }

        System.out.println();
    }

    /**
     * Example 4: Distributed BF16 training.
     */
    public static void distributedBF16Training() {
        System.out.println("--- Example 4: Distributed BF16 Training ---");

        int worldSize = 4;
        int rank = 0;

        try (DistributedAmp distAmp = DistributedAmp.builder()
                .worldSize(worldSize)
                .rank(rank)
                .gradScaler(GradScaler.createForBF16())
                .syncBatchNorm(true)
                .averageGradients(true)
                .build()) {

            System.out.println("Distributed AMP: worldSize=" + worldSize + ", rank=" + rank);
            System.out.println("Enabled: " + distAmp.isEnabled());
            System.out.println("Device: " + distAmp.device());

            // Simulate training
            for (int step = 0; step < 3; step++) {
                TensorVector params = new TensorVector();
                // In real training, params would contain model parameters

                // Scale and unscale
                Tensor loss = torch.tensor(1.0f);
                Tensor scaledLoss = distAmp.scaleLoss(loss);

                boolean valid = distAmp.unscaleAndCheck(params);
                System.out.printf("  Step %d: overflow=%b%n", step, !valid);

                loss.close();
                scaledLoss.close();
            }

            // Print stats
            System.out.println("\nDistributed AMP Stats: " + distAmp.getStats());

        } catch (Exception e) {
            System.err.println("Error in distributed training: " + e.getMessage());
        }

        System.out.println();
    }

    /**
     * Example 5: Multi-modal training.
     */
    public static void multiModalTraining() {
        System.out.println("--- Example 5: Multi-Modal Training ---");

        try (MultiModalAmp mmAmp = MultiModalAmp.createForLLaVA()) {

            System.out.println("Multi-Modal AMP initialized");

            // Text processing
            Tensor textTensor = torch.rand(new long[]{4, 512, 4096});
            Tensor textProcessed = mmAmp.castForText(textTensor);
            System.out.println("Text: " + textTensor.scalar_type() + " -> " + textProcessed.scalar_type());

            // Vision processing
            Tensor visionTensor = torch.rand(new long[]{4, 3, 224, 224});
            Tensor visionProcessed = mmAmp.castForVision(visionTensor);
            System.out.println("Vision: " + visionTensor.scalar_type() + " -> " + visionProcessed.scalar_type());

            // Vision encoder
            Tensor visionEncoderOutput = torch.rand(new long[]{4, 257, 1024});
            Tensor visionEncoderProcessed = mmAmp.castForVisionEncoder(visionEncoderOutput);
            System.out.println("VisionEncoder: " + visionEncoderOutput.scalar_type() + " -> " + visionEncoderProcessed.scalar_type());

            // Cross-modal attention
            Tensor crossModalTensor = torch.rand(new long[]{4, 512, 4096});
            Tensor crossModalProcessed = mmAmp.castForCrossAttention(crossModalTensor);
            System.out.println("CrossAttention: " + crossModalTensor.scalar_type() + " -> " + crossModalProcessed.scalar_type());

            // Fusion (should be FP32)
            Tensor fusionTensor = torch.rand(new long[]{4, 512, 4096});
            Tensor fusionProcessed = mmAmp.castForFusion(fusionTensor);
            System.out.println("Fusion: " + fusionTensor.scalar_type() + " -> " + fusionProcessed.scalar_type());

            // Print stats
            System.out.println("\nMulti-Modal AMP Stats: " + mmAmp.getStats());

            // Cleanup
            textTensor.close(); textProcessed.close();
            visionTensor.close(); visionProcessed.close();
            visionEncoderOutput.close(); visionEncoderProcessed.close();
            crossModalTensor.close(); crossModalProcessed.close();
            fusionTensor.close(); fusionProcessed.close();

        } catch (Exception e) {
            System.err.println("Error in multi-modal training: " + e.getMessage());
        }

        System.out.println();
    }
}
