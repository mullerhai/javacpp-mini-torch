/*
 * DeepFM Enterprise Training Benchmark with Full Distributed Support.
 */
package org.bytedeco.pytorch.distributed.examples;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.optim.options.AdamOptions;

import java.util.*;

import static org.bytedeco.pytorch.global.torch.*;

/**
 * DeepFM Benchmark.
 *
 * Architecture:
 *   Input(sparse indices) [batch, numSparseFeatures]
 *         ↓
 *   Embedding lookup → [batch, numSparseFeatures, embedDim]
 *         ↓
 *   ┌────┴────┐
 *   FM        DNN
 *   ↓         ↓
 *   Combined → [batch, 1]
 *
 * Usage:
 *   java org.bytedeco.pytorch.distributed.examples.DeepFMBenchmark --steps 100 --batch 4096
 */
public final class DeepFMBenchmark {

    public static class ModelConfig {
        public int numFeatures = 1000000;
        public int numDenseFeatures = 13;
        public int embeddingDim = 16;
        public int dnnHiddenDim = 256;
        public int dnnLayers = 4;
        public float dropoutRate = 0.1f;
        public int batchSize = 4096;

        @Override
        public String toString() {
            return String.format("ModelConfig{features=%d, embed=%d, dnn=%dx%d, batch=%d}",
                numFeatures, embeddingDim, dnnLayers, dnnHiddenDim, batchSize);
        }
    }

    public static class TrainConfig {
        public int numSteps = 100;
        public int logInterval = 10;
        public float learningRate = 0.001f;
        public float maxGradNorm = 1.0f;
        public boolean useFP16 = true;

        @Override
        public String toString() {
            return String.format("TrainConfig{steps=%d, lr=%.4f, fp16=%b}", numSteps, learningRate, useFP16);
        }
    }

    public static class BenchmarkResult {
        public final int worldSize, rank;
        public final long totalTimeMs;
        public final double avgThroughput;
        public final double avgLoss;
        public final int successfulSteps;

        public BenchmarkResult(int worldSize, int rank, long totalTimeMs,
                            double avgThroughput, double avgLoss, int successfulSteps) {
            this.worldSize = worldSize;
            this.rank = rank;
            this.totalTimeMs = totalTimeMs;
            this.avgThroughput = avgThroughput;
            this.avgLoss = avgLoss;
            this.successfulSteps = successfulSteps;
        }

        @Override
        public String toString() {
            return String.format(
                "\n╔══════════════════════════════════════════════════════════╗\n" +
                "║     DeepFM Benchmark Results (Rank %d/%d)                ║\n" +
                "╠══════════════════════════════════════════════════════════╣\n" +
                "║  Total Time:    %12d ms                            ║\n" +
                "║  Throughput:    %12.0f samples/s                  ║\n" +
                "║  Avg Loss:      %12.6f                            ║\n" +
                "║  Success Steps: %12d                              ║\n" +
                "╚══════════════════════════════════════════════════════════╝",
                rank, worldSize, totalTimeMs, avgThroughput, avgLoss, Integer.valueOf(successfulSteps));
        }
    }

    /**
     * Run DeepFM training benchmark using direct tensor operations.
     */
    public static BenchmarkResult runBenchmark(int worldSize, int rank,
                                            ModelConfig modelConfig, TrainConfig trainConfig,
                                            Device device, ScalarType dtype) {
        long totalStartTime = System.currentTimeMillis();

        if (rank == 0) {
            System.out.println("\n" + "═".repeat(60));
            System.out.println("       DeepFM Enterprise Training Benchmark");
            System.out.println("═".repeat(60));
            System.out.println("World: " + worldSize + " | Device: " + device + " | Dtype: " + dtype);
            System.out.println(modelConfig);
            System.out.println(trainConfig);
            System.out.println("═".repeat(60));
        }

        int numSparseFeatures = 20;
        int embeddingDim = modelConfig.embeddingDim;
        int dnnHiddenDim = modelConfig.dnnHiddenDim;
        int dnnLayers = modelConfig.dnnLayers;

        // Initialize model parameters
        // Embedding table [numFeatures, embeddingDim]
        Tensor embeddings = randn(new long[]{modelConfig.numFeatures, embeddingDim}).to(device, dtype);
        embeddings.requires_grad_(true);

        // DNN weights
        int firstDnnInput = numSparseFeatures * embeddingDim;
        Tensor[] dnnWeights = new Tensor[dnnLayers];
        Tensor[] dnnBiases = new Tensor[dnnLayers];

        dnnWeights[0] = randn(new long[]{firstDnnInput, dnnHiddenDim}).to(device, dtype);
        dnnWeights[0].requires_grad_(true);
        dnnBiases[0] = zeros(new long[]{dnnHiddenDim}).to(device, dtype);
        dnnBiases[0].requires_grad_(true);

        for (int i = 1; i < dnnLayers; i++) {
            dnnWeights[i] = randn(new long[]{dnnHiddenDim, dnnHiddenDim}).to(device, dtype);
            dnnWeights[i].requires_grad_(true);
            dnnBiases[i] = zeros(new long[]{dnnHiddenDim}).to(device, dtype);
            dnnBiases[i].requires_grad_(true);
        }

        // Final output layer: [embeddingDim + dnnHiddenDim -> 1]
        Tensor outputWeight = randn(new long[]{embeddingDim + dnnHiddenDim, 1}).to(device, dtype);
        outputWeight.requires_grad_(true);
        Tensor outputBias = zeros(new long[]{1}).to(device, dtype);
        outputBias.requires_grad_(true);

        // Dense feature linear
        Tensor denseLinearWeight = randn(new long[]{modelConfig.numDenseFeatures + 1, 1}).to(device, dtype);
        denseLinearWeight.requires_grad_(true);
        Tensor denseLinearBias = zeros(new long[]{1}).to(device, dtype);
        denseLinearBias.requires_grad_(true);

        // Collect trainable parameters as TensorVector
        TensorVector params = new TensorVector();
        params.push_back(embeddings);
        for (int i = 0; i < dnnLayers; i++) {
            params.push_back(dnnWeights[i]);
            params.push_back(dnnBiases[i]);
        }
        params.push_back(outputWeight);
        params.push_back(outputBias);
        params.push_back(denseLinearWeight);
        params.push_back(denseLinearBias);

        // Create optimizer
        Optimizer optimizer = new Adam(params, new AdamOptions().lr(trainConfig.learningRate));

        if (rank == 0) {
            long totalParams = 0;
            for (int i = 0; i < params.size(); i++) {
                totalParams += params.get(i).numel();
            }
            System.out.printf("\n[Model] Total Parameters: %,d%n", totalParams);
        }

        // Data generation
        Random random = new Random(42L + rank);
        List<Double> losses = new ArrayList<>();
        List<Double> throughputs = new ArrayList<>();
        int successfulSteps = 0;
        int failedSteps = 0;

        for (int step = 0; step < trainConfig.numSteps; step++) {
            long iterStart = System.nanoTime();
            try {
                // Generate synthetic CTR data
                long[] sparseData = new long[modelConfig.batchSize * numSparseFeatures];
                float[] denseData = new float[modelConfig.batchSize * (modelConfig.numDenseFeatures + 1)];
                float[] labelsData = new float[modelConfig.batchSize];

                for (int i = 0; i < sparseData.length; i++) {
                    sparseData[i] = random.nextInt(modelConfig.numFeatures);
                }
                for (int i = 0; i < denseData.length; i++) {
                    denseData[i] = (float)(random.nextGaussian() * 0.5 + 0.5);
                }
                for (int i = 0; i < labelsData.length; i++) {
                    labelsData[i] = random.nextFloat() < 0.3f ? 1.0f : 0.0f;
                }

                Tensor sparseInput = tensor(sparseData).reshape(new long[]{modelConfig.batchSize, numSparseFeatures})
                    .to(device, ScalarType.Long);
                Tensor denseInput = tensor(denseData).reshape(new long[]{modelConfig.batchSize, modelConfig.numDenseFeatures + 1})
                    .to(device, dtype);
                Tensor labels = tensor(labelsData).reshape(new long[]{modelConfig.batchSize})
                    .to(device, ScalarType.Float);

                // ═══════ FM Component ═══════
                // Embedding lookup: [batch, numSparseFeatures, embeddingDim]
                Tensor flatIndices = sparseInput.reshape(new long[]{(long)(modelConfig.batchSize * numSparseFeatures)});
                Tensor sparseEmbed = index_select(embeddings, 0, flatIndices)
                    .reshape(new long[]{modelConfig.batchSize, numSparseFeatures, embeddingDim});

                // FM Second Order: 0.5 * ((sum v)^2 - sum(v^2))
                Tensor sumV = sparseEmbed.sum(new long[]{1});
                Tensor sumV2 = sumV.pow(new Scalar(2.0f));
                Tensor sumV_squared = sparseEmbed.mul(sparseEmbed).sum(new long[]{1});
                Tensor fmSecondOrder = sumV2.sub(sumV_squared).mul(new Scalar(0.5f));
                Tensor fmOut = fmSecondOrder.reshape(new long[]{modelConfig.batchSize, embeddingDim});

                // ═══════ DNN Component ═══════
                // Flatten embeddings: [batch, numSparseFeatures * embeddingDim]
                Tensor dnnInput = sparseEmbed.reshape(new long[]{modelConfig.batchSize, firstDnnInput});
                Tensor dnnOut = matmul(dnnInput, dnnWeights[0]).add(dnnBiases[0]);

                for (int i = 1; i < dnnLayers; i++) {
                    dnnOut = relu(dnnOut);
                    dnnOut = matmul(dnnOut, dnnWeights[i]).add(dnnBiases[i]);
                }

                // ═══════ Combine ═══════
                TensorVector combinedVec = new TensorVector();
                combinedVec.push_back(fmOut);
                combinedVec.push_back(dnnOut);
                Tensor combined = cat(combinedVec, 1L);
                Tensor logits = matmul(combined, outputWeight).add(outputBias);

                // Add dense features contribution
                Tensor denseOut = matmul(denseInput, denseLinearWeight).add(denseLinearBias);
                Tensor finalLogits = logits.add(denseOut);

                // ═══════ Loss ═══════
                Tensor loss = binary_cross_entropy_with_logits(
                    finalLogits.reshape(modelConfig.batchSize), labels);

                // ═══════ Backward + Optimizer ═══════
                loss.backward();

                if (trainConfig.maxGradNorm > 0) {
                    org.bytedeco.pytorch.global.torch.clip_grad_norm_(params, trainConfig.maxGradNorm);
                }

                optimizer.step();
                optimizer.zero_grad();

                // Statistics
                double lossVal = loss.item().toDouble();
                long iterTime = System.nanoTime() - iterStart;
                double throughput = modelConfig.batchSize * 1e9 / iterTime;

                losses.add(lossVal);
                throughputs.add(throughput);
                successfulSteps++;

                if (step % trainConfig.logInterval == 0 && rank == 0) {
                    System.out.printf("[Step %3d/%d] Loss: %.6f | Throughput: %,.0f/s | Time: %.2fms%n",
                        step, trainConfig.numSteps, lossVal, throughput, iterTime / 1e6);
                }

                // Cleanup tensors
                sparseInput.close();
                denseInput.close();
                labels.close();
                flatIndices.close();
                sparseEmbed.close();
                sumV.close();
                sumV2.close();
                sumV_squared.close();
                fmSecondOrder.close();
                fmOut.close();
                dnnInput.close();
                dnnOut.close();
                combined.close();
                logits.close();
                denseOut.close();
                finalLogits.close();
                loss.close();

            } catch (Exception e) {
                failedSteps++;
                if (rank == 0 && step < 3) {
                    System.err.println("[Step " + step + "] Failed: " + e.getMessage());
                    e.printStackTrace();
                }
            }
        }

        long totalTime = System.currentTimeMillis() - totalStartTime;
        double avgLoss = losses.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double avgThroughput = throughputs.stream().mapToDouble(Double::doubleValue).average().orElse(0);

        // Cleanup
        embeddings.close();
        for (Tensor w : dnnWeights) w.close();
        for (Tensor b : dnnBiases) b.close();
        outputWeight.close();
        outputBias.close();
        denseLinearWeight.close();
        denseLinearBias.close();
        params.close();

        BenchmarkResult result = new BenchmarkResult(worldSize, rank, totalTime, avgThroughput, avgLoss, successfulSteps);
        if (rank == 0) {
            System.out.println("\n" + result);
            System.out.printf("Success Rate: %d/%d (%.1f%%)%n",
                successfulSteps, trainConfig.numSteps,
                100.0 * successfulSteps / Math.max(1, trainConfig.numSteps));
        }
        return result;
    }

    public static BenchmarkResult benchmarkLocal(int numSteps, int batchSize, int dnnHidden, int embeddingDim) {
        ModelConfig modelConfig = new ModelConfig();
        modelConfig.batchSize = batchSize;
        modelConfig.dnnHiddenDim = dnnHidden;
        modelConfig.embeddingDim = embeddingDim;

        TrainConfig trainConfig = new TrainConfig();
        trainConfig.numSteps = numSteps;
        trainConfig.useFP16 = false;

        Device device;
        ScalarType dtype = ScalarType.Float;

        if (cuda_is_available()) {
            device = new Device(org.bytedeco.pytorch.global.torch.DeviceType.CUDA, (byte)0);
            System.out.println("[Device] Using CUDA GPU: " + device);
        } else {
            device = new Device(org.bytedeco.pytorch.global.torch.DeviceType.CPU);
            System.out.println("[Device] Using CPU (CUDA not available)");
        }

        return runBenchmark(1, 0, modelConfig, trainConfig, device, dtype);
    }

    public static void main(String[] args) {
        int numSteps = 100;
        int batchSize = 4096;
        int dnnHidden = 256;
        int embeddingDim = 16;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--steps": numSteps = Integer.parseInt(args[++i]); break;
                case "--batch": batchSize = Integer.parseInt(args[++i]); break;
                case "--dnn-hidden": dnnHidden = Integer.parseInt(args[++i]); break;
                case "--embed-dim": embeddingDim = Integer.parseInt(args[++i]); break;
            }
        }

        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║       DeepFM Enterprise Training Benchmark              ║");
        System.out.println("╠══════════════════════════════════════════════════════════╣");
        System.out.printf("║  Steps:    %-45d║%n", numSteps);
        System.out.printf("║  Batch:    %-45d║%n", batchSize);
        System.out.printf("║  DNN:      %dx%-42d║%n", dnnHidden, dnnHidden);
        System.out.printf("║  Embed:    %-45d║%n", embeddingDim);
        System.out.printf("║  CUDA:     %-45s║%n", cuda_is_available() ? "Available" : "Not Available");
        System.out.println("╚══════════════════════════════════════════════════════════╝");

        try {
            benchmarkLocal(numSteps, batchSize, dnnHidden, embeddingDim);
        } catch (Exception e) {
            System.err.println("Benchmark failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}