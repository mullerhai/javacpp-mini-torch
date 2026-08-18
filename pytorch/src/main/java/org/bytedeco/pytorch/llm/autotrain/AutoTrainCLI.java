/*
 * Copyright (C) 2020-2026 the Java port of LLM-Finetuning project authors.
 *
 * Apache License 2.0.
 */
package org.bytedeco.pytorch.llm.autotrain;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Mirror of Hugging Face {@code autotrain-advanced} CLI. We don't actually invoke the binary;
 * we expose {@link #command(String...)} for the tutorials that document the command line.
 */
public final class AutoTrainCLI {

    private final Path python;

    public AutoTrainCLI() { this(Path.of("python")); }
    public AutoTrainCLI(Path python) { this.python = python; }

    public List<String> command(String... args) {
        List<String> cmd = new ArrayList<>();
        cmd.add(python.toString());
        cmd.add("-m");
        cmd.add("autotrain.trainers.clm");
        for (String a : args) cmd.add(a);
        return cmd;
    }

    public int run(String... args) {
        try {
            ProcessBuilder pb = new ProcessBuilder(command(args));
            pb.redirectErrorStream(true);
            Process p = pb.start();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
                String line;
                while ((line = br.readLine()) != null) System.out.println(line);
            }
            return p.waitFor();
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public static String documentedCmd(String model, String dataset, String outputDir) {
        return "autotrain llm --train --model " + model + " --train-data " + dataset +
                " --text-column text --lr 2e-4 --batch-size 1 --epochs 3 --output-dir " + outputDir;
    }
}