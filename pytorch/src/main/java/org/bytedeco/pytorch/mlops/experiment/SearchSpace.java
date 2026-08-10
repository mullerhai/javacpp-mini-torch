/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
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
package org.bytedeco.pytorch.mlops.experiment;
import org.bytedeco.pytorch.nn.*;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Hyperparameter search space definition.
 */
public class SearchSpace {

    private final Map<String, Parameter> parameters = new ConcurrentHashMap<>();

    /**
     * Add a categorical parameter.
     */
    public SearchSpace add(String name, Object... choices) {
        parameters.put(name, new CategoricalParameter(name, Arrays.asList(choices)));
        return this;
    }

    /**
     * Add a uniform parameter (continuous).
     */
    public SearchSpace addUniform(String name, double low, double high) {
        parameters.put(name, new UniformParameter(name, low, high));
        return this;
    }

    /**
     * Add a log-uniform parameter (for learning rates).
     */
    public SearchSpace addLogUniform(String name, double low, double high) {
        parameters.put(name, new LogUniformParameter(name, low, high));
        return this;
    }

    /**
     * Add an integer parameter.
     */
    public SearchSpace addInt(String name, int low, int high) {
        parameters.put(name, new IntParameter(name, low, high));
        return this;
    }

    /**
     * Add a log-integer parameter.
     */
    public SearchSpace addLogInt(String name, int low, int high) {
        parameters.put(name, new LogIntParameter(name, low, high));
        return this;
    }

    /**
     * Add a discrete parameter.
     */
    public SearchSpace addDiscrete(String name, double... values) {
        parameters.put(name, new DiscreteParameter(name, values));
        return this;
    }

    /**
     * Sample a parameter configuration.
     */
    public Map<String, Object> sample(int trialId) {
        Map<String, Object> result = new HashMap<>();
        Random rand = new Random(trialId);
        for (Parameter p : parameters.values()) {
            result.put(p.name(), p.sample(rand));
        }
        return result;
    }

    /**
     * Sample based on Bayesian optimization.
     */
    public Map<String, Object> sampleBayesian(List<HyperparameterOptimizer.Trial> trials,
                                               Map<String, Double> bestParams) {
        Map<String, Object> result = new HashMap<>();
        Random rand = new Random();

        for (Parameter p : parameters.values()) {
            if (bestParams.containsKey(p.name())) {
                // Exploit: small perturbation around best
                result.put(p.name(), p.sampleNear(bestParams.get(p.name()), rand));
            } else {
                // Explore: random sample
                result.put(p.name(), p.sample(rand));
            }
        }

        return result;
    }

    /**
     * Generate grid combinations.
     */
    public List<Map<String, Object>> gridCombinations() {
        List<Map<String, Object>> combinations = new ArrayList<>();
        List<String> names = new ArrayList<>(parameters.keySet());
        List<List<Object>> valueLists = new ArrayList<>();

        for (String name : names) {
            valueLists.add(parameters.get(name).gridValues());
        }

        // Cartesian product
        generateCombinations(combinations, names, valueLists, 0, new HashMap<>());

        return combinations;
    }

    private void generateCombinations(List<Map<String, Object>> result,
                                      List<String> names,
                                      List<List<Object>> values,
                                      int idx,
                                      Map<String, Object> current) {
        if (idx == names.size()) {
            result.add(new HashMap<>(current));
            return;
        }

        for (Object v : values.get(idx)) {
            current.put(names.get(idx), v);
            generateCombinations(result, names, values, idx + 1, current);
        }
    }

    // ============= Parameter Types =============

    public interface Parameter {
        String name();
        Object sample(Random rand);
        Object sampleNear(double value, Random rand);
        List<Object> gridValues();
    }

    public static class CategoricalParameter implements Parameter {
        private final String name;
        private final List<Object> choices;

        public CategoricalParameter(String name, List<Object> choices) {
            this.name = name;
            this.choices = choices;
        }

        @Override
        public String name() { return name; }

        @Override
        public Object sample(Random rand) {
            return choices.get(rand.nextInt(choices.size()));
        }

        @Override
        public Object sampleNear(double value, Random rand) {
            return choices.get(rand.nextInt(choices.size()));
        }

        @Override
        public List<Object> gridValues() { return choices; }
    }

    public static class UniformParameter implements Parameter {
        private final String name;
        private final double low;
        private final double high;

        public UniformParameter(String name, double low, double high) {
            this.name = name;
            this.low = low;
            this.high = high;
        }

        @Override
        public String name() { return name; }

        @Override
        public Object sample(Random rand) {
            return low + rand.nextDouble() * (high - low);
        }

        @Override
        public Object sampleNear(double value, Random rand) {
            double range = (high - low) * 0.1;
            double newValue = value + (rand.nextDouble() - 0.5) * range;
            return Math.max(low, Math.min(high, newValue));
        }

        @Override
        public List<Object> gridValues() {
            return Arrays.asList(low, (low + high) / 2, high);
        }
    }

    public static class LogUniformParameter implements Parameter {
        private final String name;
        private final double low;
        private final double high;

        public LogUniformParameter(String name, double low, double high) {
            this.name = name;
            this.low = Math.log(low);
            this.high = Math.log(high);
        }

        @Override
        public String name() { return name; }

        @Override
        public Object sample(Random rand) {
            return Math.exp(low + rand.nextDouble() * (high - low));
        }

        @Override
        public Object sampleNear(double value, Random rand) {
            double logValue = Math.log(value);
            double range = (high - low) * 0.1;
            double newLogValue = logValue + (rand.nextDouble() - 0.5) * range;
            return Math.exp(Math.max(low, Math.min(high, newLogValue)));
        }

        @Override
        public List<Object> gridValues() {
            return Arrays.asList(Math.exp(low), Math.exp((low + high) / 2), Math.exp(high));
        }
    }

    public static class IntParameter implements Parameter {
        private final String name;
        private final int low;
        private final int high;

        public IntParameter(String name, int low, int high) {
            this.name = name;
            this.low = low;
            this.high = high;
        }

        @Override
        public String name() { return name; }

        @Override
        public Object sample(Random rand) {
            return rand.nextInt(high - low + 1) + low;
        }

        @Override
        public Object sampleNear(double value, Random rand) {
            int intValue = (int) value;
            int newValue = intValue + rand.nextInt(3) - 1;
            return Math.max(low, Math.min(high, newValue));
        }

        @Override
        public List<Object> gridValues() {
            return Arrays.asList(low, (low + high) / 2, high);
        }
    }

    public static class LogIntParameter implements Parameter {
        private final String name;
        private final int low;
        private final int high;

        public LogIntParameter(String name, int low, int high) {
            this.name = name;
            this.low = low;
            this.high = high;
        }

        @Override
        public String name() { return name; }

        @Override
        public Object sample(Random rand) {
            double logLow = Math.log(low);
            double logHigh = Math.log(high);
            return (int) Math.round(Math.exp(logLow + rand.nextDouble() * (logHigh - logLow)));
        }

        @Override
        public Object sampleNear(double value, Random rand) {
            int newValue = (int) (value * (0.8 + rand.nextDouble() * 0.4));
            return Math.max(low, Math.min(high, newValue));
        }

        @Override
        public List<Object> gridValues() {
            return Arrays.asList(low, (int) Math.sqrt(low * high), high);
        }
    }

    public static class DiscreteParameter implements Parameter {
        private final String name;
        private final double[] values;

        public DiscreteParameter(String name, double... values) {
            this.name = name;
            this.values = values;
        }

        @Override
        public String name() { return name; }

        @Override
        public Object sample(Random rand) {
            return values[rand.nextInt(values.length)];
        }

        @Override
        public Object sampleNear(double value, Random rand) {
            int closest = 0;
            double minDist = Double.MAX_VALUE;
            for (int i = 0; i < values.length; i++) {
                double dist = Math.abs(values[i] - value);
                if (dist < minDist) {
                    minDist = dist;
                    closest = i;
                }
            }
            int idx = Math.max(0, Math.min(values.length - 1, closest + rand.nextInt(3) - 1));
            return values[idx];
        }

        @Override
        public List<Object> gridValues() {
            Object[] result = new Object[values.length];
            for (int i = 0; i < values.length; i++) result[i] = values[i];
            return Arrays.asList(result);
        }
    }
}
