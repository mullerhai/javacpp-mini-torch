package org.bytedeco.pytorch.serving.onnxruntime;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Complete ONNX model metadata and tensor information.
 */
public final class ONNXModelInfo {

    private final List<String> inputNames;
    private final List<String> outputNames;
    private final List<ONNXTensorInfo> inputs;
    private final List<ONNXTensorInfo> outputs;
    private final String producerName;
    private final String graphName;
    private final String domain;
    private final String description;
    private final String version;
    private final long irVersion;

    private ONNXModelInfo(List<String> inputNames, List<String> outputNames,
                         List<ONNXTensorInfo> inputs, List<ONNXTensorInfo> outputs,
                         String producerName, String graphName, String domain,
                         String description, String version, long irVersion) {
        this.inputNames = inputNames;
        this.outputNames = outputNames;
        this.inputs = inputs;
        this.outputs = outputs;
        this.producerName = producerName;
        this.graphName = graphName;
        this.domain = domain;
        this.description = description;
        this.version = version;
        this.irVersion = irVersion;
    }

    public List<String> getInputNames() {
        return inputNames;
    }

    public List<String> getOutputNames() {
        return outputNames;
    }

    public List<ONNXTensorInfo> getInputs() {
        return inputs;
    }

    public List<ONNXTensorInfo> getOutputs() {
        return outputs;
    }

    public String getProducerName() {
        return producerName;
    }

    public String getGraphName() {
        return graphName;
    }

    public String getDomain() {
        return domain;
    }

    public String getDescription() {
        return description;
    }

    public String getVersion() {
        return version;
    }

    public long getIrVersion() {
        return irVersion;
    }

    /**
     * Get total parameter count (sum of all input tensor sizes).
     */
    public long getTotalInputSize() {
        long total = 0;
        for (ONNXTensorInfo input : inputs) {
            total += input.getNumElements();
        }
        return total;
    }

    /**
     * Get summary string.
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("ONNX Model: ").append(graphName).append("\n");
        sb.append("Producer: ").append(producerName).append("\n");
        sb.append("Version: ").append(version).append("\n");
        sb.append("IR Version: ").append(irVersion).append("\n");
        sb.append("Inputs (").append(inputs.size()).append("):\n");
        for (ONNXTensorInfo input : inputs) {
            sb.append("  - ").append(input.toString()).append("\n");
        }
        sb.append("Outputs (").append(outputs.size()).append("):\n");
        for (ONNXTensorInfo output : outputs) {
            sb.append("  - ").append(output.toString()).append("\n");
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return "ONNXModelInfo{" +
                "graphName='" + graphName + '\'' +
                ", producerName='" + producerName + '\'' +
                ", inputs=" + inputNames +
                ", outputs=" + outputNames +
                '}';
    }

    /**
     * Builder for ONNXModelInfo.
     */
    public static class Builder {
        private List<String> inputNames = new ArrayList<>();
        private List<String> outputNames = new ArrayList<>();
        private List<ONNXTensorInfo> inputs = new ArrayList<>();
        private List<ONNXTensorInfo> outputs = new ArrayList<>();
        private String producerName = "";
        private String graphName = "";
        private String domain = "";
        private String description = "";
        private String version = "1";
        private long irVersion = 0;

        public Builder inputNames(List<String> inputNames) {
            this.inputNames = inputNames;
            return this;
        }

        public Builder outputNames(List<String> outputNames) {
            this.outputNames = outputNames;
            return this;
        }

        public Builder addInput(ONNXTensorInfo input) {
            this.inputs.add(input);
            return this;
        }

        public Builder addOutput(ONNXTensorInfo output) {
            this.outputs.add(output);
            return this;
        }

        public Builder producerName(String producerName) {
            this.producerName = producerName != null ? producerName : "";
            return this;
        }

        public Builder graphName(String graphName) {
            this.graphName = graphName != null ? graphName : "";
            return this;
        }

        public Builder domain(String domain) {
            this.domain = domain != null ? domain : "";
            return this;
        }

        public Builder description(String description) {
            this.description = description != null ? description : "";
            return this;
        }

        public Builder version(String version) {
            this.version = version != null ? version : "1";
            return this;
        }

        public Builder irVersion(long irVersion) {
            this.irVersion = irVersion;
            return this;
        }

        public ONNXModelInfo build() {
            return new ONNXModelInfo(
                List.copyOf(inputNames),
                List.copyOf(outputNames),
                List.copyOf(inputs),
                List.copyOf(outputs),
                producerName,
                graphName,
                domain,
                description,
                version,
                irVersion
            );
        }
    }
}
