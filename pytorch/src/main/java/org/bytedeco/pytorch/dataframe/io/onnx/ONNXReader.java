package org.bytedeco.pytorch.dataframe.io.onnx;

import org.bytedeco.pytorch.serving.onnxruntime.ONNXModelInfo;
import org.bytedeco.pytorch.serving.onnxruntime.ONNXSession;
import org.bytedeco.pytorch.serving.onnxruntime.ONNXTensorInfo;

import java.io.IOException;
import java.io.InputStream;
import java.util.*;

/**
 * ONNX model file reader with schema inference support.
 *
 * <h2>Supported Formats</h2>
 * <ul>
 *   <li>.onnx - Standard ONNX model files</li>
 * </ul>
 *
 * <h2>Usage</h2>
 *
 * <pre>{@code
 * // Read ONNX model metadata
 * DataFrame df = DataFrame.read()
 *     .format("onnx")
 *     .load("model.onnx");
 *
 * // Get schema
 * StructType schema = df.schema();
 * schema.print();
 *
 * // Show model info
 * df.show();
 * }</pre>
 */
public class ONNXReader {

    private ONNXSession session;
    private ONNXModelInfo modelInfo;
    private String sourcePath;
    private boolean closed;

    public ONNXReader() {}

    /**
     * Load ONNX model from path.
     */
    public ONNXReader load(String path) throws IOException {
        this.sourcePath = path;
        try {
            this.session = ONNXSession.load(path);
            this.modelInfo = session.getModelInfo();
        } catch (Exception e) {
            throw new IOException("Failed to load ONNX model: " + e.getMessage(), e);
        }
        return this;
    }

    /**
     * Load ONNX model from input stream.
     */
    public ONNXReader load(InputStream is) throws IOException {
        throw new UnsupportedOperationException(
            "ONNX models cannot be loaded from InputStream. Use load(Path) or load(byte[])."
        );
    }

    /**
     * Load ONNX model from bytes.
     */
    public ONNXReader load(byte[] bytes) throws IOException {
        try {
            this.session = ONNXSession.load(bytes);
            this.modelInfo = session.getModelInfo();
            this.sourcePath = "<bytes>";
        } catch (Exception e) {
            throw new IOException("Failed to load ONNX model from bytes: " + e.getMessage(), e);
        }
        return this;
    }

    public StructType schema() {
        if (modelInfo == null) {
            throw new IllegalStateException("No model loaded");
        }
        return extractSchema();
    }

    private StructType extractSchema() {
        var fields = new ArrayList<StructField>();

        // Add input fields
        for (ONNXTensorInfo input : modelInfo.getInputs()) {
            String name = input.getName();
            DataType dtype = onnxToDataType(input.getElementType());
            fields.add(new StructField("input_" + name, dtype));
        }

        // Add output fields
        for (ONNXTensorInfo output : modelInfo.getOutputs()) {
            String name = output.getName();
            DataType dtype = onnxToDataType(output.getElementType());
            fields.add(new StructField("output_" + name, dtype));
        }

        // Add metadata fields
        fields.add(new StructField("producer_name", DataType.StringType));
        fields.add(new StructField("graph_name", DataType.StringType));
        fields.add(new StructField("version", DataType.StringType));
        fields.add(new StructField("ir_version", DataType.LongType));

        return new StructType(fields);
    }

    private DataType onnxToDataType(ai.onnxruntime.OnnxJavaType onnxType) {
        if (onnxType == null) return DataType.BinaryType;

        switch (onnxType) {
            case FLOAT:
            case DOUBLE:
                return DataType.DoubleType;
            case INT64:
                return DataType.LongType;
            case INT32:
            case INT16:
                return DataType.IntegerType;
            case BOOL:
                return DataType.BooleanType;
            case STRING:
                return DataType.StringType;
            case UINT8:
            case INT8:
                return DataType.ByteType;
            default:
                return DataType.BinaryType;
        }
    }

    public List<Row> read(int maxRows) {
        if (modelInfo == null) {
            throw new IllegalStateException("No model loaded");
        }

        var rows = new ArrayList<Row>();
        int limit = maxRows > 0 ? maxRows : 1;

        for (int i = 0; i < limit; i++) {
            var values = new ArrayList<Object>();

            // Input tensor info
            for (ONNXTensorInfo input : modelInfo.getInputs()) {
                values.add(input.getShapeString());
            }

            // Output tensor info
            for (ONNXTensorInfo output : modelInfo.getOutputs()) {
                values.add(output.getShapeString());
            }

            // Metadata
            values.add(modelInfo.getProducerName());
            values.add(modelInfo.getGraphName());
            values.add(modelInfo.getVersion());
            values.add(modelInfo.getIrVersion());

            rows.add(new Row(values));
        }

        return rows;
    }

    public void printSchema() {
        if (modelInfo == null) {
            System.out.println("No model loaded");
            return;
        }

        System.out.println("root");
        System.out.println(" |-- model: struct (nullable = true)");
        System.out.println(" |    |-- producer_name: string");
        System.out.println(" |    |-- graph_name: string");
        System.out.println(" |    |-- version: string");
        System.out.println(" |    |-- ir_version: long");
        System.out.println(" |");
        System.out.println(" |-- inputs: array (nullable = true)");
        System.out.println(" |    |-- element: struct (contains nested information)");
        System.out.println(" |    |    |-- name: string");
        System.out.println(" |    |    |-- type: string");
        System.out.println(" |    |    |-- shape: string");

        System.out.println(" |    ");
        System.out.println(" |-- outputs: array (nullable = true)");
        System.out.println(" |    |-- element: struct (contains nested information)");
        System.out.println(" |    |    |-- name: string");
        System.out.println(" |    |    |-- type: string");
        System.out.println(" |    |    |-- shape: string");

        System.out.println();
        System.out.println("Model Inputs (" + modelInfo.getInputs().size() + "):");
        for (ONNXTensorInfo input : modelInfo.getInputs()) {
            System.out.printf("   %s: %s %s%n", input.getName(), input.getElementTypeString(), input.getShapeString());
        }

        System.out.println();
        System.out.println("Model Outputs (" + modelInfo.getOutputs().size() + "):");
        for (ONNXTensorInfo output : modelInfo.getOutputs()) {
            System.out.printf("   %s: %s %s%n", output.getName(), output.getElementTypeString(), output.getShapeString());
        }
    }

    public void show() {
        printSchema();
    }

    public void close() throws IOException {
        if (!closed && session != null) {
            closed = true;
            session.close();
        }
    }

    /**
     * Get ONNX session for advanced operations.
     */
    public ONNXSession getSession() {
        return session;
    }

    /**
     * Get model info.
     */
    public ONNXModelInfo getModelInfo() {
        return modelInfo;
    }

    /**
     * Get input tensor information.
     */
    public List<ONNXTensorInfo> getInputs() {
        return modelInfo != null ? modelInfo.getInputs() : Collections.emptyList();
    }

    /**
     * Get output tensor information.
     */
    public List<ONNXTensorInfo> getOutputs() {
        return modelInfo != null ? modelInfo.getOutputs() : Collections.emptyList();
    }

    // Minimal type system for ONNX schema
    public enum DataType {
        NullType, BinaryType, BooleanType, ByteType, ShortType,
        IntegerType, LongType, FloatType, DoubleType, StringType,
        TimestampType, DateType, TimeType, ArrayType, MapType, StructType
    }

    public static class StructType {
        private final List<StructField> fields;
        public StructType(List<StructField> fields) { this.fields = fields; }
        public List<StructField> fields() { return fields; }
        public void print() {
            System.out.println("struct");
            for (StructField f : fields) {
                System.out.println(" |-- " + f.name() + ": " + f.dataType());
            }
        }
    }

    public static class StructField {
        private final String name;
        private final DataType dataType;
        public StructField(String name, DataType dataType) {
            this.name = name;
            this.dataType = dataType;
        }
        public String name() { return name; }
        public DataType dataType() { return dataType; }
    }

    public static class Row {
        private final List<Object> values;
        public Row(List<Object> values) { this.values = values; }
        public Object get(int i) { return values.get(i); }
        public int size() { return values.size(); }
    }
}
