/*
 * ExperimentParameter - canonical parameter representation for MLOps sinks.
 *
 * <p>This is the parameter type expected by the various sink implementations
 * (ClearML, MLflow, Kubeflow). It is independent of the {@link Parameter}
 * class that lives next to it and represents a learnable tensor in the
 * geometric compatibility shims.
 */
package org.bytedeco.pytorch.deploy.integrations;

public class ExperimentParameter {
    public enum ParameterType {
        DOUBLE,
        LONG,
        BOOLEAN,
        STRING,
        BLOB
    }

    public final String key;
    public final Object value;
    public final ParameterType type;

    public ExperimentParameter(String key, double value) {
        this.key = key;
        this.value = value;
        this.type = ParameterType.DOUBLE;
    }

    public ExperimentParameter(String key, long value) {
        this.key = key;
        this.value = value;
        this.type = ParameterType.LONG;
    }

    public ExperimentParameter(String key, boolean value) {
        this.key = key;
        this.value = value;
        this.type = ParameterType.BOOLEAN;
    }

    public ExperimentParameter(String key, String value) {
        this.key = key;
        this.value = value;
        this.type = ParameterType.STRING;
    }

    public ExperimentParameter(String key, byte[] value) {
        this.key = key;
        this.value = value;
        this.type = ParameterType.BLOB;
    }

    /** Backwards-compatible type accessor. */
    public ParameterType getType() { return type; }
    public Object getValue() { return value; }
    public String getKey() { return key; }
}
