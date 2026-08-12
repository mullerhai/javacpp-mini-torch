/*
 * Registry of available MLOps sinks for broadcast mode.
 */
package org.bytedeco.pytorch.deploy.integrations;

import org.bytedeco.pytorch.deploy.integrations.clearml.ClearMLSink;
import org.bytedeco.pytorch.deploy.integrations.kubeflow.KubeflowSink;
import org.bytedeco.pytorch.deploy.integrations.mlflow.MlflowSink;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Registry of available sinks — picked by {@link MlopsClient}.
 */
public final class MlopsSinkRegistry {
    private static final List<MlopsSink> SINKS = new CopyOnWriteArrayList<>();

    private MlopsSinkRegistry() {}

    public static void register(MlopsSink sink) {
        SINKS.add(Objects.requireNonNull(sink));
    }

    public static List<MlopsSink> all() { return List.copyOf(SINKS); }

    public static MlopsSink byName(String simpleName) {
        for (MlopsSink s : SINKS) {
            if (s.platformName().equalsIgnoreCase(simpleName)) return s;
        }
        return null;
    }

    public static void registerIfMissing(MlopsSink sink) {
        if (byName(sink.platformName()) == null) SINKS.add(sink);
    }

    public static void registerIfMissing(MlopsCompatibility.Backend backend) {
        switch (backend) {
            case MLFLOW: registerIfMissing(new MlflowSink()); break;
            case CLEARML: registerIfMissing(new ClearMLSink()); break;
            case KUBEFLOW: registerIfMissing(new KubeflowSink()); break;
            default: break;
        }
    }

    public static void clear() { SINKS.clear(); }
}
