/*
 * Convenience factories for wiring AbTestMlopsBridge.
 */
package org.bytedeco.pytorch.deploy.integrations;

import org.bytedeco.pytorch.deploy.abtest.LayeredExperimentManager;

/**
 * Convenience: factories that wire up the bridge + sink.
 */
public final class MlopsBridgeFactory {
    private MlopsBridgeFactory() {}

    public static AbTestMlopsBridge bridgeSingle(MlopsSink sink, LayeredExperimentManager mgr,
                                                  org.bytedeco.pytorch.deploy.abtest.AbTestClient client) {
        sink.configure(SinkConfig.builder().endpointUrl("").build());
        return new AbTestMlopsBridge(MlopsClient.single(sink), mgr, client);
    }

    public static AbTestMlopsBridge bridgeBroadcast(LayeredExperimentManager mgr,
                                                     org.bytedeco.pytorch.deploy.abtest.AbTestClient client) {
        return new AbTestMlopsBridge(MlopsClient.broadcast(), mgr, client);
    }
}
