package org.bytedeco.pytorch.dataframe.faiss;

import java.lang.reflect.Method;

/**
 * Device selection for FAISS distance backends — CPU (scalar / vector-API),
 * CUDA (via javacpp-pytorch), and MPS (Apple Silicon via javacpp-pytorch).
 *
 * <p>Auto-detects each accelerator once; force with {@link #setPreferred(Device)}
 * or {@link #setBackendMode(BackendMode)}. Detection never throws; failures
 * resolve to the safest fallback (CPU scalar → CPU vector).
 */
public final class DeviceSelector {
    public enum Device {
        CPU,
        CUDA,
        MPS
    }

    /**
     * Distance-backend execution mode.
     * <ul>
     *   <li>{@link #AUTO}  — use CUDA/MPS if available, else Vector, else scalar CPU.</li>
     *   <li>{@link #CUDA}  — force CUDA, else CPU fallback.</li>
     *   <li>{@link #MPS}   — force MPS, else CPU fallback.</li>
     *   <li>{@link #VECTOR}— force jdk.incubator.vector SIMD on CPU.</li>
     *   <li>{@link #CPU}   — force scalar CPU (legacy).</li>
     * </ul>
     */
    public enum BackendMode {
        AUTO,
        CUDA,
        MPS,
        VECTOR,
        CPU
    }

    private static volatile Boolean cudaAvailable;
    private static volatile Boolean mpsAvailable;
    private static volatile Device preferred;            // null = auto
    private static volatile BackendMode backendMode = BackendMode.AUTO;
    private static volatile int cudaDeviceIndex = 0;
    private static volatile String lastProbeDetail = "unprobed";
    private static volatile String lastMpsProbeDetail = "unprobed";

    private DeviceSelector() {}

    public static void setPreferred(Device device) {
        preferred = device;
    }

    public static Device preferred() {
        return preferred;
    }

    public static void setBackendMode(BackendMode mode) {
        backendMode = mode == null ? BackendMode.AUTO : mode;
    }

    public static BackendMode backendMode() {
        return backendMode;
    }

    public static void setCudaDeviceIndex(int index) {
        cudaDeviceIndex = Math.max(0, index);
    }

    public static int cudaDeviceIndex() {
        return cudaDeviceIndex;
    }

    public static Device resolve() {
        if (preferred != null) {
            if (preferred == Device.CUDA && !isCudaAvailable()) return Device.CPU;
            if (preferred == Device.MPS && !isMpsAvailable()) return Device.CPU;
            return preferred;
        }
        if (isCudaAvailable()) return Device.CUDA;
        if (isMpsAvailable()) return Device.MPS;
        return Device.CPU;
    }

    public static boolean isCudaAvailable() {
        Boolean cached = cudaAvailable;
        if (cached != null) return cached;
        synchronized (DeviceSelector.class) {
            if (cudaAvailable != null) return cudaAvailable;
            cudaAvailable = probeCuda();
            return cudaAvailable;
        }
    }

    public static boolean isMpsAvailable() {
        Boolean cached = mpsAvailable;
        if (cached != null) return cached;
        synchronized (DeviceSelector.class) {
            if (mpsAvailable != null) return mpsAvailable;
            mpsAvailable = probeMps();
            return mpsAvailable;
        }
    }

    public static void resetCache() {
        synchronized (DeviceSelector.class) {
            cudaAvailable = null;
            mpsAvailable = null;
            lastProbeDetail = "unprobed";
            lastMpsProbeDetail = "unprobed";
        }
    }

    public static String lastProbeDetail() {
        return lastProbeDetail;
    }

    public static String lastMpsProbeDetail() {
        return lastMpsProbeDetail;
    }

    /**
     * Resolve the best {@link DistanceBackend} for the current configuration.
     * Honors {@link #backendMode} first, then device preference, then auto-detect.
     */
    public static DistanceBackend resolveBackend() {
        BackendMode mode = backendMode;
        if (mode == BackendMode.CPU) {
            return CpuDistanceBackend.INSTANCE;
        }
        if (mode == BackendMode.VECTOR) {
            return VectorCpuDistanceBackend.INSTANCE;
        }
        if (mode == BackendMode.CUDA) {
            if (isCudaAvailable()) return CudaDistanceBackend.INSTANCE;
            return VectorDistanceKernel.AVAILABLE
                ? VectorCpuDistanceBackend.INSTANCE
                : CpuDistanceBackend.INSTANCE;
        }
        if (mode == BackendMode.MPS) {
            if (isMpsAvailable()) return MpsDistanceBackend.INSTANCE;
            return VectorDistanceKernel.AVAILABLE
                ? VectorCpuDistanceBackend.INSTANCE
                : CpuDistanceBackend.INSTANCE;
        }
        // AUTO
        Device d = resolve();
        switch (d) {
            case CUDA: return CudaDistanceBackend.INSTANCE;
            case MPS:  return MpsDistanceBackend.INSTANCE;
            case CPU:
            default:
                return VectorDistanceKernel.AVAILABLE
                    ? VectorCpuDistanceBackend.INSTANCE
                    : CpuDistanceBackend.INSTANCE;
        }
    }

    // Cached Method handles resolved on first probe.
    private static volatile Method cachedCudaAvailable;
    private static volatile Method cachedTensorCuda;
    private static volatile Method cachedTensorMps;
    private static volatile Method cachedTensorTo;
    private static volatile java.lang.reflect.Constructor<?> cachedDeviceCtor;

    private static boolean probeCuda() {
        try {
            boolean hasCudaClasses = false;
            for (String cn : new String[]{
                "org.bytedeco.pytorch.cuda.global.torch_cuda",
                "org.bytedeco.pytorch.presets.torch_cuda"
            }) {
                try {
                    Class.forName(cn, false, DeviceSelector.class.getClassLoader());
                    hasCudaClasses = true;
                    break;
                } catch (Throwable ignored) {}
            }
            try {
                Class<?> torch = Class.forName("org.bytedeco.pytorch.global.torch");
                for (String name : new String[]{"cuda_is_available", "hasCUDA", "is_cuda_available"}) {
                    Method m = cachedCudaAvailable;
                    if (m == null || !m.getName().equals(name)) {
                        try { m = torch.getMethod(name); cachedCudaAvailable = m; }
                        catch (NoSuchMethodException ignored) { continue; }
                    }
                    Object r = m.invoke(null);
                    if (r instanceof Boolean b) {
                        lastProbeDetail = name + "=" + b;
                        return b;
                    }
                }
            } catch (ClassNotFoundException e) {
                lastProbeDetail = "torch class missing";
                return false;
            }
            try {
                org.bytedeco.pytorch.Tensor t =
                    org.bytedeco.pytorch.global.torch.tensor(new float[]{0f});
                try {
                    Method cudaM = cachedTensorCuda;
                    if (cudaM == null) {
                        try {
                            cudaM = t.getClass().getMethod("cuda");
                            cachedTensorCuda = cudaM;
                        } catch (NoSuchMethodException ns) {
                            cudaM = null;
                        }
                    }
                    if (cudaM != null) {
                        Object g = cudaM.invoke(t);
                        if (g instanceof org.bytedeco.pytorch.Tensor gt) {
                            boolean ok;
                            try { ok = gt.is_cuda(); } catch (Throwable ignored) { ok = true; }
                            try { gt.close(); } catch (Throwable ignored) {}
                            try { t.close(); } catch (Throwable ignored) {}
                            lastProbeDetail = "tensor.cuda() ok=" + ok;
                            return ok;
                        }
                    } else {
                        try {
                            Class<?> devCls = Class.forName("org.bytedeco.pytorch.Device");
                            java.lang.reflect.Constructor<?> ctor = cachedDeviceCtor;
                            if (ctor == null) {
                                ctor = devCls.getConstructor(String.class);
                                cachedDeviceCtor = ctor;
                            }
                            Object dev = ctor.newInstance("cuda:" + cudaDeviceIndex);
                            Method toM = cachedTensorTo;
                            if (toM == null) {
                                toM = t.getClass().getMethod("to", devCls);
                                cachedTensorTo = toM;
                            }
                            Object g = toM.invoke(t, dev);
                            boolean ok = g instanceof org.bytedeco.pytorch.Tensor gt && safeIsCuda(gt);
                            if (g instanceof AutoCloseable ac) try { ac.close(); } catch (Exception ignored) {}
                            try { t.close(); } catch (Throwable ignored) {}
                            lastProbeDetail = "tensor.to(cuda) ok=" + ok;
                            return ok;
                        } catch (Throwable e) {
                            lastProbeDetail = "to(cuda) failed: " + shortMsg(e);
                            try { t.close(); } catch (Throwable ignored) {}
                            return false;
                        }
                    }
                } catch (Throwable e) {
                    lastProbeDetail = "tensor.cuda() failed: " + shortMsg(e);
                    try { t.close(); } catch (Throwable ignored) {}
                    return false;
                }
                try { t.close(); } catch (Throwable ignored) {}
            } catch (Throwable e) {
                lastProbeDetail = "torch init failed: " + shortMsg(e);
                return false;
            }
            lastProbeDetail = hasCudaClasses
                ? "cuda classes present but no working device"
                : "no cuda path";
            return false;
        } catch (Throwable e) {
            lastProbeDetail = "outer: " + shortMsg(e);
            return false;
        }
    }

    private static boolean probeMps() {
        try {
            org.bytedeco.pytorch.Tensor t =
                org.bytedeco.pytorch.global.torch.tensor(new float[]{0f});
            try {
                Method mpsM = cachedTensorMps;
                if (mpsM == null) {
                    try {
                        mpsM = t.getClass().getMethod("mps");
                        cachedTensorMps = mpsM;
                    } catch (NoSuchMethodException ns) {
                        mpsM = null;
                    }
                }
                if (mpsM != null) {
                    Object g = mpsM.invoke(t);
                    if (g instanceof org.bytedeco.pytorch.Tensor gt) {
                        boolean ok = true;
                        try {
                            Method isMpsM = gt.getClass().getMethod("is_mps");
                            Object r = isMpsM.invoke(gt);
                            if (r instanceof Boolean b) ok = b;
                        } catch (Throwable ignored) {}
                        try { gt.close(); } catch (Throwable ignored) {}
                        lastMpsProbeDetail = "tensor.mps() ok=" + ok;
                        try { t.close(); } catch (Throwable ignored) {}
                        return ok;
                    }
                }
                // Fallback: try to(Device("mps"))
                try {
                    Class<?> devCls = Class.forName("org.bytedeco.pytorch.Device");
                    java.lang.reflect.Constructor<?> ctor = cachedDeviceCtor;
                    if (ctor == null) {
                        ctor = devCls.getConstructor(String.class);
                        cachedDeviceCtor = ctor;
                    }
                    Object dev = ctor.newInstance("mps");
                    Method toM = cachedTensorTo;
                    if (toM == null) {
                        toM = t.getClass().getMethod("to", devCls);
                        cachedTensorTo = toM;
                    }
                    Object g = toM.invoke(t, dev);
                    if (g instanceof AutoCloseable ac) try { ac.close(); } catch (Exception ignored) {}
                    try { t.close(); } catch (Throwable ignored) {}
                    lastMpsProbeDetail = "tensor.to(mps) ok=true";
                    return true;
                } catch (Throwable e) {
                    lastMpsProbeDetail = "to(mps) failed: " + shortMsg(e);
                    try { t.close(); } catch (Throwable ignored) {}
                    return false;
                }
            } catch (Throwable e) {
                lastMpsProbeDetail = "tensor.mps() failed: " + shortMsg(e);
                try { t.close(); } catch (Throwable ignored) {}
                return false;
            }
        } catch (Throwable e) {
            lastMpsProbeDetail = "torch init failed: " + shortMsg(e);
            return false;
        }
    }

    private static String shortMsg(Throwable e) {
        Throwable c = e;
        while (c.getCause() != null && c.getCause() != c) c = c.getCause();
        String m = c.getMessage();
        if (m == null || m.isEmpty()) m = c.getClass().getSimpleName();
        m = m.replace('\n', ' ');
        if (m.length() > 160) m = m.substring(0, 160) + "...";
        return m;
    }

    private static boolean safeIsCuda(org.bytedeco.pytorch.Tensor t) {
        try { return t.is_cuda(); } catch (Throwable e) { return false; }
    }

    public static String describe() {
        boolean cuda = isCudaAvailable();
        boolean mps = isMpsAvailable();
        Device r = resolve();
        return "cuda_available=" + cuda
            + " mps_available=" + mps
            + " preferred=" + preferred
            + " resolved=" + r
            + (r == Device.CUDA ? (" device=" + cudaDeviceIndex) : "")
            + " backend_mode=" + backendMode
            + " probe_cuda=" + lastProbeDetail
            + " probe_mps=" + lastMpsProbeDetail;
    }
}
