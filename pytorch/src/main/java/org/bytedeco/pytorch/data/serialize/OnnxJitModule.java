/*
 * Copyright (c) 2026 MiniMax. All rights reserved.
 *
 * Helper around JitModule that exposes the underlying CompilationUnit
 * so that callers can invoke the free function `forward` via
 * `cu.find_function("forward").run(stack)` without going through
 * torch::jit::Method (which prepends `self` to the stack, requiring a
 * ClassType self argument in the function schema).
 *
 * The JitModule itself remains fully usable via JitModule.forward(stack)
 * for ONNX graphs whose forward signature starts with the module self
 * (i.e. graphs that were registered with `_self_` as the first input).
 */
package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.c10.*;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.IValue;
import org.bytedeco.pytorch.IValueVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.c10.QualifiedName;
import org.bytedeco.pytorch.jit.CompilationUnit;
import org.bytedeco.pytorch.jit.Function;
import org.bytedeco.pytorch.jit.JitModule;

public final class OnnxJitModule implements AutoCloseable {

    private final JitModule module;
    public JitModule module() { return module; }
    private final CompilationUnit cu;
    private final String forwardName;

    OnnxJitModule(JitModule module, CompilationUnit cu) {
        this(module, cu, "forward");
    }

    OnnxJitModule(JitModule module, CompilationUnit cu, String forwardName) {
        this.module = module;
        this.cu = cu;
        this.forwardName = forwardName;
    }

    /**
     * Invoke the compiled forward as a free function on the underlying
     * CompilationUnit. This bypasses torch::jit::Method::run's automatic
     * self-prepending so the stack matches the function schema exactly
     * (no implicit self argument).
     */
    public IValue forwardFreeFunction(IValueVector stack) {
        Function fn = cu.find_function(new QualifiedName(forwardName));
        if (fn == null) {
            throw new IllegalStateException(
                    "Function '" + forwardName + "' not found in compilation unit");
        }
        return fn.apply(stack);
    }

    /**
     * Invoke forward via JitModule.forward(stack). Requires the function
     * schema to accept `self` as the first argument. Most ONNX graphs
     * compiled via {@link OnnxToJitConverter} do NOT match this shape, so
     * prefer {@link #forwardFreeFunction(IValueVector)} instead.
     */
    public IValue forwardMethod(IValueVector stack) {
        return module.forward(stack);
    }

    @Override
    public void close() {
        if (module != null) {
            module.close();
        }
        if (cu != null) {
            cu.close();
        }
    }
}
