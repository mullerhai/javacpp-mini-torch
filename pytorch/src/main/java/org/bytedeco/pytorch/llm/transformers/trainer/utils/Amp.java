/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
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
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.transformers.trainer.utils;

import org.bytedeco.pytorch.amp.AmpPrecision;
import org.bytedeco.pytorch.amp.AutocastContext;
import org.bytedeco.pytorch.amp.GradScaler;

/**
 * HuggingFace-style AMP helpers for the Trainer package.
 *
 * <p><b>Do not invent a second AMP stack.</b> This class only re-exports
 * {@link org.bytedeco.pytorch.amp} so transformers code can write
 * {@code trainer.utils.amp.autocast(...)} without colliding with the real
 * {@code org.bytedeco.pytorch.amp} package (a file named {@code amp.java}
 * would be a class named {@code amp} and clash with a sub-package of the
 * same name).
 */
public final class Amp {

    private Amp() {}

    /** Python {@code torch.cuda.amp.autocast(enabled=...)}. */
    public static AutocastContext autocast(boolean enabled) {
        AmpPrecision precision = enabled ? AmpPrecision.BF16 : AmpPrecision.FP32;
        return AutocastContext.create(null, precision);
    }

    public static AutocastContext autocast(AmpPrecision precision) {
        return AutocastContext.create(null, precision);
    }

    public static boolean isAvailable() {
        try {
            return org.bytedeco.pytorch.global.torch.cuda_is_available();
        } catch (Exception e) {
            return false;
        }
    }

    public static GradScaler gradScaler() {
        return GradScaler.createDefault();
    }

    public static GradScaler.Builder gradScalerBuilder() {
        return GradScaler.builder();
    }
}
