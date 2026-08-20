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
package org.bytedeco.pytorch.llm.transformers.trainer.data_collator;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Collator for sequence-to-sequence tasks (translation, summarisation).
 *
 * <p>Stacks encoder input ids and decoder input ids separately, with padding,
 * and creates decoder labels with -100 for positions that should be ignored in loss.
 */
public final class DataCollatorForSeq2Seq implements DataCollator {

    private final int encoderPadId;
    private final int decoderPadId;
    private final int labelIgnoreId;

    public DataCollatorForSeq2Seq(int encoderPadId, int decoderPadId, int labelIgnoreId) {
        this.encoderPadId = encoderPadId;
        this.decoderPadId = decoderPadId;
        this.labelIgnoreId = labelIgnoreId;
    }

    public DataCollatorForSeq2Seq() {
        this(0, 0, -100);
    }

    @Override
    public List<Map<String, Object>> collate_batch(List<Map<String, Object>> features) {
        if (features.isEmpty()) return List.of();

        int maxEncLen = 0, maxDecLen = 0;
        for (Map<String, Object> f : features) {
            Object enc = f.get("input_ids");
            Object dec = f.get("decoder_input_ids");
            if (enc instanceof int[] a) maxEncLen = Math.max(maxEncLen, a.length);
            if (enc instanceof long[] a) maxEncLen = Math.max(maxEncLen, a.length);
            if (dec instanceof int[] a) maxDecLen = Math.max(maxDecLen, a.length);
            if (dec instanceof long[] a) maxDecLen = Math.max(maxDecLen, a.length);
        }

        int bs = features.size();
        long[][] encIds = new long[bs][maxEncLen];
        long[][] decIds = new long[bs][maxDecLen];
        long[][] labels = new long[bs][maxDecLen];

        for (int i = 0; i < bs; i++) {
            Map<String, Object> f = features.get(i);
            fill(encIds[i], getIntArray(f.get("input_ids")), encoderPadId);
            fill(decIds[i], getIntArray(f.get("decoder_input_ids")), decoderPadId);
            fill(labels[i], getIntArray(f.get("labels")), labelIgnoreId);
        }

        return List.of(Map.of(
                "input_ids", encIds,
                "decoder_input_ids", decIds,
                "labels", labels
        ));
    }

    private static void fill(long[] out, int[] src, long padId) {
        if (src == null) return;
        int n = Math.min(src.length, out.length);
        for (int i = 0; i < n; i++) out[i] = src[i];
        for (int i = n; i < out.length; i++) out[i] = padId;
    }

    private static int[] getIntArray(Object o) {
        if (o instanceof int[] a) return a;
        if (o instanceof long[] a) {
            int[] r = new int[a.length];
            for (int i = 0; i < a.length; i++) r[i] = (int) a[i];
            return r;
        }
        return null;
    }
}
