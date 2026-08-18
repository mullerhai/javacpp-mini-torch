/*
 * Batch building utilities.
 */
package org.bytedeco.pytorch.llm.data;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

public final class BatchBuilder {
    private BatchBuilder() {}

    public static Tensor fromLongs(long[][] arr) {
        if (arr == null || arr.length == 0) return torch.zeros(new long[]{1, 1});
        int rows = arr.length;
        int cols = arr[0].length;
        Tensor t = torch.zeros(new long[]{rows, cols});
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t.select(0, i).select(1, j).fill_(new Scalar(arr[i][j]));
            }
        }
        return t;
    }

    public static Tensor fromFloats(float[][] arr) {
        if (arr == null || arr.length == 0) return torch.zeros(new long[]{1, 1});
        int rows = arr.length;
        int cols = arr[0].length;
        Tensor t = torch.zeros(new long[]{rows, cols});
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                t.select(0, i).select(1, j).fill_(new Scalar(arr[i][j]));
            }
        }
        return t;
    }
}
