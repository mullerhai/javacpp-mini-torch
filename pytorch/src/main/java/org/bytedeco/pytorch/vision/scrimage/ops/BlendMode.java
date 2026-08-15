package org.bytedeco.pytorch.vision.scrimage.ops;

import java.util.Objects;

/**
 * Blend modes (Photoshop / Pillow / scrimage coverage).
 *
 * <p>Each mode has a {@code combine(bR,bG,bB,tR,tG,tB)} that maps bottom and top
 * RGB (0..255) to a blended RGB. {@link Colors#blend} applies the result with
 * the {@code alpha} parameter against the bottom layer.
 *
 * <p>Formulas follow Pillow's {@code ImageChops} reference where applicable.
 */
public enum BlendMode {
    MULTIPLY {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{
                    Colors_clamp(bR * tR / 255),
                    Colors_clamp(bG * tG / 255),
                    Colors_clamp(bB * tB / 255)
            };
        }
    },
    SCREEN {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{
                    255 - Colors_clamp((255 - bR) * (255 - tR) / 255),
                    255 - Colors_clamp((255 - bG) * (255 - tG) / 255),
                    255 - Colors_clamp((255 - bB) * (255 - tB) / 255)
            };
        }
    },
    LIGHTEN {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{Math.max(bR, tR), Math.max(bG, tG), Math.max(bB, tB)};
        }
    },
    DARKEN {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{Math.min(bR, tR), Math.min(bG, tG), Math.min(bB, tB)};
        }
    },
    DIFFERENCE {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{Math.abs(bR - tR), Math.abs(bG - tG), Math.abs(bB - tB)};
        }
    },
    EXCLUSION {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{
                    Colors_clamp(bR + tR - 2 * bR * tR / 255),
                    Colors_clamp(bG + tG - 2 * bG * tG / 255),
                    Colors_clamp(bB + tB - 2 * bB * tB / 255)
            };
        }
    },
    OVERLAY {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{
                    Colors_blendOverlay(bR, tR),
                    Colors_blendOverlay(bG, tG),
                    Colors_blendOverlay(bB, tB)
            };
        }
    },
    SOFT_LIGHT {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{
                    Colors_blendSoftLight(bR, tR),
                    Colors_blendSoftLight(bG, tG),
                    Colors_blendSoftLight(bB, tB)
            };
        }
    },
    HARD_LIGHT {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{
                    Colors_blendHardLight(bR, tR),
                    Colors_blendHardLight(bG, tG),
                    Colors_blendHardLight(bB, tB)
            };
        }
    },
    COLOR_BURN {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{
                    Colors_blendBurn(bR, tR),
                    Colors_blendBurn(bG, tG),
                    Colors_blendBurn(bB, tB)
            };
        }
    },
    COLOR_DODGE {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{
                    Colors_blendDodge(bR, tR),
                    Colors_blendDodge(bG, tG),
                    Colors_blendDodge(bB, tB)
            };
        }
    },
    ADD {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{
                    Colors_clamp(bR + tR),
                    Colors_clamp(bG + tG),
                    Colors_clamp(bB + tB)
            };
        }
    },
    SUBTRACT {
        public int[] combine(int bR, int bG, int bB, int tR, int tG, int tB) {
            return new int[]{
                    Colors_clamp(bR - tR),
                    Colors_clamp(bG - tG),
                    Colors_clamp(bB - tB)
            };
        }
    };

    public abstract int[] combine(int bR, int bG, int bB, int tR, int tG, int tB);

    // package-private helpers accessed from Colors via package
    static int Colors_clamp(int v) {
        return v < 0 ? 0 : (v > 255 ? 255 : v);
    }

    static int Colors_blendOverlay(int b, int t) {
        if (b < 128) return Colors_clamp(2 * b * t / 255);
        return Colors_clamp(255 - 2 * (255 - b) * (255 - t) / 255);
    }

    static int Colors_blendSoftLight(int b, int t) {
        // Pegtop / Pillow
        return Colors_clamp(b + (2 * t - 255) * (b - b * b / 255) / 255);
    }

    static int Colors_blendHardLight(int b, int t) {
        // hard-light = overlay(b, t) with operands swapped
        return Colors_blendOverlay(t, b);
    }

    static int Colors_blendBurn(int b, int t) {
        if (t == 0) return 0;
        return Colors_clamp(255 - (255 - b) * 255 / t);
    }

    static int Colors_blendDodge(int b, int t) {
        if (t == 255) return 255;
        return Colors_clamp(b * 255 / (255 - t));
    }
}