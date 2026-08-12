/*
 * Daft 列类型 — Python {@code DataType.int64()}, {@code DataType.string()} 等价.
 *
 * 支持 int8/16/32/64, float32/64, bool, string, binary, date, timestamp,
 * list, struct, image, audio, video, url, embedding, document, pointcloud, mesh.
 */
package org.bytedeco.pytorch.utils.daft.expr;

import java.util.Locale;
import java.util.Objects;

/**
 * Daft column data type.
 */
public final class DataType {

    public enum Kind {
        INT8, INT16, INT32, INT64,
        FLOAT32, FLOAT64,
        BOOL, STRING, BINARY,
        DATE, TIMESTAMP,
        LIST, STRUCT,
        IMAGE, AUDIO, VIDEO, URL, EMBEDDING, DOCUMENT, POINTCLOUD, MESH,
        NULL
    }

    public final Kind kind;
    public final DataType elementType; // for LIST

    private DataType(Kind kind) { this(kind, null); }

    private DataType(Kind kind, DataType elementType) {
        this.kind = Objects.requireNonNull(kind);
        this.elementType = elementType;
    }

    public static DataType int8() { return new DataType(Kind.INT8); }
    public static DataType int16() { return new DataType(Kind.INT16); }
    public static DataType int32() { return new DataType(Kind.INT32); }
    public static DataType int64() { return new DataType(Kind.INT64); }
    public static DataType float32() { return new DataType(Kind.FLOAT32); }
    public static DataType float64() { return new DataType(Kind.FLOAT64); }
    public static DataType bool() { return new DataType(Kind.BOOL); }
    public static DataType string() { return new DataType(Kind.STRING); }
    public static DataType binary() { return new DataType(Kind.BINARY); }
    public static DataType date() { return new DataType(Kind.DATE); }
    public static DataType timestamp() { return new DataType(Kind.TIMESTAMP); }
    public static DataType image() { return new DataType(Kind.IMAGE); }
    public static DataType audio() { return new DataType(Kind.AUDIO); }
    public static DataType video() { return new DataType(Kind.VIDEO); }
    public static DataType url() { return new DataType(Kind.URL); }
    public static DataType embedding() { return new DataType(Kind.EMBEDDING); }
    public static DataType document() { return new DataType(Kind.DOCUMENT); }
    public static DataType pointcloud() { return new DataType(Kind.POINTCLOUD); }
    public static DataType mesh() { return new DataType(Kind.MESH); }
    public static DataType list(DataType element) {
        return new DataType(Kind.LIST, Objects.requireNonNull(element));
    }

    /**
     * Convert a {@link org.bytedeco.pytorch.dataframe.Column.DType} to a Daft {@link DataType}.
     */
    public static DataType fromColumn(org.bytedeco.pytorch.dataframe.Column.DType dtype) {
        if (dtype == null) return new DataType(Kind.NULL);
        switch (dtype.name()) {
            case "INT8": return int8();
            case "INT16": return int16();
            case "INT32": return int32();
            case "INT64": return int64();
            case "FLOAT32": return float32();
            case "FLOAT64": return float64();
            case "BOOLEAN":
            case "BOOL": return bool();
            case "STRING": return string();
            default: return new DataType(Kind.STRING);
        }
    }    @Override
    public String toString() {
        if (elementType != null) return "list<" + elementType + ">";
        return kind.name().toLowerCase(Locale.ROOT);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof DataType)) return false;
        DataType other = (DataType) o;
        return this.kind == other.kind
                && Objects.equals(this.elementType, other.elementType);
    }

    @Override
    public int hashCode() {
        return Objects.hash(kind, elementType);
    }
}
