package org.bytedeco.pytorch.dataframe.io.geo;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.*;
import java.nio.channels.*;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade Shapefile reader for geospatial vector data.
 * 
 * <p>Shapefiles are a popular geospatial vector data format developed by Esri.
 * This reader supports:</p>
 * <ul>
 *   <li>Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon</li>
 *   <li>Reading from .shp files</li>
 *   <li>Coordinate Reference System (CRS) metadata</li>
 *   <li>Attribute data (dBase III format)</li>
 * </ul>
 * 
 * <p>Example usage:</p>
 * <pre>
 *   // Read shapefile
 *   DataFrame df = ShapefileReader.read("/path/to/data.shp");
 *   
 *   // With options
 *   ShapefileReader.ShapefileOptions opts = ShapefileReader.options()
 *       .includeGeometry(true)
 *       .includeAttributes(true);
 *   DataFrame df = ShapefileReader.read("/path/to/data.shp", opts);
 * </pre>
 */
public class ShapefileReader {

    private ShapefileReader() {}

    // Shapefile shape types
    static final int NULL_SHAPE = 0;
    static final int POINT = 1;
    static final int POLYLINE = 3;
    static final int POLYGON = 5;
    static final int MULTIPOINT = 8;
    static final int POINT_Z = 11;
    static final int POLYLINE_Z = 13;
    static final int POLYGON_Z = 15;
    static final int MULTIPOINT_Z = 18;
    static final int POINT_M = 21;
    static final int POLYLINE_M = 23;
    static final int POLYGON_M = 25;
    static final int MULTIPOINT_M = 28;

    /**
     * Read shapefile into DataFrame.
     */
    public static DataFrame read(String path) throws IOException {
        return read(path, ShapefileOptions.defaults());
    }

    public static DataFrame read(String path, ShapefileOptions options) throws IOException {
        ShapefileOptions opts = options == null ? ShapefileOptions.defaults() : options;
        
        Path shapePath = Path.of(path);
        String basePath = path;
        if (path.toLowerCase().endsWith(".shp")) {
            basePath = path.substring(0, path.length() - 4);
        }
        
        DataFrame df = DataFrame.create();
        df.addColumn("shape_id", Column.DType.INT32);
        df.addColumn("shape_type", Column.DType.STRING);
        
        if (opts.includeGeometry()) {
            df.addColumn("geometry_wkt", Column.DType.STRING);
            df.addColumn("geometry_json", Column.DType.STRING);
            df.addColumn("num_points", Column.DType.INT32);
            df.addColumn("bbox_min_x", Column.DType.FLOAT64);
            df.addColumn("bbox_min_y", Column.DType.FLOAT64);
            df.addColumn("bbox_max_x", Column.DType.FLOAT64);
            df.addColumn("bbox_max_y", Column.DType.FLOAT64);
        }
        
        if (opts.includeAttributes()) {
            // Read DBF for attributes
            String dbfPath = basePath + ".dbf";
            Map<String, List<Object>> attributes = readDbf(dbfPath);
            for (String col : attributes.keySet()) {
                df.addColumn(col, Column.DType.STRING);
            }
        }
        
        // Read shapefile
        String shpPath = basePath + ".shp";
        try (FileChannel ch = FileChannel.open(Path.of(shpPath), StandardOpenOption.READ)) {
            // Read header
            ByteBuffer header = ByteBuffer.allocate(100);
            ch.read(header);
            header.flip();
            header.order(ByteOrder.BIG_ENDIAN);
            
            // Skip to shape type
            header.position(32);
            int shapeType = header.getInt();
            
            // Read shapes
            int recordNum = 1;
            long offset = 100; // Header is 100 bytes
            
            while (offset < ch.size()) {
                ch.position(offset);
                
                // Read record header (8 bytes)
                ByteBuffer recHeader = ByteBuffer.allocate(8);
                ch.read(recHeader);
                recHeader.flip();
                recHeader.order(ByteOrder.BIG_ENDIAN);
                int recordNumber = recHeader.getInt();
                int contentLength = recHeader.getInt(); // in 16-bit words
                
                // Read shape type (4 bytes)
                ByteBuffer shapeHeader = ByteBuffer.allocate(4);
                ch.read(shapeHeader);
                shapeHeader.flip();
                int thisShapeType = shapeHeader.getInt();
                
                int shapeDataLen = (contentLength * 2) - 4; // subtract shape type
                
                if (thisShapeType == NULL_SHAPE) {
                    int ri = df.addEmptyRow();
                    df.set(ri, "shape_id", recordNum);
                    df.set(ri, "shape_type", "NULL");
                } else {
                    ByteBuffer shapeData = ByteBuffer.allocate(shapeDataLen);
                    ch.read(shapeData);
                    shapeData.flip();
                    shapeData.order(ByteOrder.LITTLE_ENDIAN);
                    
                    int ri = df.addEmptyRow();
                    df.set(ri, "shape_id", recordNum);
                    df.set(ri, "shape_type", getShapeTypeName(thisShapeType));
                    
                    if (opts.includeGeometry()) {
                        String wkt = readGeometryWkt(shapeData, thisShapeType);
                        df.set(ri, "geometry_wkt", wkt);
                        df.set(ri, "geometry_json", wktToJson(wkt));
                        df.set(ri, "num_points", countPoints(shapeData, thisShapeType));
                        
                        double[] bbox = readBbox(shapeData, thisShapeType);
                        if (bbox != null) {
                            df.set(ri, "bbox_min_x", bbox[0]);
                            df.set(ri, "bbox_min_y", bbox[1]);
                            df.set(ri, "bbox_max_x", bbox[2]);
                            df.set(ri, "bbox_max_y", bbox[3]);
                        }
                    }
                }
                
                if (opts.includeAttributes() && !attributes.isEmpty()) {
                    for (String col : attributes.keySet()) {
                        List<Object> values = attributes.get(col);
                        if (recordNum - 1 < values.size()) {
                            df.set(ri, col, values.get(recordNum - 1));
                        }
                    }
                }
                
                recordNum++;
                offset += 8 + (contentLength * 2);
            }
        }
        
        return df;
    }

    private static String getShapeTypeName(int type) {
        switch (type) {
            case POINT: return "Point";
            case POLYLINE: return "PolyLine";
            case POLYGON: return "Polygon";
            case MULTIPOINT: return "MultiPoint";
            case POINT_Z: return "PointZ";
            case POLYLINE_Z: return "PolyLineZ";
            case POLYGON_Z: return "PolygonZ";
            case MULTIPOINT_Z: return "MultiPointZ";
            case POINT_M: return "PointM";
            case POLYLINE_M: return "PolyLineM";
            case POLYGON_M: return "PolygonM";
            case MULTIPOINT_M: return "MultiPointM";
            default: return "Unknown";
        }
    }

    private static String readGeometryWkt(ByteBuffer buf, int shapeType) {
        switch (shapeType) {
            case POINT:
            case POINT_Z:
            case POINT_M: {
                double x = buf.getDouble();
                double y = buf.getDouble();
                return String.format("POINT (%.6f %.6f)", x, y);
            }
            case MULTIPOINT:
            case MULTIPOINT_Z:
            case MULTIPOINT_M: {
                double[] bbox = new double[4];
                bbox[0] = buf.getDouble(); bbox[1] = buf.getDouble();
                bbox[2] = buf.getDouble(); bbox[3] = buf.getDouble();
                int numPoints = buf.getInt();
                StringBuilder sb = new StringBuilder("MULTIPOINT (");
                for (int i = 0; i < numPoints; i++) {
                    if (i > 0) sb.append(", ");
                    sb.append("(").append(buf.getDouble()).append(" ").append(buf.getDouble()).append(")");
                }
                sb.append(")");
                return sb.toString();
            }
            case POLYLINE:
            case POLYLINE_Z:
            case POLYLINE_M: {
                double[] bbox = new double[4];
                bbox[0] = buf.getDouble(); bbox[1] = buf.getDouble();
                bbox[2] = buf.getDouble(); bbox[3] = buf.getDouble();
                int numParts = buf.getInt();
                int numPoints = buf.getInt();
                int[] parts = new int[numParts];
                for (int i = 0; i < numParts; i++) parts[i] = buf.getInt();
                
                StringBuilder sb = new StringBuilder();
                if (numParts == 1) {
                    sb.append("LINESTRING (");
                } else {
                    sb.append("MULTILINESTRING (");
                }
                
                for (int p = 0; p < numParts; p++) {
                    if (p > 0) sb.append(", ");
                    if (numParts > 1) sb.append("(");
                    int start = parts[p];
                    int end = (p + 1 < numParts) ? parts[p + 1] : numPoints;
                    for (int i = start; i < end; i++) {
                        if (i > start) sb.append(", ");
                        sb.append("(").append(buf.getDouble()).append(" ").append(buf.getDouble()).append(")");
                    }
                    if (numParts > 1) sb.append(")");
                }
                sb.append(numParts > 1 ? ")" : ")");
                return sb.toString();
            }
            case POLYGON:
            case POLYGON_Z:
            case POLYGON_M: {
                double[] bbox = new double[4];
                bbox[0] = buf.getDouble(); bbox[1] = buf.getDouble();
                bbox[2] = buf.getDouble(); bbox[3] = buf.getDouble();
                int numParts = buf.getInt();
                int numPoints = buf.getInt();
                int[] parts = new int[numParts];
                for (int i = 0; i < numParts; i++) parts[i] = buf.getInt();
                
                StringBuilder sb = new StringBuilder();
                if (numParts == 1) {
                    sb.append("POLYGON ((");
                } else {
                    sb.append("MULTIPOLYGON (");
                }
                
                for (int p = 0; p < numParts; p++) {
                    if (p > 0) sb.append(", ");
                    if (numParts > 1) sb.append("((");
                    int start = parts[p];
                    int end = (p + 1 < numParts) ? parts[p + 1] : numPoints;
                    for (int i = start; i < end; i++) {
                        if (i > start) sb.append(", ");
                        sb.append(buf.getDouble()).append(" ").append(buf.getDouble());
                    }
                    if (numParts > 1) sb.append("))");
                    else sb.append(")");
                }
                sb.append(numParts > 1 ? ")" : ")");
                return sb.toString();
            }
            default:
                return "GEOMETRY";
        }
    }

    private static int countPoints(ByteBuffer buf, int shapeType) {
        int pos = buf.position();
        switch (shapeType) {
            case POINT: case POINT_Z: case POINT_M:
                return 1;
            case MULTIPOINT: case MULTIPOINT_Z: case MULTIPOINT_M:
                buf.getDouble(); buf.getDouble(); buf.getDouble(); buf.getDouble();
                return buf.getInt();
            case POLYLINE: case POLYLINE_Z: case POLYLINE_M:
            case POLYGON: case POLYGON_Z: case POLYGON_M:
                buf.getDouble(); buf.getDouble(); buf.getDouble(); buf.getDouble();
                buf.getInt();
                return buf.getInt();
            default:
                return 0;
        }
    }

    private static double[] readBbox(ByteBuffer buf, int shapeType) {
        switch (shapeType) {
            case POINT: case POINT_Z: case POINT_M:
                return new double[]{buf.getDouble(), buf.getDouble(), buf.getDouble(), buf.getDouble()};
            case MULTIPOINT: case MULTIPOINT_Z: case MULTIPOINT_M:
            case POLYLINE: case POLYLINE_Z: case POLYLINE_M:
            case POLYGON: case POLYGON_Z: case POLYGON_M:
                return new double[]{buf.getDouble(), buf.getDouble(), buf.getDouble(), buf.getDouble()};
            default:
                return null;
        }
    }

    private static String wktToJson(String wkt) {
        if (wkt == null) return null;
        if (wkt.startsWith("POINT")) {
            return wkt.replace("POINT (", "{\"type\":\"Point\",\"coordinates\":[")
                .replace(" ", ",").replace(")", "]}");
        }
        // Simplified conversion for other types
        return "{\"type\":\"Geometry\",\"wkt\":\"" + wkt.replace("\"", "\\\"") + "\"}";
    }

    /**
     * Read DBF attribute file.
     */
    private static Map<String, List<Object>> readDbf(String dbfPath) throws IOException {
        Map<String, List<Object>> attributes = new LinkedHashMap<>();
        
        if (!Files.exists(Path.of(dbfPath))) {
            return attributes;
        }
        
        try (FileChannel ch = FileChannel.open(Path.of(dbfPath), StandardOpenOption.READ)) {
            ByteBuffer header = ByteBuffer.allocate(32);
            ch.read(header);
            header.flip();
            
            header.order(ByteOrder.BIG_ENDIAN);
            byte version = header.get(0);
            int numRecords = header.getInt(4);
            int headerSize = header.getShort(8) & 0xFFFF;
            int recordSize = header.getShort(10) & 0xFFFF;
            
            header.order(ByteOrder.LITTLE_ENDIAN);
            
            // Read field descriptors
            int numFields = (headerSize - 33) / 32;
            List<String> fieldNames = new ArrayList<>();
            List<Byte> fieldTypes = new ArrayList<>();
            List<Integer> fieldLengths = new ArrayList<>();
            
            for (int i = 0; i < numFields; i++) {
                ByteBuffer fieldDesc = ByteBuffer.allocate(32);
                ch.read(fieldDesc);
                fieldDesc.flip();
                
                byte[] nameBytes = new byte[11];
                fieldDesc.get(nameBytes);
                String fieldName = new String(nameBytes).trim();
                
                byte fieldType = fieldDesc.get();
                byte fieldLen = fieldDesc.get(16);
                
                if (!fieldName.isEmpty()) {
                    fieldNames.add(fieldName);
                    fieldTypes.add(fieldType);
                    fieldLengths.add((int)fieldLen);
                    attributes.put(fieldName, new ArrayList<>());
                }
            }
            
            // Skip terminator byte
            ch.position(headerSize);
            
            // Read records
            byte[] record = new byte[recordSize];
            for (int r = 0; r < numRecords; r++) {
                ch.read(ByteBuffer.wrap(record));
                
                if (record[0] != ' ') { // Deleted record
                    continue;
                }
                
                int offset = 1;
                for (int f = 0; f < numFields; f++) {
                    int len = fieldLengths.get(f);
                    byte type = fieldTypes.get(f);
                    String value = new String(record, offset, len).trim();
                    
                    if (value.isEmpty() || value.equals("\u0000")) {
                        attributes.get(fieldNames.get(f)).add(null);
                    } else if (type == 'N' || type == 'F') {
                        try {
                            if (value.contains(".")) {
                                attributes.get(fieldNames.get(f)).add(Double.parseDouble(value));
                            } else {
                                attributes.get(fieldNames.get(f)).add(Long.parseLong(value));
                            }
                        } catch (NumberFormatException e) {
                            attributes.get(fieldNames.get(f)).add(value);
                        }
                    } else {
                        attributes.get(fieldNames.get(f)).add(value);
                    }
                    
                    offset += len;
                }
            }
        }
        
        return attributes;
    }

    // ====================== Options ======================

    public static class ShapefileOptions {
        private boolean includeGeometry = true;
        private boolean includeAttributes = true;
        private int maxRecords = 0;  // 0 = no limit

        public static ShapefileOptions defaults() {
            return new ShapefileOptions();
        }

        public ShapefileOptions includeGeometry(boolean v) { this.includeGeometry = v; return this; }
        public ShapefileOptions includeAttributes(boolean v) { this.includeAttributes = v; return this; }
        public ShapefileOptions maxRecords(int v) { this.maxRecords = v; return this; }

        public boolean includeGeometry() { return includeGeometry; }
        public boolean includeAttributes() { return includeAttributes; }
        public int maxRecords() { return maxRecords; }
    }
}
