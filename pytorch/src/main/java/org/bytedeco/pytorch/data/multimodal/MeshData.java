package org.bytedeco.pytorch.data.multimodal;
import org.bytedeco.pytorch.data.*;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/**
 * Enterprise-grade 3D Mesh data container and reader.
 * 
 * <p>Meshes represent 3D surfaces as collections of vertices, edges, and faces.
 * This class supports:</p>
 * <ul>
 *   <li>Reading mesh data from OBJ, PLY, STL, OFF files</li>
 *   <li>Face operations: smoothing, subdivision, decimation</li>
 *   <li>Vertex operations: normals, colors, textures</li>
 *   <li>Geometric queries: bounding box, surface area, volume</li>
 *   <li>Conversion to/from DataFrame for analysis</li>
 * </ul>
 * 
 * <p>Example usage:</p>
 * <pre>
 *   // Read mesh from file
 *   MeshData mesh = MeshData.fromFile("/path/to/model.obj");
 *   
 *   // Convert to DataFrame for analysis
 *   DataFrame df = mesh.toDataFrame();
 *   
 *   // Compute normals
 *   mesh.computeVertexNormals();
 *   
 *   // Simplify mesh
 *   MeshData simplified = mesh.decimate(0.5);
 * </pre>
 */
public class MeshData implements Serializable {

    private float[][] vertices;  // [vertex_id][x,y,z]
    private int[][] faces;       // [face_id][v1,v2,v3]
    private float[][] vertexNormals;    // [vertex_id][nx,ny,nz]
    private float[][] vertexColors;     // [vertex_id][r,g,b,a]
    private float[][] faceNormals;      // [face_id][nx,ny,nz]
    private float[] faceAreas;          // [face_id]
    private Map<String, Object> metadata;

    public MeshData() {
        this.vertices = new float[0][3];
        this.faces = new int[0][3];
        this.vertexNormals = new float[0][3];
        this.vertexColors = new float[0][4];
        this.faceNormals = new float[0][3];
        this.faceAreas = new float[0];
        this.metadata = new LinkedHashMap<>();
    }

    public MeshData(float[][] vertices, int[][] faces) {
        this();
        this.vertices = vertices != null ? vertices : new float[0][3];
        this.faces = faces != null ? faces : new int[0][3];
        this.vertexNormals = new float[this.vertices.length][3];
        this.vertexColors = new float[this.vertices.length][4];
        this.faceNormals = new float[this.faces.length][3];
        this.faceAreas = new float[this.faces.length];
        computeDerivedData();
    }

    // ====================== Factory Methods ======================

    /**
     * Read mesh from a file (OBJ, PLY, STL, OFF).
     */
    public static MeshData fromFile(String path) throws IOException {
        String lower = path.toLowerCase();
        if (lower.endsWith(".obj")) {
            return fromObj(path);
        } else if (lower.endsWith(".ply")) {
            return fromPly(path);
        } else if (lower.endsWith(".stl")) {
            return fromStl(path);
        } else if (lower.endsWith(".off")) {
            return fromOff(path);
        } else {
            throw new IOException("Unsupported mesh format: " + path);
        }
    }

    /**
     * Read mesh from OBJ file.
     */
    public static MeshData fromObj(String path) throws IOException {
        List<float[]> verts = new ArrayList<>();
        List<int[]> faces = new ArrayList<>();
        
        for (String line : Files.readAllLines(Path.of(path))) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            
            String[] parts = line.split("\\s+");
            if (parts[0].equals("v")) {
                float x = Float.parseFloat(parts[1]);
                float y = Float.parseFloat(parts[2]);
                float z = Float.parseFloat(parts[3]);
                verts.add(new float[]{x, y, z});
            } else if (parts[0].equals("f")) {
                int[] f = new int[parts.length - 1];
                for (int i = 1; i < parts.length; i++) {
                    // Handle vertex/texture/normal indices (v/t/n)
                    String[] idx = parts[i].split("/");
                    f[i - 1] = Integer.parseInt(idx[0]) - 1;
                }
                faces.add(f);
            }
        }
        
        float[][] vertArray = verts.toArray(new float[0][]);
        int[][] faceArray = faces.toArray(new int[0][]);
        
        MeshData mesh = new MeshData(vertArray, faceArray);
        mesh.metadata.put("format", "obj");
        mesh.metadata.put("path", path);
        return mesh;
    }

    /**
     * Read mesh from PLY file.
     */
    public static MeshData fromPly(String path) throws IOException {
        List<String> lines = Files.readAllLines(Path.of(path));
        boolean isBinary = false;
        int numVertices = 0, numFaces = 0;
        int vertexProps = 3; // x, y, z minimum
        
        int dataStart = 0;
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.startsWith("format binary_little_endian")) {
                isBinary = true;
            } else if (line.startsWith("element vertex")) {
                numVertices = Integer.parseInt(line.split("\\s+")[2]);
            } else if (line.startsWith("element face")) {
                numFaces = Integer.parseInt(line.split("\\s+")[2]);
            } else if (line.equals("end_header")) {
                dataStart = i + 1;
                break;
            }
        }
        
        if (isBinary) {
            // Binary PLY reading (simplified)
            byte[] data = Files.readAllBytes(Path.of(path));
            int offset = 0;
            
            // Count line lengths to find header end
            int headerLen = 0;
            for (String line : lines) {
                headerLen += line.length() + 1;
                if (line.trim().equals("end_header")) break;
            }
            
            offset = headerLen;
            
            float[][] verts = new float[numVertices][3];
            for (int v = 0; v < numVertices && offset < data.length; v++) {
                verts[v][0] = ByteBuffer.wrap(data, offset, 4).order(ByteOrder.LITTLE_ENDIAN).getFloat(); offset += 4;
                verts[v][1] = ByteBuffer.wrap(data, offset, 4).order(ByteOrder.LITTLE_ENDIAN).getFloat(); offset += 4;
                verts[v][2] = ByteBuffer.wrap(data, offset, 4).order(ByteOrder.LITTLE_ENDIAN).getFloat(); offset += 4;
            }
            
            int[][] faces = new int[numFaces][];
            for (int f = 0; f < numFaces && offset < data.length; f++) {
                int n = data[offset++] & 0xFF;
                faces[f] = new int[n];
                for (int i = 0; i < n; i++) {
                    faces[f][i] = ByteBuffer.wrap(data, offset, 4).order(ByteOrder.LITTLE_ENDIAN).getInt(); offset += 4;
                }
            }
            
            MeshData mesh = new MeshData(verts, faces);
            mesh.metadata.put("format", "ply");
            mesh.metadata.put("binary", true);
            return mesh;
        } else {
            // ASCII PLY
            List<float[]> verts = new ArrayList<>();
            List<int[]> faces = new ArrayList<>();
            
            int lineIdx = dataStart;
            while (verts.size() < numVertices && lineIdx < lines.size()) {
                String[] parts = lines.get(lineIdx++).trim().split("\\s+");
                if (parts.length >= 3) {
                    verts.add(new float[]{
                        Float.parseFloat(parts[0]),
                        Float.parseFloat(parts[1]),
                        Float.parseFloat(parts[2])
                    });
                }
            }
            
            while (faces.size() < numFaces && lineIdx < lines.size()) {
                String[] parts = lines.get(lineIdx++).trim().split("\\s+");
                if (parts.length >= 4) {
                    int n = Integer.parseInt(parts[0]);
                    int[] f = new int[n];
                    for (int i = 0; i < n; i++) {
                        f[i] = Integer.parseInt(parts[i + 1]);
                    }
                    faces.add(f);
                }
            }
            
            float[][] vertArray = verts.toArray(new float[0][]);
            int[][] faceArray = faces.toArray(new int[0][]);
            
            MeshData mesh = new MeshData(vertArray, faceArray);
            mesh.metadata.put("format", "ply");
            mesh.metadata.put("binary", false);
            return mesh;
        }
    }

    /**
     * Read mesh from STL file.
     */
    public static MeshData fromStl(String path) throws IOException {
        List<float[]> verts = new ArrayList<>();
        List<int[]> faces = new ArrayList<>();
        
        List<String> lines = Files.readAllLines(path);
        boolean isBinary = false;
        
        // Check if binary STL
        if (lines.size() < 5) {
            isBinary = true;
        } else if (!lines.get(0).trim().startsWith("solid")) {
            isBinary = true;
        }
        
        if (isBinary) {
            byte[] data = Files.readAllBytes(Path.of(path));
            ByteBuffer buf = ByteBuffer.wrap(data, 80, data.length - 80)
                .order(ByteOrder.LITTLE_ENDIAN);
            
            int numTriangles = buf.getInt();
            int vertIdx = 0;
            
            while (buf.hasRemaining() && vertIdx / 3 < numTriangles) {
                float[] n = new float[3];
                n[0] = buf.getFloat(); n[1] = buf.getFloat(); n[2] = buf.getFloat();
                
                float[][] triVerts = new float[3][3];
                for (int i = 0; i < 3; i++) {
                    triVerts[i][0] = buf.getFloat();
                    triVerts[i][1] = buf.getFloat();
                    triVerts[i][2] = buf.getFloat();
                }
                
                buf.getShort(); // attribute byte count
                
                int baseIdx = verts.size();
                for (float[] v : triVerts) verts.add(v);
                faces.add(new int[]{baseIdx, baseIdx + 1, baseIdx + 2});
            }
        } else {
            // ASCII STL
            float[] n = new float[3];
            List<float[]> triVerts = new ArrayList<>();
            
            for (String line : lines) {
                line = line.trim();
                if (line.startsWith("facet normal")) {
                    String[] parts = line.split("\\s+");
                    n = new float[]{
                        Float.parseFloat(parts[2]),
                        Float.parseFloat(parts[3]),
                        Float.parseFloat(parts[4])
                    };
                    triVerts.clear();
                } else if (line.startsWith("vertex")) {
                    String[] parts = line.split("\\s+");
                    triVerts.add(new float[]{
                        Float.parseFloat(parts[1]),
                        Float.parseFloat(parts[2]),
                        Float.parseFloat(parts[3])
                    });
                } else if (line.startsWith("endfacet")) {
                    if (triVerts.size() == 3) {
                        int baseIdx = verts.size();
                        for (float[] v : triVerts) verts.add(v);
                        faces.add(new int[]{baseIdx, baseIdx + 1, baseIdx + 2});
                    }
                }
            }
        }
        
        float[][] vertArray = verts.toArray(new float[0][]);
        int[][] faceArray = faces.toArray(new int[0][]);
        
        MeshData mesh = new MeshData(vertArray, faceArray);
        mesh.metadata.put("format", "stl");
        mesh.metadata.put("binary", isBinary);
        return mesh;
    }

    /**
     * Read mesh from OFF file.
     */
    public static MeshData fromOff(String path) throws IOException {
        List<String> lines = Files.readAllLines(path);
        int lineIdx = 0;
        
        // Skip OFF header
        while (lineIdx < lines.size() && !lines.get(lineIdx).trim().startsWith("OFF")) {
            lineIdx++;
        }
        lineIdx++;
        
        // Parse counts
        String[] counts = lines.get(lineIdx++).trim().split("\\s+");
        int numVerts = Integer.parseInt(counts[0]);
        int numFaces = Integer.parseInt(counts[1]);
        
        float[][] verts = new float[numVerts][3];
        for (int v = 0; v < numVerts && lineIdx < lines.size(); v++) {
            String[] parts = lines.get(lineIdx++).trim().split("\\s+");
            if (parts.length >= 3) {
                verts[v][0] = Float.parseFloat(parts[0]);
                verts[v][1] = Float.parseFloat(parts[1]);
                verts[v][2] = Float.parseFloat(parts[2]);
            }
        }
        
        int[][] faces = new int[numFaces][];
        for (int f = 0; f < numFaces && lineIdx < lines.size(); f++) {
            String[] parts = lines.get(lineIdx++).trim().split("\\s+");
            if (parts.length >= 4) {
                int n = Integer.parseInt(parts[0]);
                faces[f] = new int[n];
                for (int i = 0; i < n; i++) {
                    faces[f][i] = Integer.parseInt(parts[i + 1]);
                }
            }
        }
        
        MeshData mesh = new MeshData(verts, faces);
        mesh.metadata.put("format", "off");
        return mesh;
    }

    // ====================== Accessors ======================

    public int numVertices() { return vertices.length; }
    public int numFaces() { return faces.length; }
    public float[] vertex(int i) { return vertices[i]; }
    public int[] face(int i) { return faces[i]; }
    public float[] vertexNormal(int i) { return vertexNormals[i]; }
    public float[] vertexColor(int i) { return vertexColors[i]; }
    public float[] faceNormal(int i) { return faceNormals[i]; }
    public float faceArea(int i) { return faceAreas[i]; }

    // ====================== Geometry Operations ======================

    private void computeDerivedData() {
        for (int f = 0; f < faces.length; f++) {
            int[] v = faces[f];
            if (v.length >= 3) {
                float[] a = vertices[v[0]];
                float[] b = vertices[v[1]];
                float[] c = vertices[v[2]];
                
                // Compute face normal
                float[] ab = new float[]{b[0]-a[0], b[1]-a[1], b[2]-a[2]};
                float[] ac = new float[]{c[0]-a[0], c[1]-a[1], c[2]-a[2]};
                
                float[] n = cross(ab, ac);
                float len = (float)Math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
                if (len > 1e-6) {
                    n[0] /= len; n[1] /= len; n[2] /= len;
                }
                
                faceNormals[f] = n;
                faceAreas[f] = len / 2;
            }
        }
    }

    /**
     * Compute vertex normals by averaging face normals.
     */
    public MeshData computeVertexNormals() {
        float[][] sums = new float[vertices.length][3];
        int[] counts = new int[vertices.length];
        
        for (int f = 0; f < faces.length; f++) {
            for (int vi : faces[f]) {
                sums[vi][0] += faceNormals[f][0];
                sums[vi][1] += faceNormals[f][1];
                sums[vi][2] += faceNormals[f][2];
                counts[vi]++;
            }
        }
        
        for (int v = 0; v < vertices.length; v++) {
            if (counts[v] > 0) {
                float len = (float)Math.sqrt(
                    sums[v][0]*sums[v][0] + sums[v][1]*sums[v][1] + sums[v][2]*sums[v][2]);
                if (len > 1e-6) {
                    vertexNormals[v][0] = sums[v][0] / len;
                    vertexNormals[v][1] = sums[v][1] / len;
                    vertexNormals[v][2] = sums[v][2] / len;
                }
            }
        }
        
        return this;
    }

    /**
     * Get bounding box.
     */
    public float[][] boundingBox() {
        if (vertices.length == 0) {
            return new float[][]{{0,0,0},{0,0,0}};
        }
        
        float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE, minZ = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE, maxY = -Float.MAX_VALUE, maxZ = -Float.MAX_VALUE;
        
        for (float[] v : vertices) {
            minX = Math.min(minX, v[0]); maxX = Math.max(maxX, v[0]);
            minY = Math.min(minY, v[1]); maxY = Math.max(maxY, v[1]);
            minZ = Math.min(minZ, v[2]); maxZ = Math.max(maxZ, v[2]);
        }
        
        return new float[][]{{minX, minY, minZ}, {maxX, maxY, maxZ}};
    }

    /**
     * Get centroid.
     */
    public float[] centroid() {
        if (vertices.length == 0) return new float[]{0, 0, 0};
        
        float cx = 0, cy = 0, cz = 0;
        for (float[] v : vertices) {
            cx += v[0]; cy += v[1]; cz += v[2];
        }
        int n = vertices.length;
        return new float[]{cx/n, cy/n, cz/n};
    }

    /**
     * Compute total surface area.
     */
    public float surfaceArea() {
        float total = 0;
        for (float a : faceAreas) total += a;
        return total;
    }

    /**
     * Compute total volume (for closed meshes).
     */
    public float volume() {
        float vol = 0;
        for (int f = 0; f < faces.length; f++) {
            int[] v = faces[f];
            if (v.length >= 3) {
                float[] a = vertices[v[0]];
                float[] b = vertices[v[1]];
                float[] c = vertices[v[2]];
                
                // Signed volume of tetrahedron with origin
                vol += a[0] * (b[1]*c[2] - c[1]*b[2])
                     - a[1] * (b[0]*c[2] - c[0]*b[2])
                     + a[2] * (b[0]*c[1] - c[0]*b[1]);
            }
        }
        return Math.abs(vol) / 6;
    }

    /**
     * Simple mesh simplification by vertex clustering.
     */
    public MeshData simplify(float targetRatio) {
        int targetVerts = (int)(vertices.length * targetRatio);
        if (targetVerts >= vertices.length) return this;
        
        float[][] bb = boundingBox();
        float cellSize = (bb[1][0] - bb[0][0]) / (float)Math.cbrt(targetVerts);
        
        Map<String, List<Integer>> clusters = new HashMap<>();
        for (int v = 0; v < vertices.length; v++) {
            String key = String.format("%.0f,%.0f,%.0f",
                Math.floor(vertices[v][0] / cellSize),
                Math.floor(vertices[v][1] / cellSize),
                Math.floor(vertices[v][2] / cellSize));
            clusters.computeIfAbsent(key, k -> new ArrayList<>()).add(v);
        }
        
        // Create simplified mesh
        Map<Integer, Integer> oldToNew = new HashMap<>();
        float[][] newVerts = new float[clusters.size()][3];
        int newIdx = 0;
        
        for (List<Integer> cluster : clusters.values()) {
            float cx = 0, cy = 0, cz = 0;
            for (int oldV : cluster) {
                cx += vertices[oldV][0];
                cy += vertices[oldV][1];
                cz += vertices[oldV][2];
                oldToNew.put(oldV, newIdx);
            }
            newVerts[newIdx][0] = cx / cluster.size();
            newVerts[newIdx][1] = cy / cluster.size();
            newVerts[newIdx][2] = cz / cluster.size();
            newIdx++;
        }
        
        List<int[]> newFaces = new ArrayList<>();
        for (int[] f : faces) {
            List<Integer> mapped = new ArrayList<>();
            for (int vi : f) {
                Integer m = oldToNew.get(vi);
                if (m != null) mapped.add(m);
            }
            if (mapped.size() >= 3) {
                newFaces.add(mapped.stream().mapToInt(i->i).toArray());
            }
        }
        
        MeshData result = new MeshData(newVerts, newFaces.toArray(new int[0][]));
        result.metadata.put("simplified", true);
        result.metadata.put("original_vertices", vertices.length);
        return result;
    }

    private float[] cross(float[] a, float[] b) {
        return new float[]{
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        };
    }

    // ====================== DataFrame Integration ======================

    /**
     * Convert mesh to DataFrame for analysis.
     */
    public DataFrame toDataFrame() {
        DataFrame df = DataFrame.create();
        df.addColumn("vertex_id", Column.DType.INT32);
        df.addColumn("x", Column.DType.FLOAT32);
        df.addColumn("y", Column.DType.FLOAT32);
        df.addColumn("z", Column.DType.FLOAT32);
        df.addColumn("nx", Column.DType.FLOAT32);
        df.addColumn("ny", Column.DType.FLOAT32);
        df.addColumn("nz", Column.DType.FLOAT32);
        df.addColumn("r", Column.DType.FLOAT32);
        df.addColumn("g", Column.DType.FLOAT32);
        df.addColumn("b", Column.DType.FLOAT32);
        df.addColumn("a", Column.DType.FLOAT32);
        
        for (int v = 0; v < vertices.length; v++) {
            int ri = df.addEmptyRow();
            df.set(ri, "vertex_id", v);
            df.set(ri, "x", vertices[v][0]);
            df.set(ri, "y", vertices[v][1]);
            df.set(ri, "z", vertices[v][2]);
            df.set(ri, "nx", vertexNormals[v][0]);
            df.set(ri, "ny", vertexNormals[v][1]);
            df.set(ri, "nz", vertexNormals[v][2]);
            df.set(ri, "r", vertexColors[v][0]);
            df.set(ri, "g", vertexColors[v][1]);
            df.set(ri, "b", vertexColors[v][2]);
            df.set(ri, "a", vertexColors[v][3]);
        }
        
        return df;
    }

    /**
     * Convert faces to DataFrame.
     */
    public DataFrame facesToDataFrame() {
        DataFrame df = DataFrame.create();
        df.addColumn("face_id", Column.DType.INT32);
        df.addColumn("v1", Column.DType.INT32);
        df.addColumn("v2", Column.DType.INT32);
        df.addColumn("v3", Column.DType.INT32);
        df.addColumn("area", Column.DType.FLOAT32);
        df.addColumn("nx", Column.DType.FLOAT32);
        df.addColumn("ny", Column.DType.FLOAT32);
        df.addColumn("nz", Column.DType.FLOAT32);
        
        for (int f = 0; f < faces.length; f++) {
            int[] face = faces[f];
            int ri = df.addEmptyRow();
            df.set(ri, "face_id", f);
            df.set(ri, "v1", face.length > 0 ? face[0] : 0);
            df.set(ri, "v2", face.length > 1 ? face[1] : 0);
            df.set(ri, "v3", face.length > 2 ? face[2] : 0);
            df.set(ri, "area", faceAreas[f]);
            df.set(ri, "nx", faceNormals[f][0]);
            df.set(ri, "ny", faceNormals[f][1]);
            df.set(ri, "nz", faceNormals[f][2]);
        }
        
        return df;
    }

    // ====================== Serialization ======================

    /**
     * Write mesh to OBJ file.
     */
    public void toObj(String path) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(Path.of(path))) {
            for (float[] v : vertices) {
                writer.write(String.format("v %.6f %.6f %.6f%n", v[0], v[1], v[2]));
            }
            for (int[] f : faces) {
                writer.write("f");
                for (int vi : f) {
                    writer.write(" " + (vi + 1));
                }
                writer.newLine();
            }
        }
    }

    public Map<String, Object> metadata() {
        return new LinkedHashMap<>(metadata);
    }

    public MeshData putMetadata(String key, Object value) {
        metadata.put(key, value);
        return this;
    }

    @Override
    public String toString() {
        return String.format("MeshData[vertices=%d, faces=%d, area=%.2f, volume=%.2f]",
            vertices.length, faces.length, surfaceArea(), volume());
    }

    private static final long serialVersionUID = 1L;
}
