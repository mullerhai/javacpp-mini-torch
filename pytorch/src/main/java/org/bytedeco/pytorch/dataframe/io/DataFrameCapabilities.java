package org.bytedeco.pytorch.dataframe.io;

/**
 * Enterprise DataFrame I/O Capability Matrix.
 * 
 * <p>This class documents all supported data formats and their capabilities.</p>
 * 
 * <pre>
 * Usage:
 *   System.out.println(DataFrameCapabilities.report());
 * </pre>
 */
public class DataFrameCapabilities {

    private DataFrameCapabilities() {}

    /**
     * Generate a comprehensive capability report for all supported formats.
     */
    public static String report() {
        StringBuilder sb = new StringBuilder();
        
        sb.append("╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n");
        sb.append("║                           ENTERPRISE DATAFRAME I/O CAPABILITY MATRIX                                           ║\n");
        sb.append("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
        sb.append("║ Format          │ Extensions                    │ Read  │ Write │ Schema │ Show  │ Magic │ Notes                          ║\n");
        sb.append("╠─────────────────┼──────────────────────────────┼───────┼───────┼────────┼───────┼────────┼────────────────────────────────╣\n");
        
        // Structured Data Formats
        sb.append("║ STRUCTURED DATA                                                                                                            ║\n");
        addRow(sb, "CSV", ".csv", "✓", "✓", "✓", "✓", "-", "RFC 4180, BOM, multiline, type inference");
        addRow(sb, "TSV", ".tsv", "✓", "✓", "✓", "✓", "-", "Tab delimiter, same features as CSV");
        addRow(sb, "JSON", ".json", "✓", "✓", "✓", "✓", "✓", "All pandas orients, nested flatten");
        addRow(sb, "JSONL", ".jsonl, .ndjson", "✓", "✓", "✓", "✓", "✓", "Streaming, newline-delimited");
        addRow(sb, "Parquet", ".parquet, .pq", "✓", "-", "✓", "✓", "✓", "Columnar, compressed, Hive partitions");
        addRow(sb, "Arrow", ".arrow, .ipc", "✓", "-", "✓", "✓", "✓", "Apache Arrow IPC format");
        addRow(sb, "Feather", ".feather", "✓", "-", "✓", "✓", "✓", "Arrow-compatible, fast read");
        addRow(sb, "ORC", ".orc", "✓", "-", "✓", "✓", "✓", "Hive format, columnar");
        addRow(sb, "Avro", ".avro", "✓", "-", "✓", "✓", "✓", "Row-based, schema evolution");
        addRow(sb, "Excel", ".xlsx, .xls, .xlsm", "✓", "✓", "✓", "✓", "PK", "Multi-sheet, styles");
        addRow(sb, "HDF5", ".h5, .hdf5, .hdf", "✓", "✓", "✓", "✓", "✓", "Columnar layout, compound types");
        
        // Binary Formats
        sb.append("║ BINARY FORMATS                                                                                                              ║\n");
        addRow(sb, "NumPy NPY", ".npy", "✓", "-", "✓", "✓", "✓", "NumPy array format, single array");
        addRow(sb, "NumPy NPZ", ".npz", "✓", "-", "✓", "✓", "PK", "Zip with multiple .npy files");
        addRow(sb, "SafeTensors", ".safetensors", "✓", "-", "✓", "✓", "-", "PyTorch tensors, memory-mapped");
        addRow(sb, "GGUF", ".gguf", "✓", "-", "✓", "✓", "✓", "LLM quantization format");
        addRow(sb, "Bin", ".bin", "✓", "✓", "✓", "✓", "-", "MicroLens, float32/int64, matrix");
        addRow(sb, "IMDB", ".pkl, .pickle, .imdb", "✓", "✓", "✓", "✓", "-", "Python pickle, dict/list structures");
        addRow(sb, "PT/PTH", ".pt, .pth", "✓", "-", "✓", "✓", "-", "PyTorch checkpoint format");
        
        // Database Formats
        sb.append("║ DATABASE / REMOTE                                                                                                           ║\n");
        addRow(sb, "SQLite", ".sqlite, .db", "✓", "✓", "-", "-", "-", "JDBC-backed, full SQL");
        addRow(sb, "MySQL", "JDBC", "✓", "✓", "-", "-", "-", "JDBC connector");
        addRow(sb, "DuckDB", ".duckdb", "✓", "✓", "-", "-", "-", "Analytical database");
        addRow(sb, "Lance", ".lance (dir)", "✓", "✓", "✓", "✓", "-", "ML dataset format, versioning");
        
        // Config Formats
        sb.append("║ CONFIG / OTHER                                                                                                               ║\n");
        addRow(sb, "TOML", ".toml", "✓", "✓", "✓", "✓", "-", "Config format, nested tables");
        
        // Vector Stores
        sb.append("║ VECTOR STORES                                                                                                                ║\n");
        addRow(sb, "Milvus", "URI", "✓", "-", "✓", "-", "-", "Vector database");
        addRow(sb, "MongoDB", "URI", "✓", "-", "✓", "-", "-", "Document store");
        addRow(sb, "OpenSearch", "URI", "✓", "-", "✓", "-", "-", "Vector search");
        addRow(sb, "PgVector", "JDBC", "✓", "-", "✓", "-", "-", "PostgreSQL extension");
        
        // Streaming
        sb.append("║ STREAMING                                                                                                                    ║\n");
        addRow(sb, "Kafka", "bootstrap servers", "✓", "-", "-", "-", "-", "Event streaming");
        addRow(sb, "Redis", "URI", "✓", "✓", "-", "-", "-", "In-memory, pub/sub");
        
        sb.append("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
        sb.append("║ LEGEND: ✓ = Supported, - = Not Available, PK = ZIP-based (magic bytes), JDBC = Requires connection string                 ║\n");
        sb.append("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
        sb.append("║ SYNTAX EXAMPLES                                                                                                              ║\n");
        sb.append("╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣\n");
        sb.append("║ // Read (auto-detect)                                                                                                        ║\n");
        sb.append("║ DataFrame df = DataFrame.read(\"/path/to/data.csv\");                                                                          ║\n");
        sb.append("║                                                                                                                                  ║\n");
        sb.append("║ // Read with options                                                                                                         ║\n");
        sb.append("║ DataFrame df = DataFrame.readCsv(path, CsvOptions.defaults().header(true));                                                   ║\n");
        sb.append("║ DataFrame df = DataFrame.readParquet(path, \"column1\", \"column2\");                                                             ║\n");
        sb.append("║ DataFrame df = DataFrame.read().csv(path).header(true).delimiter(',').build();                                                ║\n");
        sb.append("║                                                                                                                                  ║\n");
        sb.append("║ // Write                                                                                                                     ║\n");
        sb.append("║ df.writeCsv(\"/path/to/output.csv\");                                                                                          ║\n");
        sb.append("║ df.writeJson(\"/path/to/output.json\", JsonOptions.defaults().pretty(true));                                                  ║\n");
        sb.append("║                                                                                                                                  ║\n");
        sb.append("║ // Schema inference (without loading data)                                                                                    ║\n");
        sb.append("║ Schema schema = SchemaInfer.infer(\"/path/to/data.parquet\");                                                                  ║\n");
        sb.append("║ schema.print();                                                                                                              ║\n");
        sb.append("║                                                                                                                                  ║\n");
        sb.append("║ // PrintSchema shortcut                                                                                                      ║\n");
        sb.append("║ DataFrame.printSchema(\"/path/to/data.parquet\");                                                                              ║\n");
        sb.append("║                                                                                                                                  ║\n");
        sb.append("║ // Show data preview                                                                                                          ║\n");
        sb.append("║ DataShow.show(\"/path/to/data.csv\");                                                                                         ║\n");
        sb.append("║ DataShow.show(\"/path/to/data.parquet\", new DataShow.ShowOptions().maxRows(5));                                               ║\n");
        sb.append("║                                                                                                                                  ║\n");
        sb.append("║ // Format-specific readers                                                                                                     ║\n");
        sb.append("║ DataFrame df = CsvReader.read(path);                                                                                         ║\n");
        sb.append("║ DataFrame df = JsonReader.read(path);                                                                                         ║\n");
        sb.append("║ DataFrame df = ParquetReader.read(path);                                                                                     ║\n");
        sb.append("║ DataFrame df = BinReader.read(path);                                                                                         ║\n");
        sb.append("║ DataFrame df = ImdbReader.read(path);                                                                                         ║\n");
        sb.append("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n");
        
        return sb.toString();
    }

    private static void addRow(StringBuilder sb, String format, String extensions,
            String read, String write, String schema, String show, String magic, String notes) {
        sb.append("║ ");
        sb.append(padRight(format, 15));
        sb.append(" │ ");
        sb.append(padRight(extensions, 28));
        sb.append(" │ ");
        sb.append(padCenter(read, 5));
        sb.append(" │ ");
        sb.append(padCenter(write, 5));
        sb.append(" │ ");
        sb.append(padCenter(schema, 6));
        sb.append(" │ ");
        sb.append(padCenter(show, 5));
        sb.append(" │ ");
        sb.append(padCenter(magic, 6));
        sb.append(" │ ");
        sb.append(padRight(notes, 30));
        sb.append(" ║\n");
    }

    private static String padRight(String s, int len) {
        if (s == null) s = "";
        if (s.length() >= len) return s.substring(0, len);
        StringBuilder sb = new StringBuilder(s);
        while (sb.length() < len) sb.append(' ');
        return sb.toString();
    }

    private static String padCenter(String s, int len) {
        if (s == null) s = "";
        if (s.length() >= len) return s.substring(0, len);
        int pads = len - s.length();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < pads / 2; i++) sb.append(' ');
        sb.append(s);
        while (sb.length() < len) sb.append(' ');
        return sb.toString();
    }

    /**
     * Get summary statistics.
     */
    public static String summary() {
        return """
            Enterprise DataFrame I/O Summary:
            
            Supported Formats: 24+
            - Structured: CSV, TSV, JSON, JSONL, Parquet, Arrow, Feather, ORC, Avro, Excel, HDF5
            - Binary: NumPy (NPY/NPZ), SafeTensors, GGUF, Bin, IMDB, PyTorch
            - Database: SQLite, MySQL, DuckDB
            - Streaming: Kafka, Redis
            - Vector: Milvus, MongoDB, OpenSearch, PgVector
            - Other: Lance, TOML
            
            Key Features:
            ✓ Universal read via DataFrame.read(path)
            ✓ Format auto-detection (extension + magic bytes)
            ✓ Schema inference without loading data
            ✓ Pretty data preview via DataShow.show()
            ✓ Fluent API via DataFrame.read().format(path)
            ✓ Streaming support for large files
            ✓ Type coercion and validation
            """;
    }
}
