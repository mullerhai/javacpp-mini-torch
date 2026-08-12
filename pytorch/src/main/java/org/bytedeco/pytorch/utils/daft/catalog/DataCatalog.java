/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
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
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.daft.catalog;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.daft.DaftDataFrame;

import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;

/**
 * Data Catalog abstraction for Daft.
 *
 * <p>Unified interface for:
 * <ul>
 *   <li>Iceberg tables (REST catalog)</li>
 *   <li>Unity Catalog (Databricks)</li>
 *   <li>AWS Glue Catalog</li>
 *   <li>Hive Metastore</li>
 *   <li>Delta Lake</li>
 * </ul>
 *
 * <pre>{@code
 * // Create catalog
 * DataCatalog catalog = IcebergCatalog.builder()
 *     .uri("http://localhost:8181")
 *     .credential("token")
 *     .build();
 *
 * // List namespaces
 * List<String> ns = catalog.listNamespaces();
 *
 * // Read table
 * DaftDataFrame df = catalog.readTable("production.users");
 * df.show();
 *
 * // Write table
 * catalog.writeTable("production.users", df, WriteMode.OVERWRITE);
 * }</pre>
 */
public interface DataCatalog {

    // ---- Namespace operations ----

    /** List all top-level namespaces. */
    List<String> listNamespaces();

    /** List namespaces under a parent. */
    List<String> listNamespaces(String parent);

    /** Create a namespace. */
    void createNamespace(String namespace, Map<String, String> properties);

    /** Drop a namespace. */
    void dropNamespace(String namespace);

    /** Get namespace properties. */
    Map<String, String> getNamespaceProperties(String namespace);

    // ---- Table operations ----

    /** List all tables in a namespace. */
    List<TableInfo> listTables(String namespace);

    /** Get table metadata. */
    TableInfo getTable(String fullName);

    /** Check if table exists. */
    boolean tableExists(String fullName);

    /** Read table as DaftDataFrame. */
    DaftDataFrame readTable(String fullName);

    /** Write DaftDataFrame to table. */
    void writeTable(String fullName, DaftDataFrame df, WriteMode mode);

    /** Drop a table. */
    void dropTable(String fullName);

    /** Rename a table. */
    void renameTable(String oldName, String newName);

    // ---- Partition operations ----

    /** Get partition metadata. */
    List<PartitionInfo> listPartitions(String fullName);

    /** Add partitions. */
    void addPartitions(String fullName, List<PartitionInfo> partitions);

    /** Drop partitions. */
    void dropPartitions(String fullName, List<String> partitionValues);

    // ---- Table metadata ----

    /** Refresh table metadata from catalog. */
    TableInfo refreshTable(String fullName);

    /** Update table properties. */
    void updateTableProperties(String fullName, Map<String, String> updates);

    /**
     * Table information.
     */
    final class TableInfo {
        private final String name;
        private final String namespace;
        private final String location;
        private final String format;
        private final Map<String, String> properties;
        private final Schema schema;
        private final long createdAt;
        private final long lastModified;

        private TableInfo(Builder builder) {
            this.name = builder.name;
            this.namespace = builder.namespace;
            this.location = builder.location;
            this.format = builder.format;
            this.properties = builder.properties;
            this.schema = builder.schema;
            this.createdAt = builder.createdAt;
            this.lastModified = builder.lastModified;
        }

        public String name() { return name; }
        public String namespace() { return namespace; }
        public String location() { return location; }
        public String format() { return format; }
        public Map<String, String> properties() { return properties; }
        public Schema schema() { return schema; }
        public long createdAt() { return createdAt; }
        public long lastModified() { return lastModified; }

        public String fullName() {
            return namespace != null && !namespace.isEmpty()
                ? namespace + "." + name : name;
        }

        public static Builder builder() { return new Builder(); }

        public static final class Builder {
            private String name;
            private String namespace;
            private String location;
            private String format = "parquet";
            private Map<String, String> properties = new java.util.LinkedHashMap<>();
            private Schema schema;
            private long createdAt;
            private long lastModified;

            public Builder name(String n) { this.name = n; return this; }
            public Builder namespace(String ns) { this.namespace = ns; return this; }
            public Builder location(String loc) { this.location = loc; return this; }
            public Builder format(String fmt) { this.format = fmt; return this; }
            public Builder properties(Map<String, String> p) { this.properties = p; return this; }
            public Builder schema(Schema s) { this.schema = s; return this; }
            public Builder createdAt(long t) { this.createdAt = t; return this; }
            public Builder lastModified(long t) { this.lastModified = t; return this; }

            public TableInfo build() { return new TableInfo(this); }
        }
    }

    /**
     * Schema definition.
     */
    final class Schema {
        private final List<Field> fields;

        public Schema(List<Field> fields) { this.fields = fields; }
        public List<Field> fields() { return fields; }
        public Field get(String name) {
            for (Field f : fields) {
                if (f.name().equals(name)) return f;
            }
            return null;
        }

        public static Schema of(List<Field> fields) { return new Schema(fields); }
    }

    /**
     * Schema field.
     */
    final class Field {
        private final String name;
        private final String type;
        private final String comment;
        private final boolean nullable;

        public Field(String name, String type) {
            this(name, type, null, true);
        }

        public Field(String name, String type, String comment, boolean nullable) {
            this.name = name;
            this.type = type;
            this.comment = comment;
            this.nullable = nullable;
        }

        public String name() { return name; }
        public String type() { return type; }
        public String comment() { return comment; }
        public boolean nullable() { return nullable; }
    }

    /**
     * Partition information.
     */
    final class PartitionInfo {
        private final Map<String, String> values;
        private final String location;
        private final long recordCount;
        private final long sizeBytes;

        public PartitionInfo(Map<String, String> values, String location,
                           long recordCount, long sizeBytes) {
            this.values = values;
            this.location = location;
            this.recordCount = recordCount;
            this.sizeBytes = sizeBytes;
        }

        public Map<String, String> values() { return values; }
        public String location() { return location; }
        public long recordCount() { return recordCount; }
        public long sizeBytes() { return sizeBytes; }

        public String valuesString() {
            StringBuilder sb = new StringBuilder();
            boolean first = true;
            for (Map.Entry<String, String> e : values.entrySet()) {
                if (!first) sb.append("/");
                sb.append(e.getKey()).append("=").append(e.getValue());
                first = false;
            }
            return sb.toString();
        }
    }

    /**
     * Write mode for tables.
     */
    enum WriteMode {
        APPEND,
        OVERWRITE,
        TRUNCATE,
        ERROR_IF_EXISTS,
        IGNORE
    }

    // ---- Factory ----

    /**
     * Create catalog from URI.
     */
    static DataCatalog fromUri(String uri) {
        if (uri == null) throw new IllegalArgumentException("uri cannot be null");

        if (uri.startsWith("iceberg://") || uri.startsWith("http://") || uri.startsWith("https://")) {
            return IcebergCatalog.builder().uri(uri).build();
        }
        if (uri.startsWith("unity://")) {
            return UnityCatalog.builder().uri(uri).build();
        }
        if (uri.startsWith("glue://") || uri.startsWith("s3://")) {
            return GlueCatalog.builder().region(uri).build();
        }
        if (uri.startsWith("delta://")) {
            return DeltaCatalog.builder().rootPath(uri.substring("delta://".length())).build();
        }
        if (uri.startsWith("hive://")) {
            return HiveCatalog.builder().uri(uri).build();
        }

        throw new IllegalArgumentException("Unknown catalog URI scheme: " + uri);
    }

    /**
     * Discover implementations via ServiceLoader.
     */
    static List<DataCatalog> discover() {
        List<DataCatalog> catalogs = new java.util.ArrayList<>();
        for (DataCatalog catalog : ServiceLoader.load(DataCatalog.class)) {
            catalogs.add(catalog);
        }
        return catalogs;
    }
}
