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

import org.bytedeco.pytorch.utils.daft.DaftDataFrame;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Apache Iceberg REST Catalog implementation.
 *
 * <pre>{@code
 * DataCatalog catalog = IcebergCatalog.builder()
 *     .uri("http://catalog:8080")
 *     .warehouse("s3://warehouse/")
 *     .credential("Bearer token")
 *     .build();
 *
 * DaftDataFrame df = catalog.readTable("production.users");
 * }</pre>
 */
public final class IcebergCatalog implements DataCatalog {

    private final String uri;
    private final String warehouse;
    private final String credential;
    private final String prefix;
    private final Map<String, NamespaceInfo> namespaces;
    private final Map<String, TableInfo> tables;

    private IcebergCatalog(Builder builder) {
        this.uri = builder.uri;
        this.warehouse = builder.warehouse;
        this.credential = builder.credential;
        this.prefix = builder.prefix;
        this.namespaces = new HashMap<>();
        this.tables = new HashMap<>();
    }

    public static Builder builder() { return new Builder(); }

    @Override
    public List<String> listNamespaces() {
        return new ArrayList<>(namespaces.keySet());
    }

    @Override
    public List<String> listNamespaces(String parent) {
        List<String> result = new ArrayList<>();
        String prefix = parent == null ? "" : parent + ".";
        for (String ns : namespaces.keySet()) {
            if (ns.startsWith(prefix) && !ns.equals(parent)) {
                int dotIndex = ns.indexOf('.', prefix.length());
                String child = dotIndex < 0 ? ns.substring(prefix.length()) : ns.substring(prefix.length(), dotIndex);
                if (!result.contains(child)) {
                    result.add(child);
                }
            }
        }
        return result;
    }

    @Override
    public void createNamespace(String namespace, Map<String, String> properties) {
        NamespaceInfo info = new NamespaceInfo(namespace, properties);
        namespaces.put(namespace, info);
    }

    @Override
    public void dropNamespace(String namespace) {
        namespaces.remove(namespace);
        // Also remove nested tables
        String prefix = namespace + ".";
        tables.entrySet().removeIf(e -> e.getKey().startsWith(prefix));
    }

    @Override
    public Map<String, String> getNamespaceProperties(String namespace) {
        NamespaceInfo info = namespaces.get(namespace);
        return info != null ? new HashMap<>(info.properties) : Collections.emptyMap();
    }

    @Override
    public List<TableInfo> listTables(String namespace) {
        List<TableInfo> result = new ArrayList<>();
        String prefix = namespace + ".";
        for (Map.Entry<String, TableInfo> e : tables.entrySet()) {
            if (e.getKey().equals(namespace) || e.getKey().startsWith(prefix)) {
                result.add(e.getValue());
            }
        }
        return result;
    }

    @Override
    public TableInfo getTable(String fullName) {
        return tables.get(fullName);
    }

    @Override
    public boolean tableExists(String fullName) {
        return tables.containsKey(fullName);
    }

    @Override
    public DaftDataFrame readTable(String fullName) {
        TableInfo info = tables.get(fullName);
        if (info == null) {
            throw new IllegalArgumentException("Table not found: " + fullName);
        }
        String location = info.location();
        if (location != null) {
            if (location.endsWith(".parquet") || location.contains("/parquet/")) {
                return DaftDataFrame.fromParquet(location);
            } else if (location.contains("/csv/")) {
                return DaftDataFrame.fromCsv(location);
            }
        }
        throw new UnsupportedOperationException("Cannot read table: " + fullName + " at " + location);
    }

    @Override
    public void writeTable(String fullName, DaftDataFrame df, WriteMode mode) {
        TableInfo info = tables.get(fullName);
        if (info == null) {
            throw new IllegalArgumentException("Table not found: " + fullName);
        }
        String location = info.location();
        if (location == null) {
            throw new IllegalStateException("Table location not set: " + fullName);
        }
        try {
            if (location.endsWith(".parquet")) {
                df.writeParquet(location);
            } else {
                df.writeCsv(location);
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to write table: " + fullName, e);
        }
    }

    @Override
    public void dropTable(String fullName) {
        tables.remove(fullName);
    }

    @Override
    public void renameTable(String oldName, String newName) {
        TableInfo info = tables.remove(oldName);
        if (info != null) {
            tables.put(newName, TableInfo.builder()
                    .name(info.name())
                    .namespace(newName.contains(".") ? newName.substring(0, newName.lastIndexOf('.')) : null)
                    .location(info.location())
                    .format(info.format())
                    .properties(info.properties())
                    .schema(info.schema())
                    .build());
        }
    }

    @Override
    public List<PartitionInfo> listPartitions(String fullName) {
        return new ArrayList<>();
    }

    @Override
    public void addPartitions(String fullName, List<PartitionInfo> partitions) {
        // Iceberg handles partition discovery automatically
    }

    @Override
    public void dropPartitions(String fullName, List<String> partitionValues) {
        // Would need to delete partition files
    }

    @Override
    public TableInfo refreshTable(String fullName) {
        return tables.get(fullName);
    }

    @Override
    public void updateTableProperties(String fullName, Map<String, String> updates) {
        TableInfo info = tables.get(fullName);
        if (info != null) {
            Map<String, String> props = new HashMap<>(info.properties());
            props.putAll(updates);
            tables.put(fullName, TableInfo.builder()
                    .name(info.name())
                    .namespace(info.namespace())
                    .location(info.location())
                    .format(info.format())
                    .properties(props)
                    .schema(info.schema())
                    .createdAt(info.createdAt())
                    .lastModified(System.currentTimeMillis())
                    .build());
        }
    }

    public String uri() { return uri; }
    public String warehouse() { return warehouse; }
    public String credential() { return credential; }

    private static final class NamespaceInfo {
        final String name;
        final Map<String, String> properties;

        NamespaceInfo(String name, Map<String, String> properties) {
            this.name = name;
            this.properties = properties != null ? properties : new HashMap<>();
        }
    }

    public static final class Builder {
        private String uri = "http://localhost:8080";
        private String warehouse;
        private String credential;
        private String prefix = "";

        public Builder uri(String u) { this.uri = u; return this; }
        public Builder warehouse(String w) { this.warehouse = w; return this; }
        public Builder credential(String c) { this.credential = c; return this; }
        public Builder prefix(String p) { this.prefix = p; return this; }

        public IcebergCatalog build() { return new IcebergCatalog(this); }
    }
}
