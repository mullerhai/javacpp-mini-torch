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

import java.util.*;

/**
 * Delta Lake Catalog implementation.
 */
public final class DeltaCatalog implements DataCatalog {

    private final String rootPath;
    private final Map<String, DeltaTable> tables;

    private DeltaCatalog(String rootPath) {
        this.rootPath = rootPath;
        this.tables = new HashMap<>();
    }

    public static Builder builder() { return new Builder(); }

    @Override
    public List<String> listNamespaces() {
        Set<String> ns = new HashSet<>();
        for (String table : tables.keySet()) {
            int dot = table.lastIndexOf('.');
            if (dot > 0) ns.add(table.substring(0, dot));
        }
        return new ArrayList<>(ns);
    }

    @Override
    public List<String> listNamespaces(String parent) {
        return new ArrayList<>();
    }

    @Override
    public void createNamespace(String namespace, Map<String, String> properties) {}

    @Override
    public void dropNamespace(String namespace) {}

    @Override
    public Map<String, String> getNamespaceProperties(String namespace) {
        return new HashMap<>();
    }

    @Override
    public List<TableInfo> listTables(String namespace) {
        List<TableInfo> result = new ArrayList<>();
        String prefix = namespace + ".";
        for (Map.Entry<String, DeltaTable> e : tables.entrySet()) {
            if (e.getKey().startsWith(prefix)) {
                result.add(e.getValue().toTableInfo(e.getKey()));
            }
        }
        return result;
    }

    @Override
    public TableInfo getTable(String fullName) {
        DeltaTable table = tables.get(fullName);
        return table != null ? table.toTableInfo(fullName) : null;
    }

    @Override
    public boolean tableExists(String fullName) { return tables.containsKey(fullName); }

    @Override
    public DaftDataFrame readTable(String fullName) {
        DeltaTable table = tables.get(fullName);
        if (table == null) throw new IllegalArgumentException("Table not found: " + fullName);
        return DaftDataFrame.fromParquet(table.path);
    }

    @Override
    public void writeTable(String fullName, DaftDataFrame df, WriteMode mode) {
        DeltaTable table = tables.get(fullName);
        if (table == null) {
            table = new DeltaTable(fullName, rootPath + "/" + fullName.replace('.', '/'));
            tables.put(fullName, table);
        }
        try {
            if (mode == WriteMode.APPEND) {
                DaftDataFrame existing = readTable(fullName);
                df = existing.concat(df);
            } else if (mode == WriteMode.OVERWRITE || mode == WriteMode.TRUNCATE) {
                // continue
            } else if (mode == WriteMode.ERROR_IF_EXISTS) {
                if (tableExists(fullName)) {
                    throw new IllegalStateException("Table already exists: " + fullName);
                }
            } else if (mode == WriteMode.IGNORE) {
                if (tableExists(fullName)) return;
            }
            df.writeParquet(table.path);
            table.version++;
        } catch (Exception e) {
            throw new RuntimeException("Failed to write table: " + fullName, e);
        }
    }

    @Override
    public void dropTable(String fullName) { tables.remove(fullName); }

    @Override
    public void renameTable(String oldName, String newName) {
        DeltaTable table = tables.remove(oldName);
        if (table != null) tables.put(newName, table);
    }

    @Override
    public List<PartitionInfo> listPartitions(String fullName) { return new ArrayList<>(); }

    @Override
    public void addPartitions(String fullName, List<PartitionInfo> partitions) {}

    @Override
    public void dropPartitions(String fullName, List<String> partitionValues) {}

    @Override
    public TableInfo refreshTable(String fullName) { return getTable(fullName); }

    @Override
    public void updateTableProperties(String fullName, Map<String, String> updates) {
        DeltaTable table = tables.get(fullName);
        if (table != null) table.properties.putAll(updates);
    }

    public String rootPath() { return rootPath; }

    private static final class DeltaTable {
        final String name;
        final String path;
        final Map<String, String> properties;
        long version;
        long createdAt;

        DeltaTable(String name, String path) {
            this.name = name;
            this.path = path;
            this.properties = new HashMap<>();
            this.version = 0;
            this.createdAt = System.currentTimeMillis();
        }

        TableInfo toTableInfo(String fullName) {
            String ns = null;
            if (fullName.contains(".")) {
                ns = fullName.substring(0, fullName.lastIndexOf('.'));
            }
            return TableInfo.builder()
                    .name(name)
                    .namespace(ns)
                    .location(path)
                    .format("delta")
                    .properties(properties)
                    .createdAt(createdAt)
                    .lastModified(createdAt + version * 1000)
                    .build();
        }
    }

    public static final class Builder {
        private String rootPath = "/tmp/delta";

        public Builder rootPath(String path) { this.rootPath = path; return this; }
        public DeltaCatalog build() { return new DeltaCatalog(rootPath); }
    }
}
