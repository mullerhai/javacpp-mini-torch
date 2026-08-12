/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
 *
 * Licensed under the Apache License, Version 2.0, or (at your option)
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
 * Unity Catalog implementation (Databricks).
 */
public final class UnityCatalog implements DataCatalog {

    private final String uri;
    private final String token;
    private final String warehouseId;
    private final Map<String, List<String>> namespaces;
    private final Map<String, TableInfo> tables;

    private UnityCatalog(Builder builder) {
        this.uri = builder.uri;
        this.token = builder.token;
        this.warehouseId = builder.warehouseId;
        this.namespaces = new HashMap<>();
        this.tables = new HashMap<>();
    }

    public static Builder builder() { return new Builder(); }

    @Override
    public List<String> listNamespaces() {
        List<String> result = new ArrayList<>();
        for (String ns : namespaces.keySet()) {
            if (!ns.contains(".")) result.add(ns);
        }
        return result;
    }

    @Override
    public List<String> listNamespaces(String parent) {
        String prefix = parent + ".";
        List<String> result = new ArrayList<>();
        for (String ns : namespaces.keySet()) {
            if (ns.startsWith(prefix)) {
                String child = ns.substring(prefix.length()).split("\\.")[0];
                if (!result.contains(child)) result.add(child);
            }
        }
        return result;
    }

    @Override
    public void createNamespace(String namespace, Map<String, String> properties) {
        namespaces.put(namespace, new ArrayList<>());
    }

    @Override
    public void dropNamespace(String namespace) {
        namespaces.remove(namespace);
        tables.entrySet().removeIf(e -> e.getKey().startsWith(namespace + "."));
    }

    @Override
    public Map<String, String> getNamespaceProperties(String namespace) {
        return new HashMap<>();
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
    public TableInfo getTable(String fullName) { return tables.get(fullName); }

    @Override
    public boolean tableExists(String fullName) { return tables.containsKey(fullName); }

    @Override
    public DaftDataFrame readTable(String fullName) {
        TableInfo info = tables.get(fullName);
        if (info == null) throw new IllegalArgumentException("Table not found: " + fullName);
        return DaftDataFrame.fromParquet(info.location());
    }

    @Override
    public void writeTable(String fullName, DaftDataFrame df, WriteMode mode) {
        try { df.writeParquet(fullName); } catch (Exception e) { throw new RuntimeException(e); }
    }

    @Override
    public void dropTable(String fullName) { tables.remove(fullName); }

    @Override
    public void renameTable(String oldName, String newName) {
        TableInfo info = tables.remove(oldName);
        if (info != null) tables.put(newName, info);
    }

    @Override
    public List<PartitionInfo> listPartitions(String fullName) { return new ArrayList<>(); }

    @Override
    public void addPartitions(String fullName, List<PartitionInfo> partitions) {}

    @Override
    public void dropPartitions(String fullName, List<String> partitionValues) {}

    @Override
    public TableInfo refreshTable(String fullName) { return tables.get(fullName); }

    @Override
    public void updateTableProperties(String fullName, Map<String, String> updates) {}

    public String uri() { return uri; }
    public String warehouseId() { return warehouseId; }

    public static final class Builder {
        private String uri = "https://example.cloud.databricks.com";
        private String token;
        private String warehouseId;

        public Builder uri(String u) { this.uri = u; return this; }
        public Builder token(String t) { this.token = t; return this; }
        public Builder warehouseId(String id) { this.warehouseId = id; return this; }
        public UnityCatalog build() { return new UnityCatalog(this); }
    }
}
