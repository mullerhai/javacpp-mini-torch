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
 * AWS Glue Catalog implementation.
 */
public final class GlueCatalog implements DataCatalog {

    private final String region;
    private final String accessKeyId;
    private final String secretAccessKey;
    private final String endpoint;
    private final Map<String, DatabaseInfo> databases;
    private final Map<String, TableInfo> tables;

    private GlueCatalog(Builder builder) {
        this.region = builder.region;
        this.accessKeyId = builder.accessKeyId;
        this.secretAccessKey = builder.secretAccessKey;
        this.endpoint = builder.endpoint;
        this.databases = new HashMap<>();
        this.tables = new HashMap<>();
    }

    public static Builder builder() { return new Builder(); }

    @Override
    public List<String> listNamespaces() {
        return new ArrayList<>(databases.keySet());
    }

    @Override
    public List<String> listNamespaces(String parent) {
        return new ArrayList<>();
    }

    @Override
    public void createNamespace(String namespace, Map<String, String> properties) {
        databases.put(namespace, new DatabaseInfo(namespace, properties));
    }

    @Override
    public void dropNamespace(String namespace) {
        databases.remove(namespace);
        tables.entrySet().removeIf(e -> e.getKey().startsWith(namespace + "."));
    }

    @Override
    public Map<String, String> getNamespaceProperties(String namespace) {
        DatabaseInfo db = databases.get(namespace);
        return db != null ? new HashMap<>(db.properties) : Collections.emptyMap();
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

    public String region() { return region; }

    private static final class DatabaseInfo {
        final String name;
        final Map<String, String> properties;
        DatabaseInfo(String name, Map<String, String> properties) {
            this.name = name;
            this.properties = properties != null ? properties : new HashMap<>();
        }
    }

    public static final class Builder {
        private String region = "us-east-1";
        private String accessKeyId;
        private String secretAccessKey;
        private String endpoint;

        public Builder region(String r) { this.region = r; return this; }
        public Builder accessKeyId(String key) { this.accessKeyId = key; return this; }
        public Builder secretAccessKey(String key) { this.secretAccessKey = key; return this; }
        public Builder endpoint(String ep) { this.endpoint = ep; return this; }
        public GlueCatalog build() { return new GlueCatalog(this); }
    }
}
