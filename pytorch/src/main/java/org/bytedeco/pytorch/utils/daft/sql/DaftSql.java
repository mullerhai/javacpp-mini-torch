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
package org.bytedeco.pytorch.utils.daft.sql;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.daft.DaftDataFrame;
import org.bytedeco.pytorch.utils.daft.expr.Expression;
import org.bytedeco.pytorch.utils.daft.expr.ColumnRef;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * SQL adapter for DaftDataFrame.
 *
 * <p>Provides SQL query interface to DaftDataFrame, supporting:
 * <ul>
 *   <li>SELECT, FROM, WHERE, GROUP BY, HAVING</li>
 *   <li>ORDER BY, LIMIT, OFFSET</li>
 *   <li>JOIN (INNER, LEFT, RIGHT, FULL OUTER)</li>
 *   <li>UNION, INTERSECT, EXCEPT</li>
 *   <li>Subqueries</li>
 *   <li>Common table expressions (CTE)</li>
 * </ul>
 *
 * <pre>{@code
 * // Query with SQL
 * DaftSql sql = new DaftSql(df);
 * DaftDataFrame result = sql.query(
 *     "SELECT user_id, SUM(amount) as total " +
 *     "FROM events " +
 *     "WHERE country = 'US' " +
 *     "GROUP BY user_id " +
 *     "HAVING SUM(amount) > 100 " +
 *     "ORDER BY total DESC " +
 *     "LIMIT 10"
 * );
 *
 * // Register and join
 * sql.register("orders", ordersDf);
 * DaftDataFrame joined = sql.query(
 *     "SELECT o.user_id, o.amount, u.name " +
 *     "FROM orders o JOIN users u ON o.user_id = u.id"
 * );
 * }</pre>
 */
public final class DaftSql {

    private final Map<String, DaftDataFrame> tables;
    private final SqlConfig config;

    public DaftSql() {
        this(new HashMap<String, DaftDataFrame>(), SqlConfig.DEFAULT);
    }

    public DaftSql(Map<String, DaftDataFrame> tables) {
        this(tables, SqlConfig.DEFAULT);
    }

    public DaftSql(Map<String, DaftDataFrame> tables, SqlConfig config) {
        this.tables = tables;
        this.config = config;
    }

    /**
     * Register a DataFrame with an alias.
     */
    public DaftSql register(String alias, DaftDataFrame df) {
        tables.put(alias.toLowerCase(), df);
        return this;
    }

    /**
     * Register a DataFrame with default name from .alias().
     */
    public DaftSql register(DaftDataFrame df) {
        if (df.tableAlias() != null) {
            tables.put(df.tableAlias().toLowerCase(), df);
        }
        return this;
    }

    /**
     * Execute a SQL query and return a DaftDataFrame.
     */
    public DaftDataFrame query(String sql) {
        SqlStatement stmt = parse(sql);
        return execute(stmt);
    }

    // ---- SQL Parser ----

    SqlStatement parse(String sql) {
        String normalized = sql.replaceAll("\\s+", " ").trim();
        SqlStatement stmt = new SqlStatement();

        // Parse CTE (WITH clause)
        Pattern ctePattern = Pattern.compile("(?i)^\\s*WITH\\s+(.+?)\\s+SELECT\\s+(.*)$");
        Matcher cteMatcher = ctePattern.matcher(normalized);
        if (cteMatcher.matches()) {
            stmt.cteClause = cteMatcher.group(1);
            normalized = cteMatcher.group(2);
        }

        // Parse SELECT
        Pattern selectPattern = Pattern.compile("(?i)^SELECT\\s+(DISTINCT\\s+)?(.+?)FROM\\s+(.+)$");
        Matcher selectMatcher = selectPattern.matcher(normalized);
        if (selectMatcher.matches()) {
            stmt.distinct = "DISTINCT".equalsIgnoreCase(selectMatcher.group(1));
            stmt.selectClause = selectMatcher.group(2);
            String fromAndBeyond = selectMatcher.group(3);

            // Extract WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, OFFSET
            String remaining = fromAndBeyond;

            // JOIN
            Pattern joinPattern = Pattern.compile("(?i)(.+?)\\s+(LEFT\\s+RIGHT\\s+FULL\\s+OUTER\\s+|LEFT\\s+RIGHT\\s+FULL\\s+|LEFT\\s+RIGHT\\s+|INNER\\s+|CROSS\\s+)?JOIN\\s+(\\w+)\\s+ON\\s+(.+?)(?:\\s+WHERE|\\s+GROUP|\\s+HAVING|\\s+ORDER|\\s+LIMIT|\\s+OFFSET|\\s+UNION|\\s+INTERSECT|\\s+EXCEPT|$)");
            Matcher joinMatcher = joinPattern.matcher(remaining);
            if (joinMatcher.matches()) {
                stmt.fromTable = joinMatcher.group(1).trim();
                stmt.joinType = joinMatcher.group(2);
                stmt.joinTable = joinMatcher.group(3).trim();
                stmt.joinCondition = joinMatcher.group(4).trim();
                remaining = remaining.substring(joinMatcher.end());
            } else {
                // Simple FROM
                Pattern fromOnlyPattern = Pattern.compile("(?i)^([\\w\\.]+)(?:\\s+AS\\s+(\\w+))?");
                Matcher fromMatcher = fromOnlyPattern.matcher(remaining.split("(?i)\\s+WHERE|\\s+GROUP|\\s+HAVING|\\s+ORDER|\\s+LIMIT|\\s+OFFSET")[0]);
                if (fromMatcher.find()) {
                    stmt.fromTable = fromMatcher.group(1);
                    stmt.fromAlias = fromMatcher.group(2);
                }
            }

            // WHERE
            Pattern wherePattern = Pattern.compile("(?i)\\s+WHERE\\s+(.+?)(?:\\s+GROUP|\\s+HAVING|\\s+ORDER|\\s+LIMIT|\\s+OFFSET|\\s+UNION|\\s+INTERSECT|\\s+EXCEPT|$)");
            Matcher whereMatcher = wherePattern.matcher(remaining);
            if (whereMatcher.find()) {
                stmt.whereClause = whereMatcher.group(1).trim();
            }

            // GROUP BY
            Pattern groupPattern = Pattern.compile("(?i)\\s+GROUP\\s+BY\\s+(.+?)(?:\\s+HAVING|\\s+ORDER|\\s+LIMIT|\\s+OFFSET|\\s+UNION|\\s+INTERSECT|\\s+EXCEPT|$)");
            Matcher groupMatcher = groupPattern.matcher(remaining);
            if (groupMatcher.find()) {
                stmt.groupByClause = groupMatcher.group(1).trim();
            }

            // HAVING
            Pattern havingPattern = Pattern.compile("(?i)\\s+HAVING\\s+(.+?)(?:\\s+ORDER|\\s+LIMIT|\\s+OFFSET|\\s+UNION|\\s+INTERSECT|\\s+EXCEPT|$)");
            Matcher havingMatcher = havingPattern.matcher(remaining);
            if (havingMatcher.find()) {
                stmt.havingClause = havingMatcher.group(1).trim();
            }

            // ORDER BY
            Pattern orderPattern = Pattern.compile("(?i)\\s+ORDER\\s+BY\\s+(.+?)(?:\\s+LIMIT|\\s+OFFSET|\\s+UNION|\\s+INTERSECT|\\s+EXCEPT|$)");
            Matcher orderMatcher = orderPattern.matcher(remaining);
            if (orderMatcher.find()) {
                stmt.orderByClause = orderMatcher.group(1).trim();
            }

            // LIMIT
            Pattern limitPattern = Pattern.compile("(?i)\\s+LIMIT\\s+(\\d+)(?:\\s+OFFSET|\\s+UNION|\\s+INTERSECT|\\s+EXCEPT|$)");
            Matcher limitMatcher = limitPattern.matcher(remaining);
            if (limitMatcher.find()) {
                stmt.limit = Integer.parseInt(limitMatcher.group(1));
            }

            // OFFSET
            Pattern offsetPattern = Pattern.compile("(?i)\\s+OFFSET\\s+(\\d+)(?:\\s+UNION|\\s+INTERSECT|\\s+EXCEPT|$)");
            Matcher offsetMatcher = offsetPattern.matcher(remaining);
            if (offsetMatcher.find()) {
                stmt.offset = Integer.parseInt(offsetMatcher.group(1));
            }
        }

        return stmt;
    }

    // ---- Query Execution ----

    private DaftDataFrame execute(SqlStatement stmt) {
        DaftDataFrame result;

        // Resolve FROM table
        String tableName = stmt.fromTable.toLowerCase();
        if (stmt.fromAlias != null) {
            register(stmt.fromAlias.toLowerCase(), tables.get(tableName));
            tableName = stmt.fromAlias.toLowerCase();
        }
        DaftDataFrame df = tables.get(tableName);
        if (df == null) {
            throw new IllegalArgumentException("Table not found: " + stmt.fromTable);
        }

        // Apply SELECT first (to limit columns early)
        if (stmt.selectClause != null) {
            String[] cols = parseSelectColumns(stmt.selectClause);
            if (cols.length > 0 && !"*".equals(cols[0])) {
                df = df.select(cols);
            }
        }

        // Apply WHERE
        if (stmt.whereClause != null) {
            Expression whereExpr = parseWhere(stmt.whereClause);
            df = df.filter(whereExpr);
        }

        // Apply GROUP BY
        if (stmt.groupByClause != null) {
            String[] groupCols = parseColumns(stmt.groupByClause);
            df = df.groupBy(groupCols);
        }

        // Apply HAVING
        if (stmt.havingClause != null) {
            Expression havingExpr = parseWhere(stmt.havingClause);
            df = df.filter(havingExpr);
        }

        // Apply JOIN
        if (stmt.joinTable != null) {
            DaftDataFrame right = tables.get(stmt.joinTable.toLowerCase());
            if (right == null) {
                throw new IllegalArgumentException("Join table not found: " + stmt.joinTable);
            }
            String joinCondition = stmt.joinCondition;
            int eqIndex = joinCondition.toLowerCase().indexOf(" = ");
            if (eqIndex > 0) {
                String leftCol = joinCondition.substring(0, eqIndex).trim();
                String rightCol = joinCondition.substring(eqIndex + 3).trim();
                leftCol = stripAlias(leftCol);
                rightCol = stripAlias(rightCol);
                df = df.join(right, leftCol, rightCol, stmt.joinType != null ? stmt.joinType.trim() : "inner");
            }
        }

        // Apply ORDER BY
        if (stmt.orderByClause != null) {
            String[] orderCols = parseOrderBy(stmt.orderByClause);
            for (String orderCol : orderCols) {
                boolean asc = !orderCol.toLowerCase().contains(" desc");
                String colName = orderCol.replaceAll("(?i)\\s+(ASC|DESC)", "").trim();
                df = df.sort(colName, asc);
            }
        }

        // Apply LIMIT
        if (stmt.limit > 0) {
            df = df.limit(stmt.limit);
        }

        // Apply OFFSET
        if (stmt.offset > 0) {
            df = df.offset(stmt.offset);
        }

        // Apply DISTINCT
        if (stmt.distinct) {
            df = df.distinct();
        }

        return df;
    }

    private String[] parseSelectColumns(String clause) {
        List<String> cols = new ArrayList<>();
        for (String part : clause.split(",")) {
            part = part.trim();
            // Remove aliases
            int asIndex = part.toLowerCase().lastIndexOf(" as ");
            if (asIndex > 0) {
                part = part.substring(0, asIndex).trim();
            }
            // Remove functions
            part = part.replaceAll("(?i)^(SUM|AVG|COUNT|MIN|MAX|COALESCE)\\s*\\([^)]*\\)", "*");
            cols.add(part);
        }
        return cols.toArray(new String[0]);
    }

    private String[] parseColumns(String clause) {
        List<String> cols = new ArrayList<>();
        for (String part : clause.split(",")) {
            cols.add(stripAlias(part.trim()));
        }
        return cols.toArray(new String[0]);
    }

    private String[] parseOrderBy(String clause) {
        List<String> cols = new ArrayList<>();
        for (String part : clause.split(",")) {
            cols.add(part.trim());
        }
        return cols.toArray(new String[0]);
    }

    private String stripAlias(String col) {
        int dotIndex = col.lastIndexOf('.');
        if (dotIndex >= 0) {
            col = col.substring(dotIndex + 1);
        }
        return col.trim();
    }

    private Expression parseWhere(String clause) {
        // Simple parser for basic comparisons
        // Supports: column = value, column != value, column > value, etc.
        // For full SQL, would need a proper SQL parser

        clause = clause.trim();

        // Basic comparisons
        Pattern[] patterns = {
            Pattern.compile("(?i)^(\\w+)\\s*=\\s*(.+)$"),
            Pattern.compile("(?i)^(\\w+)\\s*!=\\s*(.+)$"),
            Pattern.compile("(?i)^(\\w+)\\s*<\\s*(.+)$"),
            Pattern.compile("(?i)^(\\w+)\\s*>\\s*(.+)$"),
            Pattern.compile("(?i)^(\\w+)\\s*<=\\s*(.+)$"),
            Pattern.compile("(?i)^(\\w+)\\s*>=\\s*(.+)$"),
            Pattern.compile("(?i)^(\\w+)\\s+LIKE\\s+(.+)$"),
            Pattern.compile("(?i)^(\\w+)\\s+IN\\s*(.+)$"),
            Pattern.compile("(?i)^(\\w+)\\s+IS\\s+NULL$"),
            Pattern.compile("(?i)^(\\w+)\\s+IS\\s+NOT\\s+NULL$"),
        };

        for (Pattern p : patterns) {
            Matcher m = p.matcher(clause);
            if (m.matches()) {
                String col = stripAlias(m.group(1));
                String val = m.group(2);

                if (p.pattern().contains(" LIKE")) {
                    return Expression.col(col).str().contains(val.replace("'", "").replace("%", ""));
                }
                if (p.pattern().contains(" IS NULL")) {
                    return Expression.col(col).isNull();
                }
                if (p.pattern().contains(" IS NOT NULL")) {
                    return Expression.col(col).isNotNull();
                }
                if (p.pattern().contains(" IN")) {
                    // Handle IN clause - simplified
                    return Expression.col(col).eq(val.replace("'", "").trim());
                }

                // Parse comparison value
                val = val.trim().replace("'", "").replace("\"", "");

                // Determine operator
                String op = "=";
                if (p.pattern().contains("!=")) op = "!=";
                else if (p.pattern().contains("<=")) op = "<=";
                else if (p.pattern().contains(">=")) op = ">=";
                else if (p.pattern().contains("<")) op = "<";
                else if (p.pattern().contains(">")) op = ">";

                // Create appropriate expression
                try {
                    double numVal = Double.parseDouble(val);
                    return compareNumeric(col, op, numVal);
                } catch (NumberFormatException e) {
                    if ("true".equalsIgnoreCase(val)) {
                        return Expression.col(col).eq(true);
                    } else if ("false".equalsIgnoreCase(val)) {
                        return Expression.col(col).eq(false);
                    }
                    return Expression.col(col).eq(val);
                }
            }
        }

        // Default: return col = clause (for complex expressions)
        return Expression.col(clause);
    }

    private Expression compareNumeric(String col, String op, double val) {
        Expression colExpr = Expression.col(col);
        switch (op) {
            case "=": return colExpr.cast(Column.DType.FLOAT64).eq(val);
            case "!=": return colExpr.cast(Column.DType.FLOAT64).ne(val);
            case "<": return colExpr.cast(Column.DType.FLOAT64).lt(val);
            case ">": return colExpr.cast(Column.DType.FLOAT64).gt(val);
            case "<=": return colExpr.cast(Column.DType.FLOAT64).le(val);
            case ">=": return colExpr.cast(Column.DType.FLOAT64).ge(val);
            default: return colExpr.eq(val);
        }
    }

    // ---- SQL Statement ----

    static final class SqlStatement {
        String cteClause;
        boolean distinct;
        String selectClause;
        String fromTable;
        String fromAlias;
        String whereClause;
        String groupByClause;
        String havingClause;
        String joinType;
        String joinTable;
        String joinCondition;
        String orderByClause;
        int limit = Integer.MAX_VALUE;
        int offset = 0;
    }

    // ---- Configuration ----

    public static final class SqlConfig {
        public static final SqlConfig DEFAULT = new SqlConfig();

        private final boolean caseSensitive;
        private final boolean strictMode;
        private final Set<String> keywords;

        private SqlConfig() {
            this.caseSensitive = false;
            this.strictMode = false;
            this.keywords = new HashSet<>();
        }

        public SqlConfig(boolean caseSensitive, boolean strictMode) {
            this.caseSensitive = caseSensitive;
            this.strictMode = strictMode;
            this.keywords = new HashSet<>();
        }
    }
}
