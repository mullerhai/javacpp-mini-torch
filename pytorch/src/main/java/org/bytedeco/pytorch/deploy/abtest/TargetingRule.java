/*
 * Targeting rules — declarative predicates that gate experiment eligibility.
 *
 * Industry usage:
 *   - Meta XP: TargetingExpr combining demographic, behavioral and event-based
 *     filters (e.g. "country IN (US, CA) AND last_visit > 7d AND age > 18").
 *   - Google Ads experiment: audience targeting with custom segments.
 *   - ByteDance Libra: app version / platform / crowd filters per experiment.
 *   - Alibaba A/B: crowd + city + scene + time-of-day targeting.
 *   - Tencent: bitmask cohort ids + tag list.
 *
 * Rule design:
 *   - Pure, side-effect-free predicates over {@link DiversionContext}.
 *   - Composable via AND / OR / NOT.
 *   - Hashable for caching the compiled rule tree.
 *
 * Operators covered (matching common production surfaces):
 *   eq, neq, in, not_in, contains, prefix, suffix, regex,
 *   gte, lte, gt, lt (lexicographic or numeric),
 *   cohort, any_cohort.
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Single targeting predicate and composite rule builders.
 *
 * <p>This is a small DSL — enough for ~95% of production targeting but easy
 * to inspect, log and unit-test. For very complex audiences, production
 * systems fall back to user-defined compiled functions.
 */
public final class TargetingRule {

    private TargetingRule() {}

    /** Marker interface for composable rules. */
    public interface Rule {
        boolean matches(DiversionContext ctx);
    }

    /** Logical NOT. */
    public static Rule not(Rule inner) {
        Objects.requireNonNull(inner, "inner");
        return ctx -> !inner.matches(ctx);
    }

    /** Logical AND with short-circuit. */
    public static Rule and(Rule... rs) {
        if (rs == null || rs.length == 0) return ALWAYS;
        List<Rule> list = Arrays.asList(rs);
        return ctx -> {
            for (Rule r : list) if (!r.matches(ctx)) return false;
            return true;
        };
    }

    /** Logical OR with short-circuit. */
    public static Rule or(Rule... rs) {
        if (rs == null || rs.length == 0) return NEVER;
        List<Rule> list = Arrays.asList(rs);
        return ctx -> {
            for (Rule r : list) if (r.matches(ctx)) return true;
            return false;
        };
    }

    /** Constant true. */
    public static final Rule ALWAYS = ctx -> true;

    /** Constant false. */
    public static final Rule NEVER = ctx -> false;

    // ---- attribute rules ----------------------------------------------------

    /** {@code attr(key) == value} (case-insensitive string compare). */
    public static Rule attrEq(String key, String value) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        return ctx -> value.equalsIgnoreCase(ctx.attribute(key, ""));
    }

    /** {@code attr(key) != value}. */
    public static Rule attrNeq(String key, String value) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        return ctx -> !value.equalsIgnoreCase(ctx.attribute(key, ""));
    }

    /** {@code attr(key) IN values} (case-insensitive). */
    public static Rule attrIn(String key, String... values) {
        Objects.requireNonNull(key, "key");
        Set<String> set = new LinkedHashSet<>();
        for (String v : values) set.add(v.toLowerCase(Locale.ROOT));
        return ctx -> set.contains(ctx.attribute(key, "").toLowerCase(Locale.ROOT));
    }

    /** {@code attr(key) NOT IN values}. */
    public static Rule attrNotIn(String key, String... values) {
        Objects.requireNonNull(key, "key");
        Set<String> set = new LinkedHashSet<>();
        for (String v : values) set.add(v.toLowerCase(Locale.ROOT));
        return ctx -> !set.contains(ctx.attribute(key, "").toLowerCase(Locale.ROOT));
    }

    /** {@code attr(key).contains(substring)} (case-insensitive). */
    public static Rule attrContains(String key, String substring) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(substring, "substring");
        return ctx -> ctx.attribute(key, "").toLowerCase(Locale.ROOT)
                .contains(substring.toLowerCase(Locale.ROOT));
    }

    /** {@code attr(key).startsWith(prefix)} (case-insensitive). */
    public static Rule attrPrefix(String key, String prefix) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(prefix, "prefix");
        return ctx -> ctx.attribute(key, "").toLowerCase(Locale.ROOT)
                .startsWith(prefix.toLowerCase(Locale.ROOT));
    }

    /** Regex on attr(key). */
    public static Rule attrRegex(String key, String regex) {
        Objects.requireNonNull(key, "key");
        Pattern p = Pattern.compile(regex);
        return ctx -> {
            String v = ctx.attribute(key, "");
            return !v.isEmpty() && p.matcher(v).find();
        };
    }

    /** Numeric >= on attr(key). Non-numeric values treated as -infinity. */
    public static Rule attrGte(String key, double threshold) {
        Objects.requireNonNull(key, "key");
        return ctx -> safeParseDouble(ctx.attribute(key, null)) >= threshold;
    }

    /** Numeric <= on attr(key). */
    public static Rule attrLte(String key, double threshold) {
        Objects.requireNonNull(key, "key");
        return ctx -> safeParseDouble(ctx.attribute(key, null)) <= threshold;
    }

    /** Cohort flag check (must be true). */
    public static Rule cohort(String cohortId) {
        Objects.requireNonNull(cohortId, "cohortId");
        return ctx -> ctx.hasCohort(cohortId);
    }

    /** True if unit is in any of the listed cohorts. */
    public static Rule anyCohort(Set<String> cohortIds) {
        Objects.requireNonNull(cohortIds, "cohortIds");
        return ctx -> ctx.inAnyCohort(cohortIds);
    }

    /** Convenience: country match against ISO codes. */
    public static Rule country(String... isoCodes) {
        return attrIn("country", isoCodes);
    }

    /** Convenience: app version prefix match (e.g. "1.2.0"). */
    public static Rule appVersionAtLeast(String minVersion) {
        return ctx -> {
            String cur = ctx.appVersion();
            if (cur.isEmpty() || minVersion == null || minVersion.isEmpty()) return false;
            return compareVersion(cur, minVersion) >= 0;
        };
    }

    /** Convenience: platform match. */
    public static Rule platform(String... platforms) {
        return attrIn("platform", platforms);
    }

    private static double safeParseDouble(String s) {
        if (s == null || s.isEmpty()) return Double.NEGATIVE_INFINITY;
        try {
            return Double.parseDouble(s);
        } catch (NumberFormatException e) {
            return Double.NEGATIVE_INFINITY;
        }
    }

    private static int compareVersion(String a, String b) {
        String[] ap = a.split("\\.");
        String[] bp = b.split("\\.");
        int n = Math.max(ap.length, bp.length);
        for (int i = 0; i < n; i++) {
            int av = i < ap.length ? safeInt(ap[i]) : 0;
            int bv = i < bp.length ? safeInt(bp[i]) : 0;
            if (av != bv) return Integer.compare(av, bv);
        }
        return 0;
    }

    private static int safeInt(String s) {
        try { return Integer.parseInt(s.replaceAll("[^0-9-]", "")); }
        catch (NumberFormatException e) { return 0; }
    }
}
