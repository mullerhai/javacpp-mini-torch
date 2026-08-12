/*
 * CronSchedule -- smallest viable cron expression evaluator.
 *
 * <p>Supports five fields (minute, hour, day-of-month, month, day-of-week)
 * with the conventional syntax:
 * <ul>
 *   <li>{@code *} -- every value</li>
 *   <li>{@code 5} -- literal value</li>
 *   <li>{@code 1,3,5} -- list</li>
 *   <li>{@code 0-15} -- range</li>
 *   <li>{@code *}{@code /step} -- every step (e.g. {@code *}{@code /5})</li>
 * </ul>
 *
 * <p>Day-of-week is 0-6 with Sunday=0. Naming is a compatibility aid only.
 *
 * <p>This is intentionally not a full cron parser -- production deployments
 * should swap in Quartz / cron-utils. The minimum goal is "every minute",
 * "every 5 minutes", "every hour", "every day at 02:00".
 */
package org.bytedeco.pytorch.cache.pipeline;

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Arrays;

public final class CronSchedule {

    private final String expr;
    private final int[] minute;
    private final int[] hour;
    private final int[] dom;
    private final int[] month;
    private final int[] dow;

    public CronSchedule(String expr) {
        this.expr = expr;
        if (expr == null) throw new IllegalArgumentException("expr==null");
        String[] parts = expr.trim().split("\\s+");
        if (parts.length != 5) throw new IllegalArgumentException("cron must have 5 fields: " + expr);
        this.minute = parseField(parts[0], 0, 59);
        this.hour   = parseField(parts[1], 0, 23);
        this.dom    = parseField(parts[2], 1, 31);
        this.month  = parseField(parts[3], 1, 12);
        this.dow    = parseField(parts[4], 0, 6);
    }

    public String expression() { return expr; }

    public boolean matches(LocalDateTime t) {
        return contains(minute, t.getMinute())
                && contains(hour, t.getHour())
                && contains(dom, t.getDayOfMonth())
                && contains(month, t.getMonthValue())
                && contains(dow, t.getDayOfWeek().getValue() % 7);
    }

    public LocalDateTime nextAfter(LocalDateTime t) {
        LocalDateTime c = t.plusMinutes(1).withSecond(0).withNano(0);
        for (int i = 0; i < 366 * 24 * 60; i++) {
            if (matches(c)) return c;
            c = c.plusMinutes(1);
        }
        return null;
    }

    private static boolean contains(int[] arr, int v) {
        for (int x : arr) if (x == v) return true;
        return false;
    }

    private static int[] parseField(String f, int min, int max) {
        java.util.List<Integer> out = new java.util.ArrayList<>();
        for (String part : f.split(",")) {
            int step = 1;
            String range = part;
            if (part.contains("/")) {
                String[] bits = part.split("/", 2);
                range = bits[0];
                step = Integer.parseInt(bits[1]);
            }
            int from, to;
            if (range.equals("*") || range.isEmpty()) {
                from = min;
                to = max;
            } else if (range.contains("-")) {
                String[] bits = range.split("-", 2);
                from = Integer.parseInt(bits[0]);
                to = Integer.parseInt(bits[1]);
            } else {
                from = Integer.parseInt(range);
                to = step == 1 ? from : max;
            }
            for (int v = from; v <= to; v += step) {
                if (v >= min && v <= max) out.add(v);
            }
        }
        int[] arr = new int[out.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = out.get(i);
        Arrays.sort(arr);
        return arr;
    }

    public static CronSchedule everyMinute() {
        return new CronSchedule("* * * * *");
    }

    public static CronSchedule everyNMinutes(int n) {
        if (n < 1 || n > 59) throw new IllegalArgumentException("n must be 1..59");
        return new CronSchedule("*/" + n + " * * * *");
    }

    public static CronSchedule everyHour() {
        return new CronSchedule("0 * * * *");
    }

    public static CronSchedule dailyAt(int hour, int minute) {
        if (hour < 0 || hour > 23) throw new IllegalArgumentException("hour");
        if (minute < 0 || minute > 59) throw new IllegalArgumentException("minute");
        return new CronSchedule(minute + " " + hour + " * * *");
    }

    public static LocalDateTime nowIn(ZoneId zone) {
        return LocalDateTime.now(zone);
    }
}
