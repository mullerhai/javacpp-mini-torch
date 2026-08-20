/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.formatter;

import org.bytedeco.pytorch.deploy.argparse.Action;
import org.bytedeco.pytorch.deploy.argparse.ArgparseConstants;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Mirrors Python's {@code argparse.HelpFormatter}. The Java port is line-oriented
 * (no ANSI coloring by default; pass {@link #setColor(boolean)} to enable), and
 * the section model is identical to the Python implementation:
 *
 * <pre>
 *   _root_section = _Section(this, null)
 *   _current_section = _root_section
 *   start_section(heading) pushes a new _Section; end_section() pops it
 *   add_text / add_usage / add_argument appends to _current_section
 * </pre>
 *
 * <p>{@link #formatHelp()} walks the section tree and produces the final string.
 */
public class HelpFormatter {

    /** Section container, mirrors Python's nested {@code _Section}. */
    public static final class Section {
        private final HelpFormatter formatter;
        private final Section parent;
        private final String heading;
        private final List<Object[]> items = new ArrayList<>();

        public Section(HelpFormatter formatter, Section parent, String heading) {
            this.formatter = formatter;
            this.parent = parent;
            this.heading = heading;
        }

        public HelpFormatter getFormatter() { return formatter; }
        public Section getParent() { return parent; }
        public String getHeading() { return heading; }
        public List<Object[]> getItems() { return items; }

        public void addItem(Object fn, Object[] args) {
            items.add(new Object[] { fn, args });
        }

        /** Format this section recursively. Mirrors Python's {@code _Section.format_help}. */
        public String formatHelp() {
            if (parent != null) formatter._indent();
            StringBuilder sb = new StringBuilder();
            for (Object[] item : items) {
                Object fn = item[0];
                Object argsObj = item[1];
                sb.append(formatter.invokeRender(fn, argsObj));
            }
            String rendered = sb.toString();
            if (parent != null) formatter._dedent();
            if (rendered.isEmpty()) return "";
            String heading = "";
            if (heading != null && !ArgparseConstants.SUPPRESS.equals(heading)) {
                int indent = formatter.currentIndent();
                heading = indentString(indent) + heading + ":\n";
            }
            return "\n" + heading + rendered + "\n";
        }
    }

    // ---- state ----
    protected String prog;
    protected int indentIncrement = 2;
    protected int maxHelpPosition = 24;
    protected int width = 80;
    protected boolean color = false;

    protected int currentIndent = 0;
    protected int level = 0;
    protected int actionMaxLength = 0;

    protected Section rootSection;
    protected Section currentSection;

    protected final Pattern whitespaceMatcher = Pattern.compile("\\s+");
    protected final Pattern longBreakMatcher = Pattern.compile("\n\n\n+");

    // pre-compiled nargs patterns — equivalent of Python's regex strings
    public static final Pattern NARGS_DEFAULT_OPT = Pattern.compile("([A])");
    public static final Pattern NARGS_DEFAULT_POS = Pattern.compile("(-*A-*)");
    public static final Pattern NARGS_OPTIONAL_OPT = Pattern.compile("(A?)");
    public static final Pattern NARGS_OPTIONAL_POS = Pattern.compile("(-*A?-*)");
    public static final Pattern NARGS_ZERO_OPT = Pattern.compile("(A*)");
    public static final Pattern NARGS_ZERO_POS = Pattern.compile("(-*[A-]*)");
    public static final Pattern NARGS_ONE_OPT = Pattern.compile("(A+)");
    public static final Pattern NARGS_ONE_POS = Pattern.compile("(-*A[A-]*)");
    public static final Pattern NARGS_REMAINDER_OPT = Pattern.compile("([AO]*)");
    public static final Pattern NARGS_REMAINDER_POS = Pattern.compile("(.*)");
    public static final Pattern NARGS_PARSER_OPT = Pattern.compile("(A[AO]*)");
    public static final Pattern NARGS_PARSER_POS = Pattern.compile("(-*A[-AO]*)");
    public static final Pattern NARGS_SUPPRESS_OPT = Pattern.compile("()");
    public static final Pattern NARGS_SUPPRESS_POS = Pattern.compile("(-*)");

    public HelpFormatter(String prog) {
        this.prog = prog;
        this.width = detectWidth();
        this.rootSection = new Section(this, null, null);
        this.currentSection = rootSection;
        setColor(false);
    }

    public HelpFormatter(String prog, int indentIncrement, int maxHelpPosition, int width) {
        this(prog);
        if (indentIncrement > 0) this.indentIncrement = indentIncrement;
        if (maxHelpPosition > 0) this.maxHelpPosition = maxHelpPosition;
        if (width > 0) this.width = width;
        this.maxHelpPosition = Math.min(this.maxHelpPosition, Math.max(this.width - 20, indentIncrement * 2));
    }

    public String getProg() { return prog; }
    public void setProg(String p) { this.prog = p; }
    public int getWidth() { return width; }
    public void setWidth(int w) { this.width = w; }
    public boolean isColor() { return color; }
    public void setColor(boolean c) { this.color = c; }

    protected static int detectWidth() {
        // try COLUMNS
        String cols = System.getenv("COLUMNS");
        if (cols != null) {
            try { return Integer.parseInt(cols.trim()) - 2; } catch (NumberFormatException ignored) {}
        }
        // try stty
        try {
            Process p = new ProcessBuilder("stty", "size").redirectInput(ProcessBuilder.Redirect.INHERIT)
                    .start();
            byte[] bytes = p.getInputStream().readAllBytes();
            String out = new String(bytes).trim();
            p.waitFor();
            String[] parts = out.split("\\s+");
            if (parts.length == 2) {
                int w = Integer.parseInt(parts[1]);
                if (w > 0) return w - 2;
            }
        } catch (Exception ignored) {}
        // fallback
        return 80;
    }

    public void _indent() { currentIndent += indentIncrement; level++; }
    public void _dedent() {
        currentIndent -= indentIncrement;
        if (currentIndent < 0) throw new IllegalStateException("Indent decreased below 0.");
        level--;
    }
    public int currentIndent() { return currentIndent; }

    public void startSection(String heading) {
        _indent();
        Section s = new Section(this, currentSection, heading);
        currentSection.addItem(s, new Object[0]);
        currentSection = s;
    }

    public void endSection() {
        currentSection = currentSection.parent;
        _dedent();
    }

    public void addText(String text) {
        if (text == null || ArgparseConstants.SUPPRESS.equals(text)) return;
        currentSection.addItem(this, new Object[] { (java.util.function.Function<String, String>) this::_formatText, text });
    }

    public void addUsage(String usage, List<Action> actions, List<?> groups) {
        addUsage(usage, actions, groups, null);
    }

    public void addUsage(String usage, List<Action> actions, List<?> groups, String prefix) {
        if (!ArgparseConstants.SUPPRESS.equals(usage)) {
            currentSection.addItem(this, new Object[] {
                    (java.util.function.Function<Object[], String>) this::_formatUsage,
                    new Object[] { usage, actions, groups, prefix } });
        }
    }

    public void addArgument(Action action) {
        if (ArgparseConstants.SUPPRESS.equals(action.getHelp())) return;
        // compute invocation length
        String invocation = _formatActionInvocation(action);
        int len = stripAnsi(invocation).length() + currentIndent;
        for (Action sub : action.getSubactions()) {
            len = Math.max(len, stripAnsi(_formatActionInvocation(sub)).length() + currentIndent);
        }
        actionMaxLength = Math.max(actionMaxLength, len);
        currentSection.addItem(this, new Object[] {
                (java.util.function.Function<Action, String>) this::_formatAction, action });
    }

    public void addArguments(List<Action> actions) {
        for (Action a : actions) addArgument(a);
    }

    public String formatHelp() {
        String help = rootSection.formatHelp();
        if (!help.isEmpty()) {
            help = longBreakMatcher.matcher(help).replaceAll("\n\n");
            help = help.strip() + "\n";
        }
        return help;
    }

    /** Render one (fn, args) item. Mirrors Python {@code _join_parts}. */
    @SuppressWarnings({"rawtypes", "unchecked"})
    public String invokeRender(Object fn, Object argsArray) {
        Object result;
        if (fn instanceof Section) {
            return ((Section) fn).formatHelp();
        }
        Object[] args;
        if (argsArray instanceof Object[]) {
            args = (Object[]) argsArray;
        } else if (argsArray == null) {
            args = new Object[0];
        } else {
            args = new Object[] { argsArray };
        }
        if (fn instanceof java.util.function.Function) {
            result = ((java.util.function.Function) fn).apply(args.length == 1 ? args[0] : args);
        } else {
            return "";
        }
        return result == null ? "" : result.toString();
    }

    String _formatUsage(Object... args) {
        return _formatUsageImpl((String) args[0], (List<Action>) args[1], (List<?>) args[2], (String) args[3]);
    }

    protected String _formatUsageImpl(String usage, List<Action> actions, List<?> groups, String prefix) {
        if (prefix == null) prefix = "usage: ";
        if (usage != null) {
            usage = usage.replace("%(prog)s", prog);
        } else if (actions == null || actions.isEmpty()) {
            usage = prog;
        } else {
            Object[] parts = _getActionsUsageParts(actions, groups);
            @SuppressWarnings("unchecked")
            List<String> ps = (List<String>) parts[0];
            int posStart = (int) parts[1];
            usage = prog + " " + String.join(" ", ps);
            int textWidth = width - currentIndent;
            String stripped = stripAnsi(usage);
            if (prefix.length() + stripped.length() > textWidth) {
                String[] optParts = ps.subList(0, posStart).toArray(new String[0]);
                String[] posParts = ps.subList(posStart, ps.size()).toArray(new String[0]);
                usage = wrapUsage(prefix, prog, optParts, posParts, textWidth);
            }
        }
        return prefix + usage + "\n\n";
    }

    private String wrapUsage(String prefix, String progStr, String[] optParts, String[] posParts, int textWidth) {
        int progLen = stripAnsi(progStr).length();
        List<String> lines = new ArrayList<>();
        if (prefix.length() + progLen <= 0.75 * textWidth) {
            String indent = repeat(' ', prefix.length() + progLen + 1);
            if (optParts.length > 0) {
                lines.addAll(getLines(prefix, new String[] { progStr }, indent));
                lines.addAll(getLines(indent, optParts, null));
                lines.addAll(getLines(indent, posParts, null));
            } else {
                lines.addAll(getLines(prefix, prepend(progStr, posParts), indent));
            }
        } else {
            String indent = repeat(' ', prefix.length());
            if (optParts.length > 0) {
                lines.addAll(getLines(indent, optParts, null));
            }
            lines.addAll(getLines(indent, posParts, null));
            if (lines.size() > 1) {
                lines = new ArrayList<>();
                lines.addAll(getLines(indent, optParts, null));
                lines.addAll(getLines(indent, posParts, null));
            }
            lines.add(0, progStr);
        }
        return String.join("\n", lines);
    }

    private String[] prepend(String first, String[] rest) {
        String[] out = new String[rest.length + 1];
        out[0] = first;
        System.arraycopy(rest, 0, out, 1, rest.length);
        return out;
    }

    private List<String> getLines(String prefix, String[] parts, String indent) {
        List<String> lines = new ArrayList<>();
        StringBuilder line = new StringBuilder();
        int lineLen = indent == null ? prefix.length() - 1 : indent.length() - 1;
        if (indent == null) line.append(prefix);
        for (String part : parts) {
            int partLen = stripAnsi(part).length();
            if (lineLen + 1 + partLen > width - currentIndent && line.length() > 0) {
                String s = indent == null ? line.toString() : indent + line;
                lines.add(s);
                line = new StringBuilder();
                lineLen = (indent == null ? prefix.length() : indent.length()) - 1;
            }
            if (line.length() > 0) line.append(' ');
            line.append(part);
            lineLen += partLen + 1;
        }
        if (line.length() > 0) {
            String s = indent == null ? line.toString() : indent + line;
            lines.add(s);
        }
        if (indent == null && !lines.isEmpty()) {
            lines.set(0, lines.get(0).substring(prefix.length()));
        }
        return lines;
    }

    Object[] _getActionsUsageParts(List<Action> actions, List<?> groups) {
        List<Action> filtered = new ArrayList<>();
        for (Action a : actions) {
            if (!ArgparseConstants.SUPPRESS.equals(a.getHelp())) filtered.add(a);
        }
        java.util.Map<Action, Object> actionGroups = new java.util.LinkedHashMap<>();
        for (Action a : filtered) actionGroups.put(a, null);
        for (Object g : groups) {
            try {
                java.lang.reflect.Method m = g.getClass().getMethod("getGroupActions");
                Object list = m.invoke(g);
                if (list instanceof Iterable<?> it) {
                    for (Object a : it) {
                        if (actionGroups.containsKey(a)) actionGroups.put((Action) a, g);
                    }
                }
            } catch (ReflectiveOperationException ignored) {}
        }
        List<Object[]> positionals = new ArrayList<>();
        for (Action a : filtered) {
            if (a.getOptionStrings() == null || a.getOptionStrings().isEmpty()) {
                Object g = actionGroups.remove(a);
                if (g != null) {
                    List<Action> groupActions = new ArrayList<>();
                    try {
                        java.lang.reflect.Method m = g.getClass().getMethod("getGroupActions");
                        Object list = m.invoke(g);
                        if (list instanceof Iterable<?> it) {
                            for (Object a2 : it) {
                                if (((Action) a2).getOptionStrings() != null
                                        && !((Action) a2).getOptionStrings().isEmpty()
                                        && actionGroups.remove(a2) != null) {
                                    groupActions.add((Action) a2);
                                }
                            }
                        }
                    } catch (ReflectiveOperationException ignored) {}
                    groupActions.add(a);
                    try {
                        java.lang.reflect.Method gr = g.getClass().getMethod("isRequired");
                        positionals.add(new Object[] { gr.invoke(g), groupActions });
                    } catch (ReflectiveOperationException ignored) {
                        positionals.add(new Object[] { null, groupActions });
                    }
                } else {
                    positionals.add(new Object[] { null, List.of(a) });
                }
            }
        }
        List<Object[]> optionals = new ArrayList<>();
        for (Action a : filtered) {
            if (a.getOptionStrings() != null && !a.getOptionStrings().isEmpty()
                    && actionGroups.containsKey(a)) {
                Object g = actionGroups.remove(a);
                if (g != null) {
                    List<Action> groupActions = new ArrayList<>();
                    groupActions.add(a);
                    try {
                        java.lang.reflect.Method m = g.getClass().getMethod("getGroupActions");
                        Object list = m.invoke(g);
                        if (list instanceof Iterable<?> it) {
                            for (Object a2 : it) {
                                if (((Action) a2).getOptionStrings() != null
                                        && !((Action) a2).getOptionStrings().isEmpty()
                                        && actionGroups.remove(a2) != null) {
                                    groupActions.add((Action) a2);
                                }
                            }
                        }
                        java.lang.reflect.Method gr = g.getClass().getMethod("isRequired");
                        optionals.add(new Object[] { gr.invoke(g), groupActions });
                    } catch (ReflectiveOperationException ignored) {
                        optionals.add(new Object[] { null, groupActions });
                    }
                } else {
                    optionals.add(new Object[] { null, List.of(a) });
                }
            }
        }
        List<String> parts = new ArrayList<>();
        Integer posStart = null;
        int i = 0;
        int totalOptionals = optionals.size();
        for (Object[] pair : new ArrayList<Object[]>() {{
            addAll(optionals); addAll(positionals);
        }}) {
            if (i == totalOptionals) posStart = parts.size();
            boolean required = Boolean.TRUE.equals(pair[0]);
            @SuppressWarnings("unchecked")
            List<Action> group = (List<Action>) pair[1];
            boolean inGroup = group.size() > 1;
            int start = parts.size();
            for (Action action : group) {
                String part;
                if (action.getOptionStrings() == null || action.getOptionStrings().isEmpty()) {
                    Object defaultMetavar = _getDefaultMetavarForPositional(action);
                    part = _formatArgs(action, defaultMetavar);
                    if (inGroup && part.startsWith("[") && part.endsWith("]")) {
                        part = part.substring(1, part.length() - 1);
                    }
                } else {
                    String optionString = action.getOptionStrings().get(0);
                    if (nargsIsZero(action)) {
                        part = action.formatUsage();
                    } else {
                        Object defaultMetavar = _getDefaultMetavarForOptional(action);
                        String argsString = _formatArgs(action, defaultMetavar);
                        part = optionString + " " + argsString;
                    }
                    if (!(action.isRequired() || required || inGroup)) {
                        part = "[" + part + "]";
                    }
                }
                parts.add(part);
            }
            if (inGroup) {
                char open = required ? '(' : '[';
                char close = required ? ')' : ']';
                parts.set(start, open + parts.get(start));
                for (int j = start; j < parts.size() - 1; j++) {
                    parts.set(j, parts.get(j) + " |");
                }
                parts.set(parts.size() - 1, parts.get(parts.size() - 1) + close);
            }
            i++;
        }
        if (posStart == null) posStart = parts.size();
        return new Object[] { parts, posStart };
    }

    String _formatText(String text) {
        if (text != null && text.contains("%(prog)")) {
            text = text.replace("%(prog)s", prog);
        }
        int textWidth = Math.max(width - currentIndent, 11);
        String indentStr = repeat(' ', currentIndent);
        return _fillText(text, textWidth, indentStr) + "\n\n";
    }

    String _formatAction(Action action) {
        int helpPosition = Math.min(actionMaxLength + 2, maxHelpPosition);
        int helpWidth = Math.max(width - helpPosition, 11);
        int actionWidth = helpPosition - currentIndent - 2;
        String actionHeader = _formatActionInvocation(action);
        String headerPlain = stripAnsi(actionHeader);

        if (action.getHelp() == null || action.getHelp().isEmpty()) {
            actionHeader = String.format(Locale.ROOT, "%" + currentIndent + "s%s\n", "", actionHeader);
        } else if (headerPlain.length() <= actionWidth) {
            actionHeader = String.format(Locale.ROOT, "%" + currentIndent + "s%-" + actionWidth + "s  ",
                    "", headerPlain).replace(headerPlain, actionHeader);
        } else {
            actionHeader = String.format(Locale.ROOT, "%" + currentIndent + "s%s\n", "", actionHeader);
        }
        StringBuilder parts = new StringBuilder();
        parts.append(actionHeader);
        if (action.getHelp() != null && !action.getHelp().trim().isEmpty()) {
            String helpText = _expandHelp(action);
            if (helpText != null && !helpText.isEmpty()) {
                List<String> helpLines = _splitLines(helpText, helpWidth);
                int indentFirst = headerPlain.length() <= actionWidth ? 0 : helpPosition;
                if (!helpLines.isEmpty()) {
                    parts.append(String.format(Locale.ROOT, "%" + indentFirst + "s%s\n", "", helpLines.get(0)));
                    for (int j = 1; j < helpLines.size(); j++) {
                        parts.append(String.format(Locale.ROOT, "%" + helpPosition + "s%s\n", "", helpLines.get(j)));
                    }
                }
            }
        } else if (!actionHeader.endsWith("\n")) {
            parts.append("\n");
        }
        for (Action sub : action.getSubactions()) {
            parts.append(_formatAction(sub));
        }
        return parts.toString();
    }

    public String _formatActionInvocation(Action action) {
        if (action.getOptionStrings() == null || action.getOptionStrings().isEmpty()) {
            Object defaultMetavar = _getDefaultMetavarForPositional(action);
            return String.join(" ", _metavarFormatter(action, defaultMetavar).apply(1));
        }
        if (nargsIsZero(action)) {
            List<String> strings = new ArrayList<>();
            for (String s : action.getOptionStrings()) strings.add(s);
            return String.join(", ", strings);
        }
        Object defaultMetavar = _getDefaultMetavarForOptional(action);
        List<String> strings = new ArrayList<>();
        for (String s : action.getOptionStrings()) strings.add(s);
        return String.join(", ", strings) + " " + _formatArgs(action, defaultMetavar);
    }

    private boolean nargsIsZero(Action action) {
        Object n = action.getNargs();
        return n instanceof Integer && ((Integer) n) == 0;
    }

    public java.util.function.Function<Integer, List<String>> _metavarFormatter(Action action, Object defaultMetavar) {
        Object result;
        if (action.getMetavar() != null) {
            result = action.getMetavar();
        } else if (action.getChoices() != null) {
            result = "{" + joinChoices(action.getChoices()) + "}";
        } else {
            result = defaultMetavar;
        }
        Object finalResult = result;
        return tupleSize -> {
            if (finalResult instanceof Object[] arr) {
                List<String> out = new ArrayList<>();
                for (Object o : arr) out.add(o == null ? "" : o.toString());
                return out;
            }
            if (finalResult instanceof List<?> list) {
                List<String> out = new ArrayList<>();
                for (Object o : list) out.add(o == null ? "" : o.toString());
                return out;
            }
            String s = finalResult == null ? "" : finalResult.toString();
            List<String> out = new ArrayList<>();
            for (int i = 0; i < tupleSize; i++) out.add(s);
            return out;
        };
    }

    public String _formatArgs(Action action, Object defaultMetavar) {
        java.util.function.Function<Integer, List<String>> getMetavar = _metavarFormatter(action, defaultMetavar);
        Object n = action.getNargs();
        String[] sv = getMetavar.apply(1).toArray(new String[0]);
        if (n == null) {
            return sv[0];
        } else if (ArgparseConstants.OPTIONAL.equals(n)) {
            return "[" + sv[0] + "]";
        } else if (ArgparseConstants.ZERO_OR_MORE.equals(n)) {
            if (sv.length == 2) return "[" + sv[0] + " [" + sv[1] + " ...]]";
            return "[" + sv[0] + " ...]";
        } else if (ArgparseConstants.ONE_OR_MORE.equals(n)) {
            String[] sv2 = getMetavar.apply(2).toArray(new String[0]);
            return sv2[0] + " [" + sv2[1] + " ...]";
        } else if (ArgparseConstants.REMAINDER.equals(n)) {
            return "...";
        } else if (ArgparseConstants.PARSER.equals(n)) {
            return sv[0] + " ...";
        } else if (ArgparseConstants.SUPPRESS.equals(n)) {
            return "";
        } else if (n instanceof Integer) {
            int count = (Integer) n;
            String[] parts = getMetavar.apply(count).toArray(new String[0]);
            return String.join(" ", parts);
        } else {
            throw new IllegalArgumentException("invalid nargs value: " + n);
        }
    }

    public String _expandHelp(Action action) {
        String helpString = _getHelpString(action);
        if (helpString == null || !helpString.contains("%")) return helpString == null ? "" : helpString;
        java.util.Map<String, Object> params = new java.util.HashMap<>();
        params.put("prog", prog);
        for (String name : new String[] {"option_strings", "dest", "nargs", "const",
                "default", "type", "choices", "required", "help", "metavar", "deprecated"}) {
            Object v = action.readAttribute(name);
            if (v == ArgparseConstants.SUPPRESS) continue;
            if (v instanceof java.lang.reflect.Method) {
                try { v = ((java.lang.reflect.Method) v).invoke(action); } catch (Exception ignored) {}
            }
            if (v != null) params.put(name, v);
        }
        if (params.get("choices") instanceof Iterable<?> it) {
            StringBuilder sb = new StringBuilder();
            for (Object c : it) {
                if (sb.length() > 0) sb.append(", ");
                sb.append(c);
            }
            params.put("choices", sb.toString());
        }
        // expand
        StringBuilder sb = new StringBuilder();
        Matcher m = Pattern.compile("%\\(([^)]+)\\)s").matcher(helpString);
        while (m.find()) {
            String k = m.group(1);
            Object v = params.get(k);
            m.appendReplacement(sb, Matcher.quoteReplacement(v == null ? "" : v.toString()));
        }
        m.appendTail(sb);
        return sb.toString();
    }

    public String _getHelpString(Action action) {
        return action.getHelp();
    }

    public Object _getDefaultMetavarForOptional(Action action) {
        return action.getDest() == null ? "" : action.getDest().toString().toUpperCase(Locale.ROOT);
    }

    public Object _getDefaultMetavarForPositional(Action action) {
        return action.getDest() == null ? "" : action.getDest().toString();
    }

    public List<String> _splitLines(String text, int width) {
        text = whitespaceMatcher.matcher(text).replaceAll(" ").trim();
        return wrap(text, width);
    }

    public String _fillText(String text, int width, String indent) {
        text = whitespaceMatcher.matcher(text).replaceAll(" ").trim();
        List<String> lines = wrap(text, width);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < lines.size(); i++) {
            if (i == 0) sb.append(indent);
            sb.append(lines.get(i));
            if (i < lines.size() - 1) sb.append("\n").append(indent);
        }
        return sb.toString();
    }

    /** Lightweight textwrap (long-word-safe) replacement for Python's textwrap. */
    public static List<String> wrap(String text, int width) {
        if (text == null || text.isEmpty()) return new ArrayList<>();
        List<String> out = new ArrayList<>();
        String[] words = text.split("\\s+");
        StringBuilder cur = new StringBuilder();
        for (String w : words) {
            if (cur.length() == 0) {
                cur.append(w);
            } else if (cur.length() + 1 + w.length() <= width) {
                cur.append(' ').append(w);
            } else {
                out.add(cur.toString());
                cur.setLength(0);
                cur.append(w);
            }
        }
        if (cur.length() > 0) out.add(cur.toString());
        return out;
    }

    protected static String indentString(int n) {
        return repeat(' ', n);
    }

    protected static String repeat(char c, int n) {
        if (n <= 0) return "";
        char[] arr = new char[n];
        java.util.Arrays.fill(arr, c);
        return new String(arr);
    }

    private static String joinChoices(Object choices) {
        if (choices instanceof Iterable<?> it) {
            StringBuilder sb = new StringBuilder();
            for (Object c : it) {
                if (sb.length() > 0) sb.append(",");
                sb.append(c);
            }
            return sb.toString();
        }
        return choices == null ? "" : choices.toString();
    }

    /** Strip ANSI escape sequences (no-op when color=false). */
    public static String stripAnsi(String s) {
        if (s == null) return null;
        return s.replaceAll("\u001b\\[[0-9;]*m", "");
    }
}