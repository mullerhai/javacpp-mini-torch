/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.container;

import org.bytedeco.pytorch.deploy.argparse.*;
import org.bytedeco.pytorch.deploy.argparse.action._AppendAction;
import org.bytedeco.pytorch.deploy.argparse.action._AppendConstAction;
import org.bytedeco.pytorch.deploy.argparse.action._CountAction;
import org.bytedeco.pytorch.deploy.argparse.action._ExtendAction;
import org.bytedeco.pytorch.deploy.argparse.action._HelpAction;
import org.bytedeco.pytorch.deploy.argparse.action._StoreAction;
import org.bytedeco.pytorch.deploy.argparse.action._StoreConstAction;
import org.bytedeco.pytorch.deploy.argparse.action._StoreFalseAction;
import org.bytedeco.pytorch.deploy.argparse.action._StoreTrueAction;
import org.bytedeco.pytorch.deploy.argparse.action._SubParsersAction;
import org.bytedeco.pytorch.deploy.argparse.action._VersionAction;
import org.bytedeco.pytorch.deploy.argparse.formatter.HelpFormatter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Mirrors Python's {@code _ActionsContainer}: the core mixin providing
 * {@code addArgument}, {@code addArgumentGroup}, {@code addMutuallyExclusiveGroup},
 * the action registry, conflict detection, and the optional {@code type=} function
 * registry.
 *
 * <p>ArgumentParser inherits from this class; { argparse.container._ArgumentGroup}
 * also extends it (sharing registries but adding a title).
 */
public abstract class _ActionsContainer {

    protected String description;
    protected String argumentDefault; // marker; null means "leave alone"
    protected String prefixChars = "-";
    protected String conflictHandler = "error";

    /** Two-level registry: registry name → (key → object). */
    protected final Map<String, Map<String, Object>> registries = new LinkedHashMap<>();

    /** Flat list of all actions owned by this container. */
    protected final List<Action> actions = new ArrayList<>();

    /** Action index by option string (short or long). */
    protected final Map<String, Action> optionStringActions = new LinkedHashMap<>();

    /** User-defined argument groups (also includes _positionals and _optionals). */
    protected final List<_ArgumentGroup> actionGroups = new ArrayList<>();
    protected final List<_MutuallyExclusiveGroup> mutuallyExclusiveGroups = new ArrayList<>();

    /** Parser-wide defaults that win over action defaults. */
    protected final Map<String, Object> defaults = new LinkedHashMap<>();

    /** Matcher that flags negative-number-looking tokens. Mirrors Python {@code _negative_number_matcher}. */
    protected final Pattern negativeNumberMatcher = Pattern.compile("-\\.?\\d");

    /** Set (a list with a sentinel element) when negative-number optionals are present. */
    protected final List<Boolean> hasNegativeNumberOptionals = new ArrayList<>();

    protected _ActionsContainer() {
        registerDefaultActions();
    }

    protected _ActionsContainer(String description,
                                String prefixChars,
                                Object argumentDefault,
                                String conflictHandler) {
        this.description = description;
        if (prefixChars != null) this.prefixChars = prefixChars;
        if (argumentDefault != null) this.argumentDefault = argumentDefault.toString();
        if (conflictHandler != null) this.conflictHandler = conflictHandler;
        registerDefaultActions();
        getHandler(); // validate
    }

    private void registerDefaultActions() {
        register("action", null, _StoreAction.class);
        register("action", "store", _StoreAction.class);
        register("action", "store_const", _StoreConstAction.class);
        register("action", "store_true", _StoreTrueAction.class);
        register("action", "store_false", _StoreFalseAction.class);
        register("action", "append", _AppendAction.class);
        register("action", "append_const", _AppendConstAction.class);
        register("action", "count", _CountAction.class);
        register("action", "help", _HelpAction.class);
        register("action", "version", _VersionAction.class);
        register("action", "parsers", _SubParsersAction.class);
        register("action", "extend", _ExtendAction.class);
        register("action", "boolean_optional", BooleanOptionalAction.class);
    }

    // ------------------ registries ------------------

    public void register(String registryName, String value, Object object) {
        registries.computeIfAbsent(registryName, k -> new LinkedHashMap<>()).put(value, object);
    }

    @SuppressWarnings("unchecked")
    public <T> T registryGet(String registryName, Object value, Object defaultValue) {
        Map<String, Object> r = registries.get(registryName);
        if (r == null) return (T) defaultValue;
        Object v = r.get(value == null ? null : value.toString());
        return v == null ? (T) defaultValue : (T) v;
    }

    // ------------------ defaults ------------------

    public void setDefaults(Map<String, Object> kwargs) {
        if (kwargs != null) {
            for (Map.Entry<String, Object> e : kwargs.entrySet()) {
                defaults.put(e.getKey(), e.getValue());
            }
            for (Action a : actions) {
                if (kwargs.containsKey(a.getDest())) a.setDefault(kwargs.get(a.getDest()));
            }
        }
    }

    public void setDefault(String key, Object value) {
        defaults.put(key, value);
        for (Action a : actions) {
            if (Objects.equals(a.getDest(), key)) a.setDefault(value);
        }
    }

    public Object getDefault(String dest) {
        for (Action a : actions) {
            if (Objects.equals(a.getDest(), dest) && a.getDefault() != null
                    && !ArgparseSentinel.isSuppress(a.getDefault())) {
                return a.getDefault();
            }
        }
        return defaults.get(dest);
    }

    // ------------------ add_argument ------------------

    /**
     * Add a positional argument (or multiple option strings).
     * Mirrors Python: {@code parser.add_argument("name")} or {@code parser.add_argument("-v", "--verbose")}.
     */
    public Action addArgument(String... args) {
        return _addArgumentImpl(Arrays.asList(args), new LinkedHashMap<>());
    }

    /**
     * Add an argument with keyword settings.
     * Mirrors Python: {@code parser.add_argument("--foo", default="bar")}.
     */
    public Action addArgument(Map<String, Object> kwargs) {
        return _addArgumentImpl(new ArrayList<>(), new LinkedHashMap<>(kwargs));
    }

    /**
     * Add an argument with option strings and keyword settings.
     * Mirrors Python: {@code parser.add_argument("-v", "--verbose", action="store_true")}.
     */
    public Action addArgument(String[] args, Map<String, Object> kwargs) {
        return _addArgumentImpl(Arrays.asList(args), new LinkedHashMap<>(kwargs));
    }

    /** Abstract: concrete parsers implement this with their argument registration logic. */
    protected abstract Action _addArgumentImpl(List<String> args, Map<String, Object> kwargs);

    public Action _addAction(Action action) {
        if (action == null) throw new IllegalArgumentException("action is null");
        _checkConflict(action);
        actions.add(action);
        action.container = this;
        if (action.getOptionStrings() != null) {
            for (String s : action.getOptionStrings()) {
                optionStringActions.put(s, action);
                if (negativeNumberMatcher.matcher(s).find() && hasNegativeNumberOptionals.isEmpty()) {
                    hasNegativeNumberOptionals.add(Boolean.TRUE);
                }
            }
        }
        return action;
    }

    public void removeAction(Action action) {
        actions.remove(action);
        if (action.getOptionStrings() != null) {
            for (String s : action.getOptionStrings()) optionStringActions.remove(s);
        }
        if (action.container instanceof _ActionsContainer ac) {
            ac.removeAction(action);
            return;
        }
        action.container = null;
    }

    protected void _checkConflict(Action action) {
        if (action.getOptionStrings() == null) return;
        List<Object[]> conflicts = new ArrayList<>();
        for (String opt : action.getOptionStrings()) {
            Action existing = optionStringActions.get(opt);
            if (existing != null) conflicts.add(new Object[] { opt, existing });
        }
        if (!conflicts.isEmpty()) {
            _handleConflict(action, conflicts);
        }
    }

    protected void _handleConflict(Action action, List<Object[]> conflictingActions) {
        if ("error".equals(conflictHandler)) {
            String opts = String.join(", ",
                    conflictingActions.stream().map(a -> (String) a[0]).toArray(String[]::new));
            String msg = "conflicting option string" + (conflictingActions.size() > 1 ? "s" : "")
                    + ": " + opts;
            throw new ArgumentError(action, msg);
        } else if ("resolve".equals(conflictHandler)) {
            for (Object[] pair : conflictingActions) {
                String opt = (String) pair[0];
                Action existing = optionStringActions.get(opt);
                if (existing != null) {
                    existing.getOptionStrings().remove(opt);
                    optionStringActions.remove(opt);
                    if (existing.getOptionStrings().isEmpty()
                            && existing.container instanceof _ActionsContainer ac) {
                        ac.removeAction(existing);
                    }
                }
            }
        }
    }

    /** Throw on unknown conflict handler names — mirrors Python {@code _get_handler}. */
    public java.util.function.BiConsumer<Action, List<Object[]>> getHandler() {
        switch (conflictHandler) {
            case "error": return this::_handleConflict;
            case "resolve": return this::_handleConflict;
            default: throw new IllegalArgumentException(
                    "invalid conflict_resolution value: '" + conflictHandler + "'");
        }
    }

    public void checkHelp(Action action) {
        if (action.getHelp() != null && this instanceof ArgumentParser p) {
            try {
                HelpFormatter f = p.getFormatter();
                f._expandHelp(action);
            } catch (RuntimeException e) {
                throw new IllegalArgumentException("badly formed help string", e);
            }
        }
    }

    /** Group management. */
    public abstract _ArgumentGroup addArgumentGroup(String title, String description);

    public abstract _MutuallyExclusiveGroup addMutuallyExclusiveGroup(boolean required);

    public abstract Object addSubparsers(Map<String, Object> kwargs);

    /** Returns true if a token could be a valid prefix-char option. */
    public boolean startsWithPrefix(String s) {
        if (s == null || s.isEmpty()) return false;
        return prefixChars.indexOf(s.charAt(0)) >= 0;
    }

    public Map<String, Action> getOptionStringActions() { return optionStringActions; }
    public List<Action> getActions() { return actions; }
    public List<_ArgumentGroup> getActionGroups() { return actionGroups; }
    public List<_MutuallyExclusiveGroup> getMutuallyExclusiveGroups() { return mutuallyExclusiveGroups; }
    public String getPrefixChars() { return prefixChars; }
    public Map<String, Object> getDefaults() { return defaults; }
    public List<Boolean> getHasNegativeNumberOptionals() { return hasNegativeNumberOptionals; }
    public String getDescription() { return description; }
    public void setDescription(String d) { this.description = d; }
    public String getArgumentDefault() { return argumentDefault; }
    public void setArgumentDefault(Object v) { this.argumentDefault = v == null ? null : v.toString(); }
    public String getConflictHandler() { return conflictHandler; }
    public void setConflictHandler(String h) { this.conflictHandler = h; }
    public void setPrefixChars(String p) { this.prefixChars = p; }

    /**
     * Apply a default value to an action only when the key matches.
     * Mirrors Python's logic that "if no default was supplied, use the parser-level default".
     */
    protected void applyParserDefault(Map<String, Object> kwargs, String dest) {
        if (!kwargs.containsKey("default")) {
            if (defaults.containsKey(dest)) {
                kwargs.put("default", defaults.get(dest));
            } else if (argumentDefault != null) {
                kwargs.put("default", argumentDefault);
            }
        }
    }

    /** Quick helper to identify the registered action classes. */
    public static Class<? extends Action> resolveActionClass(_ActionsContainer container, Object actionName) {
        Object cls = container.registryGet("action", actionName, actionName);
        if (cls instanceof Class<?>) {
            Class<?> c = (Class<?>) cls;
            if (Action.class.isAssignableFrom(c)) {
                @SuppressWarnings("unchecked")
                Class<? extends Action> ac = (Class<? extends Action>) c;
                return ac;
            }
        }
        throw new IllegalArgumentException("unknown action " + actionName);
    }

    /** Special-cases for FileType — must be passed as instance, not class. */
    protected void validateType(Action action) {
        Object type = action.getType();
        if (type == null) return;
        if (type instanceof Class<?>) {
            Class<?> c = (Class<?>) type;
            if (FileType.class.equals(c)) {
                throw new IllegalArgumentException("FileType class object must be passed as instance");
            }
            try {
                c.getMethod("call", Object.class);
            } catch (NoSuchMethodException e) {
                throw new IllegalArgumentException("'" + c.getName() + "' is not callable");
            }
        }
    }

    /** Utility: get an nargs-aware validation pattern. */
    public static Pattern nargsPattern(Action action) {
        Object n = action.getNargs();
        boolean option = action.getOptionStrings() != null && !action.getOptionStrings().isEmpty();
        if (n == null) {
            return option ? HelpFormatter.NARGS_DEFAULT_OPT : HelpFormatter.NARGS_DEFAULT_POS;
        } else if (ArgparseConstants.OPTIONAL.equals(n)) {
            return option ? HelpFormatter.NARGS_OPTIONAL_OPT : HelpFormatter.NARGS_OPTIONAL_POS;
        } else if (ArgparseConstants.ZERO_OR_MORE.equals(n)) {
            return option ? HelpFormatter.NARGS_ZERO_OPT : HelpFormatter.NARGS_ZERO_POS;
        } else if (ArgparseConstants.ONE_OR_MORE.equals(n)) {
            return option ? HelpFormatter.NARGS_ONE_OPT : HelpFormatter.NARGS_ONE_POS;
        } else if (ArgparseConstants.REMAINDER.equals(n)) {
            return option ? HelpFormatter.NARGS_REMAINDER_OPT : HelpFormatter.NARGS_REMAINDER_POS;
        } else if (ArgparseConstants.PARSER.equals(n)) {
            return option ? HelpFormatter.NARGS_PARSER_OPT : HelpFormatter.NARGS_PARSER_POS;
        } else if (ArgparseConstants.SUPPRESS.equals(n)) {
            return option ? HelpFormatter.NARGS_SUPPRESS_OPT : HelpFormatter.NARGS_SUPPRESS_POS;
        } else if (n instanceof Integer) {
            // build dynamic
            int count = (Integer) n;
            String body = option ? "[AO]{" + count + "}" : "(?:-*A){" + count + "}-*";
            return Pattern.compile("(" + body + ")");
        } else {
            throw new IllegalArgumentException("invalid nargs value: " + n);
        }
    }

    public ArgumentError newArgumentError(Action action, String message) {
        return new ArgumentError(action, message);
    }

    // helper marker
    public static final class ArgparseSentinel {
        private ArgparseSentinel() {}
        public static boolean isSuppress(Object o) {
            return o != null && o.equals(ArgparseConstants.SUPPRESS);
        }
    }
}