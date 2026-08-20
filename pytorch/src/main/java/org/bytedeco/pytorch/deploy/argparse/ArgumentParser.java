/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse;

import org.bytedeco.pytorch.deploy.argparse.action._StoreAction;
import org.bytedeco.pytorch.deploy.argparse.action._SubParsersAction;
import org.bytedeco.pytorch.deploy.argparse.container._ActionsContainer;
import org.bytedeco.pytorch.deploy.argparse.container._ArgumentGroup;
import org.bytedeco.pytorch.deploy.argparse.container._MutuallyExclusiveGroup;
import org.bytedeco.pytorch.deploy.argparse.formatter.HelpFormatter;
import org.bytedeco.pytorch.deploy.argparse.internal.Identity;
import org.bytedeco.pytorch.deploy.argparse.internal.ProgName;
import org.bytedeco.pytorch.deploy.argparse.internal.SystemArgsBridge;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.InstantiationException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * The main entry point. Mirrors Python's {@code argparse.ArgumentParser}.
 *
 * <p>Construct with any subset of the named parameters, then call
 * {#addArgument(Object...)} to register actions and {@link #parseArgs(List)}
 * to do the parsing.
 *
 * <p>By default {@code error()} calls {@link System#exit(int)}; for
 * embeddable / library use, set {@code exitOnError=false} and the parser
 * will throw {@link ArgumentError} instead. Tests can also catch
 * {@link ExitTrappedException} when {@code print_help()} / {@code exit()}
 * would normally terminate.
 */
public class ArgumentParser extends _ActionsContainer {

    // -------- public fields --------
    private String prog;
    private String usage;
    private String epilog;
    private Class<? extends HelpFormatter> formatterClass = HelpFormatter.class;
    private String fromfilePrefixChars;
    private boolean addHelp = true;
    private boolean allowAbbrev = true;
    private boolean exitOnError = true;
    private boolean suggestOnError = false;
    private boolean color = false;
    private String version;

    // -------- internal state --------
    private _ArgumentGroup positionals;
    private _ArgumentGroup optionals;
    private Object subparsers;
    private HelpFormatter cachedFormatter;

    // -------- constructor --------
    public ArgumentParser() {
        this(null, null, null, null, null, null, null, null, null, "error", true, true, true, false, false, null);
    }

    public ArgumentParser(String prog) {
        this(prog, null, null, null, null, null, null, null, null, "error", true, true, true, false, false, null);
    }

    public ArgumentParser(String prog,
                          String usage,
                          String description,
                          String epilog,
                          List<ArgumentParser> parents,
                          Class<? extends HelpFormatter> formatterClass,
                          String prefixChars,
                          String fromfilePrefixChars,
                          Object argumentDefault,
                          String conflictHandler,
                          boolean addHelp,
                          boolean allowAbbrev,
                          boolean exitOnError,
                          boolean suggestOnError,
                          boolean color,
                          String version) {
        super(description, prefixChars, argumentDefault, conflictHandler);
        this.prog = ProgName.resolve(prog);
        this.usage = usage;
        this.epilog = epilog;
        if (formatterClass != null) this.formatterClass = formatterClass;
        if (fromfilePrefixChars != null) this.fromfilePrefixChars = fromfilePrefixChars;
        this.addHelp = addHelp;
        this.allowAbbrev = allowAbbrev;
        this.exitOnError = exitOnError;
        this.suggestOnError = suggestOnError;
        this.color = color;
        this.version = version;
        register("type", null, Identity.class);

        this.positionals = addArgumentGroup("positional arguments", null);
        this.optionals = addArgumentGroup("options", null);
        this.subparsers = null;

        String firstPrefix = prefixChars == null || prefixChars.isEmpty() ? "-" : prefixChars.substring(0, 1);
        if (addHelp) {
            Map<String, Object> helpKw = new LinkedHashMap<>();
            helpKw.put("action", "help");
            helpKw.put("default", ArgparseConstants.SUPPRESS);
            helpKw.put("help", "show this help message and exit");
            addArgument(new String[]{firstPrefix + "h", firstPrefix + firstPrefix + "help"}, helpKw);
        }

        if (parents != null) {
            for (ArgumentParser p : parents) {
                if (!(p instanceof ArgumentParser)) {
                    throw new IllegalArgumentException("parents must be a list of ArgumentParser");
                }
                _addContainerActions(p);
                this.defaults.putAll(p.defaults);
            }
        }
    }

    public static Builder builder() { return new Builder(); }
    public static Builder builder(String prog) { return new Builder().prog(prog); }

    public static class Builder {
        private String prog;
        private String usage;
        private String description;
        private String epilog;
        private List<ArgumentParser> parents;
        private Class<? extends HelpFormatter> formatterClass;
        private String prefixChars = "-";
        private String fromfilePrefixChars;
        private Object argumentDefault;
        private String conflictHandler = "error";
        private boolean addHelp = true;
        private boolean allowAbbrev = true;
        private boolean exitOnError = true;
        private boolean suggestOnError = false;
        private boolean color = false;

        public Builder prog(String v) { this.prog = v; return this; }
        public Builder usage(String v) { this.usage = v; return this; }
        public Builder description(String v) { this.description = v; return this; }
        public Builder epilog(String v) { this.epilog = v; return this; }
        public Builder parents(List<ArgumentParser> v) { this.parents = v; return this; }
        public Builder formatterClass(Class<? extends HelpFormatter> v) { this.formatterClass = v; return this; }
        public Builder prefixChars(String v) { this.prefixChars = v; return this; }
        public Builder fromfilePrefixChars(String v) { this.fromfilePrefixChars = v; return this; }
        public Builder argumentDefault(Object v) { this.argumentDefault = v; return this; }
        public Builder conflictHandler(String v) { this.conflictHandler = v; return this; }
        public Builder addHelp(boolean v) { this.addHelp = v; return this; }
        public Builder allowAbbrev(boolean v) { this.allowAbbrev = v; return this; }
        public Builder exitOnError(boolean v) { this.exitOnError = v; return this; }
        public Builder suggestOnError(boolean v) { this.suggestOnError = v; return this; }
        public Builder color(boolean v) { this.color = v; return this; }

        public ArgumentParser build() {
            return new ArgumentParser(prog, usage, description, epilog, parents, formatterClass,
                    prefixChars, fromfilePrefixChars, argumentDefault, conflictHandler,
                    addHelp, allowAbbrev, exitOnError, suggestOnError, color, null);
        }
    }

    // ====================================================
    //  add_argument machinery
    // ====================================================

    /** Add a positional or option argument. Mirrors Python: {@code parser.add_argument("name")} or {@code parser.add_argument("-v", "--verbose")}. */
    @Override
    public Action addArgument(String... args) {
        return _addArgumentImpl(Arrays.asList(args), new LinkedHashMap<>());
    }

    /** Add a help/action argument. Mirrors Python: {@code parser.add_argument(action="help")}. */
    @Override
    public Action addArgument(Map<String, Object> kwargs) {
        return _addArgumentImpl(new ArrayList<>(), new LinkedHashMap<>(kwargs));
    }

    /** Add an argument with both option strings and keyword settings. Mirrors Python: {@code parser.add_argument("--foo", action="store_true")}. */
    @Override
    public Action addArgument(String[] args, Map<String, Object> kwargs) {
        return _addArgumentImpl(Arrays.asList(args), new LinkedHashMap<>(kwargs));
    }

    /** Convenience: add a single-string argument with kwargs. Mirrors Python: {@code parser.add_argument("--foo", action="store_true")}. */
    public Action addArgument(String arg, Map<String, Object> kwargs) {
        return _addArgumentImpl(Arrays.asList(arg), new LinkedHashMap<>(kwargs));
    }

    /** Convenience: add two-string arguments with kwargs. */
    public Action addArgument(String arg1, String arg2, Map<String, Object> kwargs) {
        return _addArgumentImpl(Arrays.asList(arg1, arg2), new LinkedHashMap<>(kwargs));
    }

    /** Overload accepting {@code List<String>} for programmatic use. */
    public Action addArgument(List<String> args) {
        return _addArgumentImpl(new ArrayList<>(args), new LinkedHashMap<>());
    }

    @Override
    protected Action _addArgumentImpl(List<String> args, Map<String, Object> kwargs) {
        Map<String, Object> k = new LinkedHashMap<>(kwargs == null ? new LinkedHashMap<>() : kwargs);
        List<String> optionStrings = new ArrayList<>();
        List<String> purePositional = new ArrayList<>();
        for (String a : args) {
            if (!a.isEmpty() && prefixChars.indexOf(a.charAt(0)) >= 0) optionStrings.add(a);
            else purePositional.add(a);
        }
        if (args.isEmpty() || (args.size() == 1 && purePositional.size() == 1)) {
            if (!purePositional.isEmpty() && k.containsKey("dest")) {
                throw new IllegalArgumentException("dest supplied twice for positional argument, did you mean metavar?");
            }
            k = _getPositionalKwargs(purePositional.isEmpty() ? null : purePositional.get(0), k);
        } else {
            k = _getOptionalKwargs(args, k);
        }
        String dest = (String) k.get("dest");
        if (!k.containsKey("default")) {
            if (defaults.containsKey(dest)) k.put("default", defaults.get(dest));
            else if (argumentDefault != null) k.put("default", argumentDefault);
        }

        Object actionName = k.get("action");
        Class<? extends Action> actionClass = resolveActionClass(this, actionName);
        if (actionClass == null) throw new IllegalArgumentException("unknown action " + actionName);

        Action action;
        try {
            Constructor<? extends Action> ctor = actionClass.getDeclaredConstructor();
            action = ctor.newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("action has no default constructor: " + actionClass, e);
        }
        for (Map.Entry<String, Object> e : k.entrySet()) {
            applyKwargsToAction(action, e.getKey(), e.getValue());
        }
        if (k.containsKey("nargs")) action.setNargs(k.get("nargs"));

        if ((action.getOptionStrings() == null || action.getOptionStrings().isEmpty())
                && Integer.valueOf(0).equals(action.getNargs())) {
            throw new IllegalArgumentException("action '" + actionName + "' is not valid for positional arguments");
        }
        validateTypeCallable(action);
        checkHelp(action);

        return _addAction(action);
    }

    private static void applyKwargsToAction(Action action, String key, Object value) {
        switch (key) {
            case "option_strings":
                if (value instanceof List<?> opts) {
                    action.setOptionStrings(opts.isEmpty() ? null
                            : new ArrayList<>((List<String>) (List<?>) opts));
                } else {
                    action.setOptionStrings(null);
                }
                break;
            case "dest": action.setRawDest(value); break;
            case "nargs": action.setNargs(value); break;
            case "const": action.setConst(value); break;
            case "default": action.setDefault(value); break;
            case "type": action.setType(value); break;
            case "choices": action.setChoices(value); break;
            case "required": action.setRequired((Boolean) value); break;
            case "help": action.setHelp(value == null ? null : value.toString()); break;
            case "metavar": action.setMetavar(value); break;
            case "deprecated": action.setDeprecated((Boolean) value); break;
            default: break;
        }
    }

    private void validateTypeCallable(Action action) {
        Object t = action.getType();
        if (t == null) return;
        if (t instanceof Class<?>) {
            Class<?> c = (Class<?>) t;
            if (FileType.class.equals(c)) {
                throw new IllegalArgumentException("FileType class object must be passed as instance");
            }
            // Class<?> types are handled specially in getValue() — no callable check needed.
            return;
        }
    }

    private Map<String, Object> _getPositionalKwargs(String dest, Map<String, Object> kwargs) {
        if (kwargs.containsKey("required")) {
            throw new IllegalArgumentException("'required' is an invalid argument for positionals");
        }
        Object nargs = kwargs.get("nargs");
        if (Integer.valueOf(0).equals(nargs)) {
            throw new IllegalArgumentException("nargs for positionals must be != 0");
        }
        if (nargs != null && !Arrays.asList(ArgparseConstants.OPTIONAL, ArgparseConstants.ZERO_OR_MORE,
                ArgparseConstants.REMAINDER, ArgparseConstants.SUPPRESS).contains(nargs)) {
            kwargs.put("required", true);
        }
        Map<String, Object> out = new LinkedHashMap<>(kwargs);
        out.put("dest", dest);
        out.put("option_strings", null);
        return out;
    }

    private Map<String, Object> _getOptionalKwargs(List<String> args, Map<String, Object> kwargs) {
        List<String> optionStrings = new ArrayList<>();
        List<String> longOptionStrings = new ArrayList<>();
        for (String opt : args) {
            if (!startsWithPrefix(opt)) {
                throw new IllegalArgumentException("invalid option string '" + opt
                        + "': must start with a character '" + prefixChars + "'");
            }
            optionStrings.add(opt);
            if (opt.length() > 1 && prefixChars.indexOf(opt.charAt(1)) >= 0) longOptionStrings.add(opt);
        }
        String dest = (String) kwargs.get("dest");
        if (dest == null) {
            String src;
            if (!longOptionStrings.isEmpty()) src = longOptionStrings.get(0);
            else src = optionStrings.get(0);
            dest = stripPrefixChars(src);
            if (dest.isEmpty()) {
                throw new IllegalArgumentException("dest= is required for options like '" + src + "'");
            }
            dest = dest.replace('-', '_');
        }
        Map<String, Object> out = new LinkedHashMap<>(kwargs);
        out.put("dest", dest);
        out.put("option_strings", optionStrings);
        return out;
    }

    private String stripPrefixChars(String s) {
        int i = 0;
        while (i < s.length() && prefixChars.indexOf(s.charAt(i)) >= 0) i++;
        return s.substring(i);
    }

    // ====================================================
    //  groups
    // ====================================================

    @Override
    public _ArgumentGroup addArgumentGroup(String title, String description) {
        _ArgumentGroup g = new _ArgumentGroup(this, title, description,
                prefixChars, argumentDefault, conflictHandler);
        actionGroups.add(g);
        return g;
    }

    public _ArgumentGroup addArgumentGroup(String title) {
        return addArgumentGroup(title, null);
    }

    @Override
    public _MutuallyExclusiveGroup addMutuallyExclusiveGroup(boolean required) {
        _MutuallyExclusiveGroup g = new _MutuallyExclusiveGroup(this, required);
        mutuallyExclusiveGroups.add(g);
        return g;
    }

    public _MutuallyExclusiveGroup addMutuallyExclusiveGroup() {
        return addMutuallyExclusiveGroup(false);
    }

    @Override
    public Object addSubparsers(Map<String, Object> kwargs) {
        if (this.subparsers != null) {
            throw new IllegalStateException("cannot have multiple subparser arguments");
        }
        Map<String, Object> k = new LinkedHashMap<>(kwargs == null ? new LinkedHashMap<>() : kwargs);
        k.putIfAbsent("parserClass", this.getClass());

        _ArgumentGroup group;
        if (k.containsKey("title") || k.containsKey("description")) {
            String title = (String) k.getOrDefault("title", "subcommands");
            String desc = (String) k.get("description");
            group = addArgumentGroup(title, desc);
            this.subparsers = group;
        } else {
            group = this.positionals;
            this.subparsers = group;
        }
        if (!k.containsKey("prog")) {
            StringBuilder sub = new StringBuilder();
            sub.append(prog);
            for (Action a : _getOptionalActions()) {
                if (a.getOptionStrings() == null || a.getOptionStrings().isEmpty()) continue;
                if (a.getOptionStrings().get(0).length() > 2
                        && prefixChars.indexOf(a.getOptionStrings().get(0).charAt(1)) >= 0) {
                    sub.append(" ").append(a.getOptionStrings().get(0));
                }
            }
            k.put("prog", sub.toString().trim());
        }
        Class<? extends Action> cls = resolveActionClass(this, k.getOrDefault("action", "parsers"));
        try {
            Constructor<? extends Action> ctor = cls.getDeclaredConstructor();
            Action action = ctor.newInstance();
            applyKwargsToAction(action, "dest", k.getOrDefault("dest", ArgparseConstants.SUPPRESS));
            action.setNargs(ArgparseConstants.PARSER);
            action.setRequired(Boolean.TRUE.equals(k.get("required")));
            applyKwargsToAction(action, "help", k.get("help"));
            applyKwargsToAction(action, "metavar", k.get("metavar"));
            action.setRawOptions(k);
            @SuppressWarnings("unchecked")
            List<String> optStrings = (List<String>) k.getOrDefault("option_strings", new ArrayList<>());
            action.setOptionStrings(optStrings);
            checkHelp(action);
            group._addAction(action);
            ((_SubParsersAction) action).setColor(color);
            return action;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("could not construct subparsers action", e);
        }
    }

    public _SubParsersAction addSubparsers() {
        return (_SubParsersAction) addSubparsers(new LinkedHashMap<>());
    }

    public _SubParsersAction addSubparsers(String title, String description) {
        Map<String, Object> kw = new LinkedHashMap<>();
        kw.put("title", title);
        kw.put("description", description);
        return (_SubParsersAction) addSubparsers(kw);
    }

    @Override
    public Action _addAction(Action action) {
        if (action.getOptionStrings() != null && !action.getOptionStrings().isEmpty()) {
            return optionals._addAction(action);
        } else {
            return positionals._addAction(action);
        }
    }

    public List<Action> _getOptionalActions() {
        List<Action> out = new ArrayList<>();
        for (Action a : actions) if (a.getOptionStrings() != null && !a.getOptionStrings().isEmpty()) out.add(a);
        return out;
    }

    public List<Action> _getPositionalActions() {
        List<Action> out = new ArrayList<>();
        for (Action a : actions) if (a.getOptionStrings() == null || a.getOptionStrings().isEmpty()) out.add(a);
        return out;
    }

    protected void _addContainerActions(_ActionsContainer container) {
        Map<String, _ArgumentGroup> titleGroupMap = new LinkedHashMap<>();
        for (_ArgumentGroup g : container.getActionGroups()) {
            if (titleGroupMap.containsKey(g.getTitle())) {
                throw new IllegalStateException("cannot merge actions - two groups are named '" + g.getTitle() + "'");
            }
            titleGroupMap.put(g.getTitle(), g);
        }
        Map<Action, _ArgumentGroup> groupMap = new LinkedHashMap<>();
        for (_ArgumentGroup g : container.getActionGroups()) {
            String title = g.getTitle();
            _ArgumentGroup host = titleGroupMap.computeIfAbsent(title, t ->
                    addArgumentGroup(g.getTitle(), g.getGroupDescription()));
            for (Action a : g.getGroupActions()) groupMap.put(a, host);
        }
        for (_MutuallyExclusiveGroup g : container.getMutuallyExclusiveGroups()) {
            _MutuallyExclusiveGroup mg;
            if (g.parent == container || g.parent == this) {
                mg = addMutuallyExclusiveGroup(g.required);
            } else if (g.parent instanceof _ArgumentGroup parentGroup) {
                _ArgumentGroup targetGroup = titleGroupMap.get(parentGroup.getTitle());
                mg = targetGroup.addMutuallyExclusiveGroup(g.required);
            } else {
                mg = addMutuallyExclusiveGroup(g.required);
            }
            for (Action a : g.groupActions) groupMap.put(a, mg);
        }
        for (Action a : container.getActions()) {
            _ArgumentGroup host = groupMap.getOrDefault(a, this.positionals);
            host._addAction(a);
        }
    }

    // ====================================================
    //  parse_args / parse_known_args
    // ====================================================

    public Namespace parseArgs() { return parseArgs((List<String>) null); }

    public Namespace parseArgs(List<String> args) {
        return _parseKnownArgs2(args == null ? null : new ArrayList<>(args), null, false).namespace;
    }

    /** Legacy alias for {@code parseArgs(List<String>)}. */
    public Namespace parseArgs(String[] args) {
        return parseArgs(args == null ? null : Arrays.asList(args));
    }

    public ParseResult parseKnownArgs() { return parseKnownArgs(null, null); }

    public ParseResult parseKnownArgs(List<String> args) { return parseKnownArgs(args, null); }

    public ParseResult parseKnownArgs(List<String> args, Namespace namespace) {
        return _parseKnownArgs2(args == null ? null : new ArrayList<>(args), namespace, false);
    }

    public static class ParseResult {
        public final Namespace namespace;
        public final List<String> extras;
        public ParseResult(Namespace ns, List<String> extras) {
            this.namespace = ns;
            this.extras = extras == null ? new ArrayList<>() : extras;
        }
    }

    protected ParseResult _parseKnownArgs2(List<String> args, Namespace namespace, boolean intermixed) {
        if (args == null) args = new ArrayList<>(SystemArgsBridge.getSystemArgs());
        if (namespace == null) namespace = new Namespace();
        for (Action action : actions) {
            if (!ArgparseConstants.SUPPRESS.equals(action.getRawDest())
                    && !namespace.has(action.getDest())) {
                if (!ArgparseConstants.SUPPRESS.equals(action.getDefault())) {
                    namespace.set(action.getDest(), action.getDefault());
                }
            }
        }
        for (Map.Entry<String, Object> e : defaults.entrySet()) {
            if (!namespace.has(e.getKey())) namespace.set(e.getKey(), e.getValue());
        }
        List<String> extras;
        try {
            extras = _parseKnownArgs(args, namespace, intermixed);
        } catch (ArgumentError err) {
            if (exitOnError) error(err.getMessage());
            else throw err;
            return new ParseResult(namespace, new ArrayList<>());
        }
        if (namespace.has(ArgparseConstants.UNRECOGNIZED_ARGS_ATTR)) {
            @SuppressWarnings("unchecked")
            List<String> u = (List<String>) namespace.get(ArgparseConstants.UNRECOGNIZED_ARGS_ATTR);
            if (u != null) extras.addAll(u);
        }
        return new ParseResult(namespace, extras);
    }

    protected List<String> _parseKnownArgs(List<String> argStrings, Namespace namespace, boolean intermixed) {
        if (fromfilePrefixChars != null) argStrings = _readArgsFromFiles(argStrings);
        Map<Action, List<Action>> actionConflicts = new LinkedHashMap<>();
        for (_MutuallyExclusiveGroup g : mutuallyExclusiveGroups) {
            List<Action> ga = g.groupActions;
            for (int i = 0; i < ga.size(); i++) {
                List<Action> conflicts = actionConflicts.computeIfAbsent(ga.get(i), k -> new ArrayList<>());
                conflicts.addAll(ga.subList(0, i));
                conflicts.addAll(ga.subList(i + 1, ga.size()));
            }
        }
        Map<Integer, List<Object[]>> optionStringIndices = new LinkedHashMap<>();
        StringBuilder argStringPattern = new StringBuilder();
        java.util.Iterator<String> iter = argStrings.iterator();
        int i = 0;
        while (iter.hasNext()) {
            String argString = iter.next();
            if ("--".equals(argString)) {
                argStringPattern.append('-');
                while (iter.hasNext()) {
                    iter.next();
                    argStringPattern.append('A');
                    i++;
                }
                break;
            }
            List<Object[]> optionTuples = _parseOptional(argString);
            if (optionTuples == null) argStringPattern.append('A');
            else {
                optionStringIndices.put(i, optionTuples);
                argStringPattern.append('O');
            }
            i++;
        }

        java.util.Set<Action> seenActions = new java.util.HashSet<>();
        java.util.Set<Action> seenNonDefaultActions = new java.util.HashSet<>();
        java.util.Set<String> warned = new java.util.HashSet<>();
        List<String> extras = new ArrayList<>();

        // Loop state stored in a small holder so inner-method reads see mutations.
        int[] maxOptionHolder = { optionStringIndices.isEmpty() ? -1
                : optionStringIndices.keySet().stream().max(Integer::compare).get() };
        String[] argStringsPatternBox = { argStringPattern.toString() };
        List<String>[] argStringsBox = new List[]{ argStrings };

        int startIndex = 0;
        while (startIndex <= maxOptionHolder[0]) {
            int nextOptionStringIndex = startIndex;
            while (nextOptionStringIndex <= maxOptionHolder[0]) {
                if (optionStringIndices.containsKey(nextOptionStringIndex)) break;
                nextOptionStringIndex++;
            }
            if (!intermixed && startIndex != nextOptionStringIndex) {
                int positionalsEndIndex = consumePositionals(startIndex, argStringsBox[0],
                        argStringsPatternBox, extras, warned, seenActions, namespace);
                if (positionalsEndIndex > startIndex) {
                    startIndex = positionalsEndIndex;
                    continue;
                }
                startIndex = positionalsEndIndex;
            }
            if (!optionStringIndices.containsKey(startIndex)) {
                for (int k = startIndex; k < nextOptionStringIndex; k++) {
                    extras.add(argStringsBox[0].get(k));
                }
                startIndex = nextOptionStringIndex;
            }
            startIndex = consumeOptional(startIndex, argStringsBox[0], argStringsPatternBox,
                    optionStringIndices, extras, warned, seenActions, namespace);
        }
        if (!intermixed) {
            int stopIndex = consumePositionals(startIndex, argStringsBox[0], argStringsPatternBox,
                    extras, warned, seenActions, namespace);
            extras.addAll(argStringsBox[0].subList(stopIndex, argStringsBox[0].size()));
        } else {
            extras.addAll(argStringsBox[0].subList(startIndex, argStringsBox[0].size()));
            List<String> newExtras = new ArrayList<>();
            for (String ex : extras) newExtras.add(ex);
            argStringsBox[0] = newExtras;
            argStringsPatternBox[0] = argStringsPatternBox[0].replace("O", "");
            consumePositionals(0, argStringsBox[0], argStringsPatternBox,
                    extras, warned, seenActions, namespace);
        }

        List<String> requiredActions = new ArrayList<>();
        for (Action action : actions) {
            if (!seenActions.contains(action)) {
                if (action.isRequired()) requiredActions.add(ArgumentNameExtractor.nameOf(action));
                else if (action.getDefault() instanceof String
                        && namespace.has(action.getDest())
                        && Objects.equals(action.getDefault(), namespace.get(action.getDest()))) {
                    namespace.set(action.getDest(), getValue(action, (String) action.getDefault()));
                }
            }
        }
        if (!requiredActions.isEmpty()) {
            throw new ArgumentError("the following arguments are required: " + String.join(", ", requiredActions));
        }
        for (_MutuallyExclusiveGroup g : mutuallyExclusiveGroups) {
            if (g.required) {
                boolean present = false;
                for (Action a : g.groupActions) if (seenNonDefaultActions.contains(a)) { present = true; break; }
                if (!present) {
                    StringBuilder names = new StringBuilder();
                    for (Action a : g.groupActions) {
                        if (!ArgparseConstants.SUPPRESS.equals(a.getHelp())) {
                            if (names.length() > 0) names.append(' ');
                            names.append(ArgumentNameExtractor.nameOf(a));
                        }
                    }
                    throw new ArgumentError("one of the arguments " + names + " is required");
                }
            }
        }
        return extras;
    }

    private int consumeOptional(int startIndex, List<String> argStrings, String[] argStringsPatternBox,
                                Map<Integer, List<Object[]>> optionStringIndices, List<String> extras,
                                java.util.Set<String> warned, java.util.Set<Action> seenActions,
                                Namespace namespace) {
        List<Object[]> optionTuples = optionStringIndices.get(startIndex);
        if (optionTuples == null) return startIndex + 1;
        if (optionTuples.size() > 1) {
            String matches = String.join(", ",
                    optionTuples.stream().map(t -> (String) t[1]).toArray(String[]::new));
            String msg = "ambiguous option: '" + argStrings.get(startIndex) + "' could match " + matches;
            throw new ArgumentError(msg);
        }
        Object[] tup = optionTuples.get(0);
        Action action = (Action) tup[0];
        String optionString = (String) tup[1];
        String sep = (String) tup[2];
        String explicitArg = (String) tup[3];
        List<Object[]> actionTuples = new ArrayList<>();
        int stop = startIndex + 1;
        while (true) {
            if (action == null) {
                extras.add(argStrings.get(startIndex));
                return startIndex + 1;
            }
            if (explicitArg != null) {
                int argCount = matchArgument(action, "A");
                if (argCount == 0 && optionString.length() > 1
                        && prefixChars.indexOf(optionString.charAt(1)) < 0
                        && !explicitArg.isEmpty()) {
                    if (sep != null || prefixChars.indexOf(explicitArg.charAt(0)) >= 0) {
                        throw new ArgumentError(action, "ignored explicit argument '" + explicitArg + "'");
                    }
                    actionTuples.add(new Object[] { action, new ArrayList<>(), optionString });
                    char firstChar = optionString.charAt(0);
                    optionString = String.valueOf(firstChar) + explicitArg.charAt(0);
                    Action next = optionStringActions.get(optionString);
                    if (next != null) {
                        action = next;
                        explicitArg = explicitArg.substring(1);
                        if (explicitArg.isEmpty()) sep = explicitArg = null;
                        else if (explicitArg.charAt(0) == '=') { sep = "="; explicitArg = explicitArg.substring(1); }
                        else sep = "";
                    } else {
                        extras.add(String.valueOf(firstChar) + explicitArg);
                        stop = startIndex + 1;
                        break;
                    }
                } else if (argCount == 1) {
                    stop = startIndex + 1;
                    List<String> args = new ArrayList<>();
                    args.add(explicitArg);
                    actionTuples.add(new Object[] { action, args, optionString });
                    break;
                } else {
                    throw new ArgumentError(action, "ignored explicit argument '" + explicitArg + "'");
                }
            } else {
                int sStart = startIndex + 1;
                String selectedPatterns = argStringsPatternBox[0].substring(sStart);
                int argCount = matchArgument(action, selectedPatterns);
                stop = sStart + argCount;
                List<String> args = new ArrayList<>(argStrings.subList(sStart, stop));
                actionTuples.add(new Object[] { action, args, optionString });
                break;
            }
        }
        for (Object[] at : actionTuples) {
            Action a = (Action) at[0];
            @SuppressWarnings("unchecked")
            List<String> args = (List<String>) at[1];
            String optStr = (String) at[2];
            if (a.isDeprecated() && !warned.contains(optStr)) {
                warning("option '" + optStr + "' is deprecated");
                warned.add(optStr);
            }
            seenActions.add(a);
            Object values = getValues(a, args, optStr);
            if (!ArgparseConstants.SUPPRESS.equals(values)) {
                a.call(this, namespace, values, optStr);
            }
        }
        return stop;
    }

    private int consumePositionals(int startIndex, List<String> argStrings, String[] argStringsPatternBox,
                                   List<String> extras, java.util.Set<String> warned,
                                   java.util.Set<Action> seenActions, Namespace namespace) {
        List<Action> positionals = _getPositionalActions();
        String selectedPattern = argStringsPatternBox[0].substring(startIndex);
        List<Integer> argCounts = matchArgumentsPartial(positionals, selectedPattern);
        int cursor = startIndex;
        for (int j = 0; j < argCounts.size(); j++) {
            Action action = positionals.get(j);
            int argCount = argCounts.get(j);
            List<String> args = new ArrayList<>(argStrings.subList(cursor, cursor + argCount));
            if (ArgparseConstants.PARSER.equals(action.getNargs())) {
                if (argStringsPatternBox[0].charAt(cursor) == '-') {
                    args.remove("--");
                }
            } else if (!ArgparseConstants.REMAINDER.equals(action.getNargs())) {
                if (argStringsPatternBox[0].indexOf("-", cursor) >= 0
                        && argStringsPatternBox[0].indexOf("-", cursor) < cursor + argCount) {
                    args.remove("--");
                }
            }
            cursor += argCount;
            if (!args.isEmpty() && action.isDeprecated() && !warned.contains(action.getDest())) {
                warning("argument '" + action.getDest() + "' is deprecated");
                warned.add(action.getDest());
            }
            Object values = getValues(action, args, null);
            if (!ArgparseConstants.SUPPRESS.equals(values)) {
                action.call(this, namespace, values, null);
            }
            seenActions.add(action);
        }
        positionals.subList(0, argCounts.size()).clear();
        return cursor;
    }

    /** Helper for subparsers / sub-action callback that calls without an optionString. */
    public Object getValues(Action action, List<String> argStrings, String optionString) {
        if (argStrings.isEmpty() && ArgparseConstants.OPTIONAL.equals(action.getNargs())) {
            Object value = action.getOptionStrings() != null && !action.getOptionStrings().isEmpty()
                    ? action.getConst()
                    : action.getDefault();
            if (value instanceof String s && !ArgparseConstants.SUPPRESS.equals(s)) {
                value = getValue(action, s);
            }
            return value;
        } else if (argStrings.isEmpty()
                && ArgparseConstants.ZERO_OR_MORE.equals(action.getNargs())
                && (action.getOptionStrings() == null || action.getOptionStrings().isEmpty())) {
            if (action.getDefault() != null) return action.getDefault();
            return new ArrayList<>();
        } else if (argStrings.size() == 1
                && (action.getNargs() == null || ArgparseConstants.OPTIONAL.equals(action.getNargs()))) {
            String s = argStrings.get(0);
            Object v = getValue(action, s);
            checkValue(action, v);
            return v;
        } else if (ArgparseConstants.REMAINDER.equals(action.getNargs())) {
            List<Object> out = new ArrayList<>();
            for (String v : argStrings) out.add(getValue(action, v));
            return out;
        } else if (ArgparseConstants.PARSER.equals(action.getNargs())) {
            List<Object> out = new ArrayList<>();
            for (String v : argStrings) out.add(getValue(action, v));
            if (!out.isEmpty()) checkValue(action, out.get(0));
            return out;
        } else if (ArgparseConstants.SUPPRESS.equals(action.getNargs())) {
            return ArgparseConstants.SUPPRESS;
        } else {
            List<Object> out = new ArrayList<>();
            for (String v : argStrings) out.add(getValue(action, v));
            for (Object v : out) checkValue(action, v);
            return out;
        }
    }

    /** Apply the {@code type=} function, surfacing {@link ArgumentError} on failure. */
    public Object getValue(Action action, String argString) {
        Object typeFunc = action.getType();
        Object result;
        try {
            if (typeFunc == null || Identity.class.equals(typeFunc)) {
                result = argString;
            } else if (typeFunc instanceof Class<?> cls) {
                result = convertValue(cls, argString);
            } else if (typeFunc instanceof java.util.function.Function<?, ?> f) {
                @SuppressWarnings("unchecked")
                java.util.function.Function<String, Object> fn =
                        (java.util.function.Function<String, Object>) f;
                result = fn.apply(argString);
            } else if (typeFunc instanceof FileType ft) {
                result = ft.call(argString);
            } else {
                Method call = typeFunc.getClass().getMethod("call", Object.class);
                result = call.invoke(typeFunc, argString);
            }
        } catch (InvocationTargetException ite) {
            Throwable cause = ite.getCause();
            if (cause instanceof ArgumentTypeError) {
                throw new ArgumentError(action, cause.getMessage());
            }
            String name = getTypeName(action);
            throw new ArgumentError(action, "invalid " + name + " value: '" + argString + "'");
        } catch (ReflectiveOperationException ex) {
            throw new IllegalArgumentException("type callable failed: " + ex.getMessage(), ex);
        }
        return result;
    }

    /**
     * Convert a string value to the target type.
     * Handles all Java built-in numeric/wrapper types plus any class with
     * a static {@code valueOf(String)} or public String-arg constructor.
     */
    private static Object convertValue(Class<?> cls, String argString) {
        if (cls == Integer.class || cls == int.class)   return Integer.parseInt(argString);
        if (cls == Long.class || cls == long.class)    return Long.parseLong(argString);
        if (cls == Short.class || cls == short.class)  return Short.parseShort(argString);
        if (cls == Byte.class || cls == byte.class)    return Byte.parseByte(argString);
        if (cls == Double.class || cls == double.class) return Double.parseDouble(argString);
        if (cls == Float.class || cls == float.class)  return Float.parseFloat(argString);
        if (cls == Boolean.class || cls == boolean.class) return Boolean.parseBoolean(argString);
        // Generic: try valueOf(String) then new cls(String)
        try {
            Method vo = cls.getMethod("valueOf", String.class);
            return vo.invoke(null, argString);
        } catch (NoSuchMethodException e) {
            try {
                Constructor<?> ctor = cls.getConstructor(String.class);
                return ctor.newInstance(argString);
            } catch (NoSuchMethodException ex) {
                throw new IllegalArgumentException(
                        cls.getName() + " has neither valueOf(String) nor String constructor");
            } catch (IllegalAccessException | InvocationTargetException | InstantiationException exx) {
                throw new IllegalArgumentException(
                        cls.getName() + " has neither valueOf(String) nor String constructor");
            }
        } catch (IllegalAccessException | InvocationTargetException e) {
            throw new IllegalArgumentException("type conversion failed: " + e.getMessage());
        }
    }

    private String getTypeName(Action action) {
        Object t = action.getType();
        if (t instanceof Class<?>) return ((Class<?>) t).getSimpleName();
        if (t != null) return t.toString();
        return "str";
    }

    public void checkValue(Action action, Object value) {
        Object choices = action.getChoices();
        if (choices == null) return;
        if (choices instanceof String s) {
            if (s.indexOf(value == null ? "" : value.toString()) < 0) {
                throw new ArgumentError(action, "invalid choice: '" + value + "'");
            }
            return;
        }
        boolean ok = false;
        if (choices instanceof Iterable<?> it) {
            for (Object c : it) if (Objects.equals(c, value)) { ok = true; break; }
        }
        if (!ok) {
            StringBuilder cs = new StringBuilder();
            if (choices instanceof Iterable<?> it) for (Object c : it) {
                if (cs.length() > 0) cs.append(", ");
                cs.append(reprLike(c));
            }
            String msg = "invalid choice: '" + value + "' (choose from " + cs + ")";
            if (suggestOnError && value instanceof String) {
                if (choices instanceof Iterable<?> it) {
                    String suggestion = suggestMatch((String) value, it);
                    if (suggestion != null) {
                        msg = "invalid choice: '" + value + "', maybe you meant '" + suggestion
                                + "'? (choose from " + cs + ")";
                    }
                }
            }
            throw new ArgumentError(action, msg);
        }
    }

    private static String reprLike(Object o) {
        if (o == null) return "null";
        return "'" + o + "'";
    }

    private static String suggestMatch(String value, Iterable<?> choices) {
        String best = null;
        int bestScore = Integer.MAX_VALUE;
        for (Object c : choices) {
            String s = c == null ? "" : c.toString();
            int score = levenshtein(value, s);
            if (best == null || score < bestScore) {
                best = s;
                bestScore = score;
            }
        }
        return bestScore <= Math.max(1, value.length() / 3) ? best : null;
    }

    private static int levenshtein(String a, String b) {
        int[] prev = new int[b.length() + 1];
        int[] cur = new int[b.length() + 1];
        for (int j = 0; j <= b.length(); j++) prev[j] = j;
        for (int i = 1; i <= a.length(); i++) {
            cur[0] = i;
            for (int j = 1; j <= b.length(); j++) {
                int cost = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1;
                cur[j] = Math.min(Math.min(cur[j - 1] + 1, prev[j] + 1), prev[j - 1] + cost);
            }
            int[] tmp = prev; prev = cur; cur = tmp;
        }
        return prev[b.length()];
    }

    public int matchArgument(Action action, String argStringsPattern) {
        Pattern p = nargsPattern(action);
        java.util.regex.Matcher m = p.matcher(argStringsPattern);
        if (!m.find() || m.start() != 0) {
            String msg;
            Object n = action.getNargs();
            if (n == null) msg = "expected one argument";
            else if (ArgparseConstants.OPTIONAL.equals(n)) msg = "expected at most one argument";
            else if (ArgparseConstants.ONE_OR_MORE.equals(n)) msg = "expected at least one argument";
            else msg = "expected " + n + " arguments";
            throw new ArgumentError(action, msg);
        }
        return m.group(1).length();
    }

    public List<Integer> matchArgumentsPartial(List<Action> actions, String pattern) {
        for (int i = actions.size(); i > 0; i--) {
            List<Action> slice = actions.subList(0, i);
            StringBuilder fullPattern = new StringBuilder();
            for (Action a : slice) fullPattern.append(nargsPattern(a).pattern());
            java.util.regex.Matcher m = Pattern.compile("^" + fullPattern.toString()).matcher(pattern);
            if (m.find()) {
                List<Integer> counts = new ArrayList<>();
                for (int g = 1; g <= m.groupCount(); g++) counts.add(m.group(g) == null ? 0 : m.group(g).length());
                if (m.end() < pattern.length() && pattern.charAt(m.end()) == 'O') {
                    while (!counts.isEmpty() && counts.get(counts.size() - 1) == 0) counts.remove(counts.size() - 1);
                }
                return counts;
            }
        }
        return new ArrayList<>();
    }

    public List<Object[]> _parseOptional(String argString) {
        if (argString == null || argString.isEmpty()) return null;
        if (!startsWithPrefix(argString)) return null;
        if (optionStringActions.containsKey(argString)) {
            Action a = optionStringActions.get(argString);
            List<Object[]> r = new ArrayList<>();
            r.add(new Object[] { a, argString, null, null });
            return r;
        }
        if (argString.length() == 1) return null;
        String[] parts = argString.split("=", 2);
        String option = parts[0];
        String explicitArg = parts.length > 1 ? parts[1] : null;
        if (parts.length > 1 && optionStringActions.containsKey(option)) {
            Action a = optionStringActions.get(option);
            List<Object[]> r = new ArrayList<>();
            r.add(new Object[] { a, option, "=", explicitArg });
            return r;
        }
        List<Object[]> optionTuples = _getOptionTuples(argString);
        if (optionTuples != null && !optionTuples.isEmpty()) return optionTuples;
        if (negativeNumberMatcher.matcher(argString).find()
                && hasNegativeNumberOptionals.isEmpty()) return null;
        if (argString.contains(" ")) return null;
        List<Object[]> singleNull = new ArrayList<>();
        singleNull.add(new Object[] { null, argString, null, null });
        return singleNull;
    }

    public List<Object[]> _getOptionTuples(String optionString) {
        List<Object[]> result = new ArrayList<>();
        if (prefixChars.indexOf(optionString.charAt(0)) >= 0
                && optionString.length() > 1
                && prefixChars.indexOf(optionString.charAt(1)) >= 0) {
            int sepIdx = optionString.indexOf('=');
            String optionPrefix = sepIdx >= 0 ? optionString.substring(0, sepIdx) : optionString;
            String sep = sepIdx >= 0 ? "=" : null;
            String explicitArg = sepIdx >= 0 ? optionString.substring(sepIdx + 1) : null;
            if (allowAbbrev) {
                for (Map.Entry<String, Action> e : optionStringActions.entrySet()) {
                    if (e.getKey().startsWith(optionPrefix)) {
                        result.add(new Object[] { e.getValue(), e.getKey(), sep, explicitArg });
                    }
                }
            }
        } else if (prefixChars.indexOf(optionString.charAt(0)) >= 0
                && optionString.length() > 1
                && prefixChars.indexOf(optionString.charAt(1)) < 0) {
            String shortOptionPrefix = optionString.substring(0, 2);
            String shortExplicitArg = optionString.substring(2);
            for (Map.Entry<String, Action> e : optionStringActions.entrySet()) {
                String key = e.getKey();
                if (key.equals(shortOptionPrefix)) {
                    result.add(new Object[] { e.getValue(), key, "", shortExplicitArg });
                } else if (allowAbbrev && key.startsWith(optionString)) {
                    int sepIdx = optionString.indexOf('=');
                    String sep = sepIdx >= 0 ? "=" : null;
                    String explicitArg = sepIdx >= 0 ? optionString.substring(sepIdx + 1) : null;
                    result.add(new Object[] { e.getValue(), key, sep, explicitArg });
                }
            }
        }
        return result;
    }

    protected List<String> _readArgsFromFiles(List<String> argStrings) {
        List<String> out = new ArrayList<>();
        for (String s : argStrings) {
            if (s == null || s.isEmpty() || fromfilePrefixChars.indexOf(s.charAt(0)) < 0) {
                out.add(s);
                continue;
            }
            try {
                String content = new String(Files.readAllBytes(Paths.get(s.substring(1))), StandardCharsets.UTF_8);
                List<String> sub = new ArrayList<>();
                for (String line : content.split("\\r?\\n", -1)) {
                    for (String a : convertArgLineToArgs(line)) sub.add(a);
                }
                out.addAll(_readArgsFromFiles(sub));
            } catch (IOException e) {
                throw new ArgumentError(e.getMessage());
            }
        }
        return out;
    }

    public List<String> convertArgLineToArgs(String argLine) {
        return Arrays.asList(argLine.trim().split("\\s+"));
    }

    public Namespace parseIntermixedArgs() { return parseIntermixedArgs(null); }
    public Namespace parseIntermixedArgs(List<String> args) {
        ParseResult r = parseKnownIntermixedArgs(args, null);
        if (!r.extras.isEmpty()) {
            String msg = "unrecognized arguments: " + String.join(" ", r.extras);
            if (exitOnError) error(msg);
            else throw new ArgumentError(msg);
        }
        return r.namespace;
    }
    public ParseResult parseKnownIntermixedArgs() { return parseKnownIntermixedArgs(null, null); }
    public ParseResult parseKnownIntermixedArgs(List<String> args) { return parseKnownIntermixedArgs(args, null); }
    public ParseResult parseKnownIntermixedArgs(List<String> args, Namespace namespace) {
        for (Action a : _getPositionalActions()) {
            if (ArgparseConstants.PARSER.equals(a.getNargs()) || ArgparseConstants.REMAINDER.equals(a.getNargs())) {
                throw new IllegalStateException("parse_intermixed_args: positional arg with nargs="
                        + a.getNargs());
            }
        }
        return _parseKnownArgs2(args == null ? null : new ArrayList<>(args), namespace, true);
    }

    // ====================================================
    //  help / message / exit
    // ====================================================

    public HelpFormatter getFormatter() {
        if (cachedFormatter == null) {
            try {
                Constructor<? extends HelpFormatter> ctor = formatterClass.getConstructor(String.class);
                cachedFormatter = ctor.newInstance(prog);
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("cannot construct formatter", e);
            }
        }
        cachedFormatter.setColor(color);
        return cachedFormatter;
    }

    public String formatUsage() {
        HelpFormatter f = getFormatter();
        f.addUsage(usage, actions, new ArrayList<>(mutuallyExclusiveGroups));
        return f.formatHelp();
    }

    public String formatHelp() {
        HelpFormatter f = getFormatter();
        f.addUsage(usage, actions, new ArrayList<>(mutuallyExclusiveGroups));
        f.addText(description);
        for (_ArgumentGroup g : actionGroups) {
            f.startSection(g.getTitle());
            f.addText(g.getGroupDescription());
            f.addArguments(g.groupActions);
            f.endSection();
        }
        f.addText(epilog);
        return f.formatHelp();
    }

    public void printUsage() { printUsage(System.out); }
    public void printUsage(java.io.PrintStream out) {
        out.print(formatUsage());
    }
    public void printHelp() { printHelp(System.out); }
    public void printHelp(java.io.PrintStream out) {
        out.print(formatHelp());
    }
    public void printHelp(File file) throws IOException {
        try (PrintWriter w = new PrintWriter(file, StandardCharsets.UTF_8)) {
            w.print(formatHelp());
        }
    }

    public void printMessage(String message) { printMessage(message, System.err); }
    public void printMessage(String message, java.io.PrintStream stream) {
        if (message != null) stream.print(message);
    }

    public void exit(int status) { exit(status, null); }
    public void exit(int status, String message) {
        if (message != null) printMessage(message, System.err);
        if (!exitOnError && status != 0) {
            throw new ExitTrappedException(status, message);
        }
        System.exit(status);
    }

    public void error(String message) {
        printUsage(System.err);
        String full = prog + ": error: " + message + "\n";
        if (exitOnError) exit(2, full);
        else throw new ArgumentError(message);
    }

    public void warning(String message) {
        printMessage(prog + ": warning: " + message + "\n", System.err);
    }

    public String getProg() { return prog; }
    public void setProg(String p) { this.prog = p; this.cachedFormatter = null; }
    public String getUsage() { return usage; }
    public void setUsage(String u) { this.usage = u; }
    public String getEpilog() { return epilog; }
    public void setEpilog(String e) { this.epilog = e; }
    public Class<? extends HelpFormatter> getFormatterClass() { return formatterClass; }
    public void setFormatterClass(Class<? extends HelpFormatter> cls) {
        this.formatterClass = cls;
        this.cachedFormatter = null;
    }
    public String getFromfilePrefixChars() { return fromfilePrefixChars; }
    public void setFromfilePrefixChars(String s) { this.fromfilePrefixChars = s; }
    public boolean isAddHelp() { return addHelp; }
    public void setAddHelp(boolean v) { this.addHelp = v; }
    public boolean isAllowAbbrev() { return allowAbbrev; }
    public void setAllowAbbrev(boolean v) { this.allowAbbrev = v; }
    public boolean isExitOnError() { return exitOnError; }
    public void setExitOnError(boolean v) { this.exitOnError = v; }
    public boolean isSuggestOnError() { return suggestOnError; }
    public void setSuggestOnError(boolean v) { this.suggestOnError = v; }
    public boolean isColor() { return color; }
    public void setColor(boolean v) { this.color = v; }
    public String getVersion() { return version; }
    public void setVersion(String v) { this.version = v; }
    public _ArgumentGroup getPositionalsGroup() { return positionals; }
    public _ArgumentGroup getOptionalsGroup() { return optionals; }
    public Object getSubparsers() { return subparsers; }
}