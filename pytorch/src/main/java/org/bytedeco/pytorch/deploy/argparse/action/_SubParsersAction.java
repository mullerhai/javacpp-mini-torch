/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.action;

import org.bytedeco.pytorch.deploy.argparse.Action;
import org.bytedeco.pytorch.deploy.argparse.ArgumentError;
import org.bytedeco.pytorch.deploy.argparse.ArgumentParser;
import org.bytedeco.pytorch.deploy.argparse.Namespace;
import org.bytedeco.pytorch.deploy.argparse.ArgparseConstants;
import org.bytedeco.pytorch.deploy.argparse.formatter.HelpFormatter;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Mirrors Python {@code _SubParsersAction}.
 *
 * <p>The full construction (with the registered subparser map, choice pseudo-actions,
 * aliases and deprecation flags) is performed by {@code _ActionsContainer.addSubparsers},
 * which delegates to {@link #addParser}.
 */
public class _SubParsersAction extends Action {

    /** Pseudo-action used only to render the list of choices in help. */
    public static class _ChoicesPseudoAction extends Action {
        private final List<String> aliases;
        public _ChoicesPseudoAction(String name, List<String> aliases, String help) {
            super(new ArrayList<>(), name, null, null, null, null, null, false,
                    help, name + (aliases == null || aliases.isEmpty()
                            ? ""
                            : " (" + String.join(", ", aliases) + ")"),
                    false);
            this.aliases = aliases;
        }
        public List<String> getAliases() { return aliases; }
        @Override public void call(ArgumentParser parser, Namespace namespace,
                                   Object values, String optionString) {
            // Pseudo-action never executes; just a placeholder for help rendering.
        }
    }

    private final String progPrefix;
    private final Class<? extends ArgumentParser> parserClass;
    private final Map<String, ArgumentParser> nameParserMap = new LinkedHashMap<>();
    private final List<_ChoicesPseudoAction> choicesActions = new ArrayList<>();
    private final java.util.Set<String> deprecatedNames = new java.util.HashSet<>();
    private boolean color = true;

    public _SubParsersAction() {
        this.progPrefix = "";
        this.parserClass = ArgumentParser.class;
    }

    public _SubParsersAction(String progPrefix,
                             Class<? extends ArgumentParser> parserClass,
                             List<String> optionStrings,
                             String dest,
                             boolean required,
                             String help,
                             Object metavar) {
        super(optionStrings, dest, ArgparseConstants.PARSER, null, null, null,
                null, required, help, metavar, false);
        this.progPrefix = progPrefix;
        this.parserClass = parserClass;
        setChoices(nameParserMap);
    }

    public String getProgPrefix() { return progPrefix; }
    public Class<? extends ArgumentParser> getParserClass() { return parserClass; }
    public Map<String, ArgumentParser> getNameParserMap() { return nameParserMap; }
    public List<_ChoicesPseudoAction> getChoicesActions() { return choicesActions; }
    public java.util.Set<String> getDeprecatedNames() { return deprecatedNames; }
    public boolean isColor() { return color; }
    public void setColor(boolean c) { this.color = c; }

    /**
     * Convenience overload that takes an existing parser and a kwargs map.
     * Mirrors Python: {@code subparsers.add_parser("install", parents=[install])}.
     */
    public ArgumentParser addParser(String name, ArgumentParser parser,
                                     Map<String, Object> kwargs) {
        if (nameParserMap.containsKey(name)) {
            throw new IllegalArgumentException("conflicting subparser: " + name);
        }
        @SuppressWarnings("unchecked")
        List<String> aliases = (List<String>) kwargs.getOrDefault("aliases", new ArrayList<>());
        for (String alias : aliases) {
            if (nameParserMap.containsKey(alias)) {
                throw new IllegalArgumentException("conflicting subparser alias: " + alias);
            }
        }
        Object helpObj = kwargs.get("help");
        if (helpObj instanceof String h) {
            _ChoicesPseudoAction choiceAction = new _ChoicesPseudoAction(name, aliases, h);
            choicesActions.add(choiceAction);
        }
        nameParserMap.put(name, parser);
        for (String alias : aliases) {
            nameParserMap.put(alias, parser);
        }
        Object depObj = kwargs.get("deprecated");
        if (Boolean.TRUE.equals(depObj)) {
            deprecatedNames.add(name);
            deprecatedNames.addAll(aliases);
        }
        return parser;
    }

    public ArgumentParser addParser(String name,
                                     Map<String, Object> kwargs) {
        if (nameParserMap.containsKey(name)) {
            throw new IllegalArgumentException("conflicting subparser: " + name);
        }
        @SuppressWarnings("unchecked")
        List<String> aliases = (List<String>) kwargs.getOrDefault("aliases", new ArrayList<>());
        for (String alias : aliases) {
            if (nameParserMap.containsKey(alias)) {
                throw new IllegalArgumentException("conflicting subparser alias: " + alias);
            }
        }
        Object helpObj = kwargs.get("help");
        _ChoicesPseudoAction choiceAction = null;
        if (helpObj instanceof String h) {
            choiceAction = new _ChoicesPseudoAction(name, aliases, h);
            choicesActions.add(choiceAction);
        }
        String prog = (String) kwargs.getOrDefault("prog", progPrefix + " " + name);
        kwargs.put("prog", prog);

        ArgumentParser parser;
        try {
            parser = parserClass.getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException ex) {
            throw new IllegalStateException("parser_class must have a no-arg constructor", ex);
        }
        for (Map.Entry<String, Object> e : kwargs.entrySet()) {
            String k = e.getKey();
            Object v = e.getValue();
            if ("prog".equals(k)) continue;
            applyKwargs(parser, k, v);
        }
        if (choiceAction != null) {
            parser.checkHelp(choiceAction);
        }
        nameParserMap.put(name, parser);
        for (String alias : aliases) {
            nameParserMap.put(alias, parser);
        }
        Object depObj = kwargs.get("deprecated");
        if (Boolean.TRUE.equals(depObj)) {
            deprecatedNames.add(name);
            deprecatedNames.addAll(aliases);
        }
        return parser;
    }

    private static void applyKwargs(ArgumentParser parser, String key, Object value) {
        switch (key) {
            case "description" -> parser.setDescription((String) value);
            case "epilog" -> parser.setEpilog((String) value);
            case "usage" -> parser.setUsage((String) value);
            case "conflict_handler" -> parser.setConflictHandler((String) value);
            case "argument_default" -> parser.setArgumentDefault(value);
            case "add_help" -> parser.setAddHelp((Boolean) value);
            case "allow_abbrev" -> parser.setAllowAbbrev((Boolean) value);
            case "exit_on_error" -> parser.setExitOnError((Boolean) value);
            case "prefix_chars" -> parser.setPrefixChars((String) value);
            case "fromfile_prefix_chars" -> parser.setFromfilePrefixChars((String) value);
            case "formatter_class" -> parser.setFormatterClass((Class<? extends HelpFormatter>) value);
            case "suggest_on_error" -> parser.setSuggestOnError((Boolean) value);
            case "color" -> parser.setColor((Boolean) value);
            default -> { /* ignore unknown */ }
        }
    }

    @Override
    public Iterable<Action> getSubactions() {
        return new ArrayList<>(choicesActions);
    }

    @Override
    @SuppressWarnings("unchecked")
    public void call(ArgumentParser parser, Namespace namespace,
                     Object values, String optionString) {
        if (values == null) values = new ArrayList<>();
        List<String> vals = values instanceof List
                ? (List<String>) values
                : List.of(values.toString());
        if (vals.isEmpty()) {
            throw new ArgumentError("expected at least one argument");
        }
        String parserName = vals.get(0);
        List<String> argStrings = vals.subList(1, vals.size());
        if (!ArgparseConstants.SUPPRESS.equals(getDest())) {
            namespace.set(getDest(), parserName);
        }
        ArgumentParser subparser = nameParserMap.get(parserName);
        if (subparser == null) {
            throw new ArgumentError("unknown parser " + repr(parserName)
                    + " (choices: " + String.join(", ", nameParserMap.keySet()) + ")");
        }
        if (deprecatedNames.contains(parserName)) {
            parser.warning("command '" + parserName + "' is deprecated");
        }
        Namespace subns = subparser.parseKnownArgs(argStrings, null).namespace;
        for (Map.Entry<String, Object> e : subns.asMap().entrySet()) {
            namespace.set(e.getKey(), e.getValue());
        }
        List<String> extra = subparser.parseKnownArgs(argStrings, null).extras;
        if (!extra.isEmpty()) {
            @SuppressWarnings("unchecked")
            List<String> ext = (List<String>) namespace.get(ArgparseConstants.UNRECOGNIZED_ARGS_ATTR);
            if (ext == null) {
                ext = new ArrayList<>();
                namespace.set(ArgparseConstants.UNRECOGNIZED_ARGS_ATTR, ext);
            }
            ext.addAll(extra);
        }
    }

    private static String repr(String s) {
        return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'";
    }

    public static _SubParsersAction create(Object optionStrings, String prog, Class<?> parserClass,
                                           String dest, boolean required, String help,
                                           Object metavar) {
        @SuppressWarnings("unchecked")
        Class<? extends ArgumentParser> pc = (Class<? extends ArgumentParser>) parserClass;
        return new _SubParsersAction(prog, pc, (List<String>) optionStrings, dest, required,
                help, metavar);
    }
}