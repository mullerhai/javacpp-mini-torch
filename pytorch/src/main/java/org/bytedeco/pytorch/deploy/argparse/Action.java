/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse;

import java.util.List;
import java.util.Set;

/**
 * Base class for all parser Actions. Mirrors Python {@code argparse.Action}.
 *
 * <p>An Action encapsulates:
 * <ul>
 *   <li>how to bind command-line strings to a Java value (the {@link #call} method),</li>
 *   <li>how to render itself in usage / help strings ({@link #formatUsage()}),</li>
 *   <li>its optional/positional nature ({@link #getOptionStrings()}),</li>
 *   <li>validation metadata ({@link #getChoices()}, {@link #getType()}, {@link #getDest()}).</li>
 * </ul>
 *
 * <p>Built-in subclasses live in { argparse.action}. Custom subclasses
 * may be passed via {@code addArgument().actionClass(...)}.
 */
public abstract class Action extends _AttributeHolder {

    private List<String> optionStrings;
    private Object dest;
    private Object nargs; // Integer or String
    private Object constValue;
    private Object defaultValue;
    private Object type; // Class<?> or String-keyed registered type
    private Object choices; // Set<?> or List<?>
    private boolean required;
    private String help;
    private Object metavar; // String or String[]
    private boolean deprecated;

    protected Action() {
    }

    protected Action(List<String> optionStrings,
                     String dest,
                     Object nargs,
                     Object constValue,
                     Object defaultValue,
                     Object type,
                     Object choices,
                     boolean required,
                     String help,
                     Object metavar,
                     boolean deprecated) {
        this.optionStrings = optionStrings;
        this.dest = dest;
        this.nargs = nargs;
        this.constValue = constValue;
        this.defaultValue = defaultValue;
        this.type = type;
        this.choices = choices;
        this.required = required;
        this.help = help;
        this.metavar = metavar;
        this.deprecated = deprecated;
    }

    public List<String> getOptionStrings() {
        return optionStrings;
    }

    public void setOptionStrings(List<String> optionStrings) {
        this.optionStrings = optionStrings;
    }

    /**
     * Container holding raw kwargs needed for subparsers construction
     * (the original map captured at {@code addArgument()} time).
     */
    private java.util.Map<String, Object> rawOptions;
    public void setRawOptions(java.util.Map<String, Object> opts) { this.rawOptions = opts; }
    public java.util.Map<String, Object> getRawOptions() { return rawOptions; }

    /** Container reference, mirrors Python's {@code action.container}. */
    public Object container;

    public String getDest() {
        return dest == null ? null : dest.toString();
    }

    public Object getRawDest() {
        return dest;
    }

    public void setRawDest(Object dest) {
        this.dest = dest;
    }

    public Object getNargs() {
        return nargs;
    }

    public void setNargs(Object nargs) {
        this.nargs = nargs;
    }

    public Object getConst() {
        return constValue;
    }

    public void setConst(Object c) {
        this.constValue = c;
    }

    public Object getDefault() {
        return defaultValue;
    }

    public void setDefault(Object d) {
        this.defaultValue = d;
    }

    public Object getType() {
        return type;
    }

    public void setType(Object type) {
        this.type = type;
    }

    public Object getChoices() {
        return choices;
    }

    public void setChoices(Object choices) {
        this.choices = choices;
    }

    public boolean isRequired() {
        return required;
    }

    public void setRequired(boolean required) {
        this.required = required;
    }

    public String getHelp() {
        return help;
    }

    public void setHelp(String help) {
        this.help = help;
    }

    public Object getMetavar() {
        return metavar;
    }

    public void setMetavar(Object metavar) {
        this.metavar = metavar;
    }

    public boolean isDeprecated() {
        return deprecated;
    }

    public void setDeprecated(boolean d) {
        this.deprecated = d;
    }

    /** Parse the action invocation. Mirrors Python {@code Action.__call__}.
     *  The {@code values} parameter holds the result of {@link ArgumentParser#getValues};
     *  it is either a single converted value (for nargs=0/1/?/PARSER) or a List. */
    public abstract void call(ArgumentParser parser, Namespace namespace,
                              Object values, String optionString);

    /** Renders the short usage form. Mirrors Python {@code Action.format_usage}. */
    public String formatUsage() {
        return optionStrings == null || optionStrings.isEmpty()
                ? ""
                : optionStrings.get(0);
    }

    /** Iterates sub-actions for formatter (e.g. {@link BooleanOptionalAction}). */
    public Iterable<Action> getSubactions() {
        return List.of();
    }

    @Override
    protected List<String> collectKwargNames() {
        return List.of(
                "option_strings", "dest", "nargs", "const", "default",
                "type", "choices", "required", "help", "metavar", "deprecated");
    }

    @Override
    public Object readAttribute(String name) {
        return switch (name) {
            case "option_strings" -> optionStrings;
            case "dest" -> dest;
            case "nargs" -> nargs;
            case "const" -> constValue;
            case "default" -> defaultValue;
            case "type" -> type;
            case "choices" -> choices;
            case "required" -> required;
            case "help" -> help;
            case "metavar" -> metavar;
            case "deprecated" -> deprecated;
            default -> null;
        };
    }
}