/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.container;

import org.bytedeco.pytorch.deploy.argparse.Action;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Mirrors Python's {@code _ArgumentGroup}: a labeled bucket for organizing
 * actions in help output. Shares registries / actions / optionStringActions /
 * defaults with its parent container (so actions added to the group are
 * visible to the parser).
 */
public class _ArgumentGroup extends _ActionsContainer {

    public final _ActionsContainer parent;
    public String title;
    /** Public description for cross-package access in formatter. */
    public String groupDescription;
    public final List<Action> groupActions = new ArrayList<>();

    public _ArgumentGroup(_ActionsContainer parent, String title, String description,
                          String prefixChars, Object argumentDefault, String conflictHandler) {
        super(description, prefixChars, argumentDefault, conflictHandler);
        this.parent = parent;
        this.title = title;
        this.groupDescription = description;
        // share registries and state with parent
        this.registries.putAll(parent.registries);
        this.actions.addAll(0, new ArrayList<>(parent.getActions())); // shared reference
        // share optionStringActions by reference
        shareMaps(parent);
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    private void shareMaps(_ActionsContainer parent) {
        try {
            java.lang.reflect.Field f = _ActionsContainer.class.getDeclaredField("optionStringActions");
            f.setAccessible(true);
            this.optionStringActions.clear();
            this.optionStringActions.putAll((Map) f.get(parent));
        } catch (ReflectiveOperationException ignored) {}
    }

    @Override
    protected Action _addArgumentImpl(List<String> args, Map<String, Object> kwargs) {
        return parent._addArgumentImpl(args, kwargs);
    }

    @Override
    public _ArgumentGroup addArgumentGroup(String title, String description) {
        throw new IllegalArgumentException("argument groups cannot be nested");
    }

    @Override
    public _MutuallyExclusiveGroup addMutuallyExclusiveGroup(boolean required) {
        return parent.addMutuallyExclusiveGroup(required);
    }

    @Override
    public Object addSubparsers(Map<String, Object> kwargs) {
        return parent.addSubparsers(kwargs);
    }

    @Override
    public void _checkConflict(Action action) {
        parent._checkConflict(action);
    }

    @Override
    public Action _addAction(Action action) {
        action = super._addAction(action);
        groupActions.add(action);
        return action;
    }

    public List<Action> getGroupActions() { return groupActions; }
    public String getTitle() { return title; }
    public String getGroupDescription() { return groupDescription; }
}