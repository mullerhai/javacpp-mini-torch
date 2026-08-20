/*
 * Python argparse 1:1 Java port. See ARGPARSE_IMPLEMENTATION_PLAN.md.
 *
 * Licensed under the Apache License, Version 2.0.
 */
package org.bytedeco.pytorch.deploy.argparse.container;

import org.bytedeco.pytorch.deploy.argparse.Action;
import org.bytedeco.pytorch.deploy.argparse.ArgumentError;

import java.util.Map;

/**
 * Mirrors Python's {@code _MutuallyExclusiveGroup}: an {@link _ArgumentGroup}
 * whose actions cannot be combined. Actions in the group are routed to the
 * parent container for parsing; the group keeps its own list of those actions
 * so that {@code formatter} can render them as {@code (a | b | c)} blocks.
 */
public class _MutuallyExclusiveGroup extends _ArgumentGroup {

    private final _ActionsContainer realContainer;
    public final boolean required;

    public _MutuallyExclusiveGroup(_ActionsContainer container, boolean required) {
        super(container, null, null,
                container.getPrefixChars(), container.getArgumentDefault(),
                container.getConflictHandler());
        this.required = required;
        this.realContainer = container;
    }

    @Override
    public Action _addAction(Action action) {
        if (action.isRequired()) {
            throw new ArgumentError(action, "mutually exclusive arguments must be optional");
        }
        action = realContainer._addAction(action);
        groupActions.add(action);
        return action;
    }

    @Override
    public _ArgumentGroup addArgumentGroup(String title, String description) {
        throw new IllegalArgumentException("argument groups cannot be nested");
    }

    @Override
    public _MutuallyExclusiveGroup addMutuallyExclusiveGroup(boolean required) {
        throw new IllegalArgumentException("mutually exclusive groups cannot be nested");
    }

    public boolean isRequired() { return required; }

    @Override
    protected Action _addArgumentImpl(java.util.List<String> args, Map<String, Object> kwargs) {
        return realContainer._addArgumentImpl(args, kwargs);
    }
}