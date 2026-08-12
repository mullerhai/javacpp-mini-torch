/*
 * Ramp scheduler — automated canary / ramp schedule controller.
 *
 * Industry usage:
 *   - Meta / Google: timed canary ramp 1% -> 5% -> 25% -> 50% -> 100%
 *     over hours/days with automatic promotion if guardrails stay green.
 *   - ByteDance / Alibaba: 灰度发布 (gray release) with schedule JSON.
 *   - Uber: "auto-promotion" service that advances ramp stages based on
 *     guardrail health.
 *
 * This module produces, given an experiment definition, a sequence of
 * (time -> trafficPercent, duration) stages and tracks current progress.
 * It does NOT itself advance the experiment; the caller (or a separate
 * scheduler thread) is responsible for invoking
 * {@link LayeredExperimentManager#setTrafficPercent} at each step.
 */
package org.bytedeco.pytorch.deploy.abtest;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Computes and tracks a canary ramp schedule.
 */
public final class RampScheduler {

    /** A single ramp step. */
    public static final class RampStep {
        public final int index;
        public final double trafficPercent;
        public final Duration dwell;

        public RampStep(int index, double trafficPercent, Duration dwell) {
            this.index = index;
            this.trafficPercent = trafficPercent;
            this.dwell = dwell;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "RampStep{idx=%d traffic=%.2f%% dwell=%s}",
                    index, trafficPercent, dwell);
        }
    }

    /** Configurable ramp schedule (immutable). */
    public static final class Schedule {
        public final String experimentId;
        public final List<RampStep> steps;
        public final Instant startedAt;
        public final boolean autoAdvance;

        public Schedule(String experimentId, List<RampStep> steps, Instant startedAt,
                        boolean autoAdvance) {
            this.experimentId = experimentId;
            this.steps = Collections.unmodifiableList(new ArrayList<>(steps));
            this.startedAt = startedAt;
            this.autoAdvance = autoAdvance;
        }

        public Duration totalDwell() {
            Duration d = Duration.ZERO;
            for (RampStep s : steps) d = d.plus(s.dwell);
            return d;
        }
    }

    /** Schedule builder following industry canary curve. */
    public static Schedule standardCanary(String experimentId, Instant startedAt) {
        return Schedule(experimentId)
                .step(1.0, Duration.ofHours(1))
                .step(5.0, Duration.ofHours(2))
                .step(10.0, Duration.ofHours(6))
                .step(25.0, Duration.ofHours(12))
                .step(50.0, Duration.ofHours(24))
                .step(100.0, Duration.ofDays(7))
                .autoAdvance(true)
                .startedAt(startedAt)
                .build();
    }

    /** Custom schedule builder. */
    public static ScheduleBuilder Schedule(String experimentId) {
        return new ScheduleBuilder(experimentId);
    }

    /** Mutable tracker for an active schedule. */
    public static final class RampTracker {
        private final Schedule schedule;
        private final AtomicReference<State> state = new AtomicReference<>();

        public RampTracker(Schedule schedule) {
            this.schedule = Objects.requireNonNull(schedule);
            this.state.set(new State(0, schedule.startedAt, schedule.startedAt, schedule.steps.get(0).trafficPercent));
        }

        public Schedule schedule() { return schedule; }

        public State currentState() { return state.get(); }

        /** Advance the tracker to now. Returns true if a new stage is reached. */
        public boolean tick(Instant now) {
            State cur = state.get();
            int idx = cur.stageIndex;
            if (idx >= schedule.steps.size() - 1) {
                return false;
            }
            RampStep curStep = schedule.steps.get(idx);
            Instant stageStart = cur.stageStartedAt;
            if (Duration.between(stageStart, now).compareTo(curStep.dwell) < 0) {
                return false;
            }
            int nextIdx = idx + 1;
            RampStep nextStep = schedule.steps.get(nextIdx);
            State next = new State(nextIdx, now, now, nextStep.trafficPercent);
            if (state.compareAndSet(cur, next)) {
                return true;
            }
            return false;
        }

        public static final class State {
            public final int stageIndex;
            public final Instant lastAdvancedAt;
            public final Instant stageStartedAt;
            public final double currentTrafficPercent;

            public State(int stageIndex, Instant lastAdvancedAt, Instant stageStartedAt,
                         double currentTrafficPercent) {
                this.stageIndex = stageIndex;
                this.lastAdvancedAt = lastAdvancedAt;
                this.stageStartedAt = stageStartedAt;
                this.currentTrafficPercent = currentTrafficPercent;
            }

            @Override
            public String toString() {
                return String.format(Locale.ROOT,
                        "RampState{stage=%d traffic=%.2f%% stageStarted=%s lastAdvance=%s}",
                        stageIndex, currentTrafficPercent, stageStartedAt, lastAdvancedAt);
            }
        }
    }

    /** Builder for arbitrary schedules. */
    public static final class ScheduleBuilder {
        private final String experimentId;
        private final List<RampStep> steps = new ArrayList<>();
        private Instant startedAt = Instant.now();
        private boolean autoAdvance = false;

        private ScheduleBuilder(String experimentId) {
            this.experimentId = experimentId;
        }

        public ScheduleBuilder step(double percent, Duration dwell) {
            steps.add(new RampStep(steps.size(), percent, dwell));
            return this;
        }

        public ScheduleBuilder startedAt(Instant t) { this.startedAt = t; return this; }
        public ScheduleBuilder autoAdvance(boolean b) { this.autoAdvance = b; return this; }

        public Schedule build() {
            if (steps.isEmpty()) {
                throw new IllegalStateException("at least one ramp step required");
            }
            return new Schedule(experimentId, steps, startedAt, autoAdvance);
        }
    }
}