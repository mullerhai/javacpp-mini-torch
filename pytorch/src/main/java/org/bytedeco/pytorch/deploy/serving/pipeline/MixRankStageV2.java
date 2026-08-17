/*
 * Enhanced Mix Rank Stage — enterprise-grade multi-queue blending with business rules.
 *
 * Key enhancements:
 *   1. Multi-queue blending with configurable policies
 *   2. Ad frequency cap and pacing controls
 *   3. Business rule injection (safety, campaigns, notices)
 *   4. Budget pacing for ads
 *   5. Cold-start slot protection
 *   6. Multi-business logic priority
 *   7. Traffic shaping per queue
 *
 * Production patterns (Alibaba, ByteDance, Tencent):
 *   - 混排 (mix-ranking) with multiple queues
 *   - Ad frequency cap (daily/hourly limits)
 *   - Budget pacing (spend rate control)
 *   - Force-insert for campaigns and operations
 *   - Safety filtering
 *   - Cold-start exploration slots
 */
package org.bytedeco.pytorch.deploy.serving.pipeline;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.*;

/**
 * Enterprise-grade mix-rank stage with business rules and ad controls.
 */
public final class MixRankStageV2 implements RankStage {

    /**
     * Queue type for different content sources.
     */
    public enum QueueType {
        ORGANIC,        // Regular recommendations
        AD,             // Advertisements
        PROMOTED,       // Promoted/organic content
        SOCIAL,         // Social graph insertions
        CAMPAIGN,       // Campaign/ops force-insert
        COLD_START,     // Exploration slots
        SAFETY_NOTICE   // Safety/system notices
    }

    /**
     * Queue configuration with business rules.
     */
    public static final class QueueV2 {
        public final String name;
        public final QueueType type;
        public final List<Candidate> candidates;
        public final int minGap;           // Minimum positions between inserts
        public final int maxInserts;       // Maximum total inserts
        public final List<Integer> fixedPositions;
        public final double fillWeight;     // Priority weight when filling gaps
        public final boolean required;     // Must fill if possible
        public final List<QueueFilter> filters;

        private QueueV2(String name, QueueType type, List<Candidate> candidates,
                       int minGap, int maxInserts, List<Integer> fixedPositions,
                       double fillWeight, boolean required, List<QueueFilter> filters) {
            this.name = name;
            this.type = type;
            this.candidates = candidates;
            this.minGap = minGap;
            this.maxInserts = maxInserts;
            this.fixedPositions = fixedPositions;
            this.fillWeight = fillWeight;
            this.required = required;
            this.filters = filters;
        }

        public static Builder builder(String name, QueueType type) {
            return new Builder(name, type);
        }

        public static QueueV2 organic(String name, List<Candidate> items) {
            return new Builder(name, QueueType.ORGANIC)
                    .candidates(items)
                    .minGap(0)
                    .maxInserts(Integer.MAX_VALUE)
                    .fillWeight(1.0)
                    .build();
        }

        public static QueueV2 ad(String name, List<Candidate> items, int interval, int maxInserts) {
            return new Builder(name, QueueType.AD)
                    .candidates(items)
                    .minGap(interval - 1)
                    .maxInserts(maxInserts)
                    .fillWeight(0.5)
                    .required(false)
                    .build();
        }

        public static QueueV2 campaign(String name, List<Candidate> items, int position) {
            return new Builder(name, QueueType.CAMPAIGN)
                    .candidates(items)
                    .minGap(0)
                    .maxInserts(items.size())
                    .fixedPositions(position)
                    .fillWeight(0.0)
                    .required(true)
                    .build();
        }

        public static final class Builder {
            private final String name;
            private final QueueType type;
            private List<Candidate> candidates = new ArrayList<>();
            private int minGap = 0;
            private int maxInserts = Integer.MAX_VALUE;
            private List<Integer> fixedPositions = new ArrayList<>();
            private double fillWeight = 1.0;
            private boolean required = false;
            private List<QueueFilter> filters = new ArrayList<>();

            private Builder(String name, QueueType type) {
                this.name = Objects.requireNonNull(name);
                this.type = Objects.requireNonNull(type);
            }

            public Builder candidates(List<Candidate> candidates) {
                this.candidates = candidates != null ? new ArrayList<>(candidates) : new ArrayList<>();
                return this;
            }

            public Builder minGap(int minGap) {
                this.minGap = Math.max(0, minGap);
                return this;
            }

            public Builder maxInserts(int maxInserts) {
                this.maxInserts = Math.max(0, maxInserts);
                return this;
            }

            public Builder fixedPositions(int... positions) {
                this.fixedPositions = new ArrayList<>();
                for (int p : positions) {
                    this.fixedPositions.add(p);
                }
                return this;
            }

            public Builder fillWeight(double weight) {
                this.fillWeight = Math.max(0, weight);
                return this;
            }

            public Builder required(boolean required) {
                this.required = required;
                return this;
            }

            public Builder addFilter(QueueFilter filter) {
                this.filters.add(filter);
                return this;
            }

            public QueueV2 build() {
                return new QueueV2(name, type, candidates, minGap, maxInserts,
                        fixedPositions, fillWeight, required, filters);
            }
        }
    }

    /**
     * Queue filter for business rules.
     */
    public interface QueueFilter {
        /**
         * Check if a candidate can be inserted at a position.
         */
        boolean canInsert(RequestContext ctx, Candidate candidate, int position, List<Candidate> currentSlots);

        default String reason() { return ""; }
    }

    /**
     * Frequency cap for ad control.
     */
    public static class FrequencyCap implements QueueFilter {
        private final Map<String, AtomicLong> dailyImpressions = new ConcurrentHashMap<>();
        private final Map<String, AtomicLong> hourlyImpressions = new ConcurrentHashMap<>();
        private final int dailyLimit;
        private final int hourlyLimit;

        public FrequencyCap(int dailyLimit, int hourlyLimit) {
            this.dailyLimit = dailyLimit;
            this.hourlyLimit = hourlyLimit;
        }

        @Override
        public boolean canInsert(RequestContext ctx, Candidate candidate, int position, List<Candidate> currentSlots) {
            String itemId = candidate.itemId();
            long now = System.currentTimeMillis();

            // Clean old entries periodically
            cleanupOldEntries(now);

            AtomicLong daily = dailyImpressions.computeIfAbsent(itemId, k -> new AtomicLong(0));
            AtomicLong hourly = hourlyImpressions.computeIfAbsent(itemId, k -> new AtomicLong(0));

            return daily.get() < dailyLimit && hourly.get() < hourlyLimit;
        }

        public void recordImpression(String itemId) {
            long now = System.currentTimeMillis();
            dailyImpressions.computeIfAbsent(itemId, k -> new AtomicLong(0)).incrementAndGet();
            hourlyImpressions.computeIfAbsent(itemId + "_hour", k -> new AtomicLong(now)).incrementAndGet();
        }

        private void cleanupOldEntries(long now) {
            long oneHourAgo = now - 3600000;
            long oneDayAgo = now - 86400000;
            hourlyImpressions.entrySet().removeIf(e -> e.getValue().get() < oneHourAgo);
            dailyImpressions.entrySet().removeIf(e -> e.getValue().get() < oneDayAgo);
        }

        @Override
        public String reason() {
            return "frequency_cap";
        }
    }

    /**
     * Safety filter for content policy.
     */
    public static class SafetyFilter implements QueueFilter {
        private final Set<String> blockedCategories = ConcurrentHashMap.newKeySet();
        private final Set<String> blockedItemIds = ConcurrentHashMap.newKeySet();
        private final Predicate<Candidate> customPredicate;

        public SafetyFilter(Predicate<Candidate> customPredicate) {
            this.customPredicate = customPredicate;
        }

        @Override
        public boolean canInsert(RequestContext ctx, Candidate candidate, int position, List<Candidate> currentSlots) {
            // Check blocked categories
            String category = candidate.tag("category");
            if (category != null && blockedCategories.contains(category)) {
                return false;
            }

            // Check blocked items
            if (blockedItemIds.contains(candidate.itemId())) {
                return false;
            }

            // Custom predicate
            if (customPredicate != null && !customPredicate.test(candidate)) {
                return false;
            }

            return true;
        }

        public void blockCategory(String category) {
            blockedCategories.add(category);
        }

        public void blockItem(String itemId) {
            blockedItemIds.add(itemId);
        }

        public void unblockCategory(String category) {
            blockedCategories.remove(category);
        }

        public void unblockItem(String itemId) {
            blockedItemIds.remove(itemId);
        }

        @Override
        public String reason() {
            return "safety_filter";
        }
    }

    /**
     * Budget pacing controller for ads.
     */
    public static class BudgetPacer {
        private final Map<String, BudgetState> budgets = new ConcurrentHashMap<>();

        public static class BudgetState {
            private final double totalBudget;
            private final long startTime;
            private final long endTime;
            private final AtomicLong spent = new AtomicLong(0);
            private volatile boolean exhausted = false;

            public BudgetState(double totalBudget, long startTime, long endTime) {
                this.totalBudget = totalBudget;
                this.startTime = startTime;
                this.endTime = endTime;
            }

            public double remainingBudget() {
                return Math.max(0, totalBudget - spent.get());
            }

            public double spendRate() {
                long elapsed = System.currentTimeMillis() - startTime;
                if (elapsed <= 0) return 0;
                return spent.get() / (elapsed / 3600000.0); // per hour
            }

            public boolean canSpend(double amount) {
                if (exhausted) return false;
                long currentSpent = spent.get();
                if (currentSpent + (long) amount > totalBudget) {
                    exhausted = true;
                    return false;
                }
                return true;
            }

            public void recordSpend(double amount) {
                spent.addAndGet((long) amount);
                if (spent.get() >= totalBudget) {
                    exhausted = true;
                }
            }

            public boolean isExhausted() {
                return exhausted || spent.get() >= totalBudget;
            }
        }

        public void setBudget(String campaignId, double totalBudget, long startTime, long endTime) {
            budgets.put(campaignId, new BudgetState(totalBudget, startTime, endTime));
        }

        public boolean canServe(String campaignId, double cost) {
            BudgetState state = budgets.get(campaignId);
            if (state == null) return true;
            return state.canSpend(cost);
        }

        public void recordServe(String campaignId, double cost) {
            BudgetState state = budgets.get(campaignId);
            if (state != null) {
                state.recordSpend(cost);
            }
        }

        public double remainingBudget(String campaignId) {
            BudgetState state = budgets.get(campaignId);
            return state != null ? state.remainingBudget() : 0;
        }

        public boolean isExhausted(String campaignId) {
            BudgetState state = budgets.get(campaignId);
            return state != null && state.isExhausted();
        }
    }

    /**
     * Blending strategy.
     */
    public enum BlendStrategy {
        FIXED_INTERVAL,     // Fixed position intervals
        WEIGHTED_ROUND_ROBIN, // Score-weighted round-robin
        PRIORITY_FILL,     // Fill by priority order
        OPTIMAL             // Optimization-based blending
    }

    private final int defaultPageSize;
    private final BlendStrategy defaultStrategy;
    private final Map<String, QueueV2> queues;
    private final BudgetPacer budgetPacer;
    private final FrequencyCap frequencyCap;
    private final SafetyFilter safetyFilter;
    private final ExecutorService executor;
    private final boolean ownsExecutor;

    public MixRankStageV2() {
        this(20, BlendStrategy.WEIGHTED_ROUND_ROBIN, null, null, null);
    }

    public MixRankStageV2(int defaultPageSize, BlendStrategy strategy,
                         BudgetPacer budgetPacer, FrequencyCap frequencyCap,
                         SafetyFilter safetyFilter) {
        this.defaultPageSize = defaultPageSize;
        this.defaultStrategy = strategy;
        this.queues = new LinkedHashMap<>();
        this.budgetPacer = budgetPacer != null ? budgetPacer : new BudgetPacer();
        this.frequencyCap = frequencyCap != null ? frequencyCap : new FrequencyCap(Integer.MAX_VALUE, Integer.MAX_VALUE);
        this.safetyFilter = safetyFilter != null ? safetyFilter : new SafetyFilter(c -> true);
        this.executor = null;
        this.ownsExecutor = false;
    }

    public static Builder builder() {
        return new Builder();
    }

    public void addQueue(QueueV2 queue) {
        queues.put(queue.name, queue);
    }

    public void removeQueue(String name) {
        queues.remove(name);
    }

    public QueueV2 getQueue(String name) {
        return queues.get(name);
    }

    @Override
    public String name() {
        return "mix";
    }

    @Override
    public StageResult execute(RequestContext ctx, List<Candidate> input) {
        long t0 = System.currentTimeMillis();

        int pageSize = ctx.expParamInt("mix.page_size", defaultPageSize);
        String strategyStr = ctx.expParam("mix.strategy", defaultStrategy.name());
        BlendStrategy strategy;
        try {
            strategy = BlendStrategy.valueOf(strategyStr);
        } catch (IllegalArgumentException e) {
            strategy = defaultStrategy;
        }

        // Get queues from context if provided, otherwise use registered queues
        List<QueueV2> queueList = getQueuesFromContext(ctx);
        if (queueList == null || queueList.isEmpty()) {
            // Fall back to single organic queue
            QueueV2 organic = QueueV2.organic("organic", input != null ? input : List.of());
            queueList = List.of(organic);
        }

        List<Candidate> result;
        switch (strategy) {
            case FIXED_INTERVAL:
                result = blendFixedInterval(ctx, queueList, pageSize);
                break;
            case WEIGHTED_ROUND_ROBIN:
                result = blendWeightedRoundRobin(ctx, queueList, pageSize);
                break;
            case PRIORITY_FILL:
                result = blendPriorityFill(ctx, queueList, pageSize);
                break;
            case OPTIMAL:
                result = blendOptimal(ctx, queueList, pageSize);
                break;
            default:
                result = blendWeightedRoundRobin(ctx, queueList, pageSize);
        }

        renumber(result);

        // Record impressions for frequency cap
        for (Candidate c : result) {
            if ("ad".equals(c.tag("mix_queue"))) {
                frequencyCap.recordImpression(c.itemId());
                budgetPacer.recordServe(c.tag("campaign_id"), c.getScore("cost", 0));
            }
        }

        return StageResult.ok(name(), result, System.currentTimeMillis() - t0);
    }

    private List<QueueV2> getQueuesFromContext(RequestContext ctx) {
        // Check if queues are passed via context
        Object queuesObj = ctx.getAttribute("_queues");
        if (queuesObj instanceof List) {
            @SuppressWarnings("unchecked")
            List<QueueV2> result = (List<QueueV2>) queuesObj;
            return result;
        }

        // Use registered queues
        return new ArrayList<>(queues.values());
    }

    private List<Candidate> blendFixedInterval(RequestContext ctx, List<QueueV2> queues, int pageSize) {
        Candidate[] slots = new Candidate[pageSize];
        Set<String> used = new HashSet<>();
        Map<String, Integer> inserted = new LinkedHashMap<>();
        Map<String, Integer> lastPos = new LinkedHashMap<>();
        Map<String, Integer> cursor = new LinkedHashMap<>();

        for (QueueV2 q : queues) {
            inserted.put(q.name, 0);
            lastPos.put(q.name, -q.minGap - 1);
            cursor.put(q.name, 0);
        }

        // Fixed positions first
        for (QueueV2 q : queues) {
            for (int pos : q.fixedPositions) {
                if (pos < 0 || pos >= pageSize || slots[pos] != null) continue;
                if (inserted.get(q.name) >= q.maxInserts) continue;

                Candidate next = nextUnused(q, cursor, used);
                if (next == null || !canInsert(ctx, next, pos, Arrays.asList(slots))) continue;

                Candidate copy = next.copy();
                copy.tag("mix_queue", q.name);
                copy.tag("queue_type", q.type.name());
                slots[pos] = copy;
                used.add(copy.itemId());
                inserted.merge(q.name, 1, Integer::sum);
                lastPos.put(q.name, pos);
            }
        }

        // Fill remaining by interval
        for (int pos = 0; pos < pageSize; pos++) {
            if (slots[pos] != null) continue;

            for (QueueV2 q : queues) {
                if (inserted.get(q.name) >= q.maxInserts) continue;
                int last = lastPos.get(q.name);
                if (pos - last <= q.minGap) continue;

                Candidate next = nextUnused(q, cursor, used);
                if (next == null || !canInsert(ctx, next, pos, Arrays.asList(slots))) continue;

                Candidate copy = next.copy();
                copy.tag("mix_queue", q.name);
                copy.tag("queue_type", q.type.name());
                slots[pos] = copy;
                used.add(copy.itemId());
                inserted.merge(q.name, 1, Integer::sum);
                lastPos.put(q.name, pos);
                break;
            }
        }

        List<Candidate> result = new ArrayList<>();
        for (Candidate c : slots) {
            if (c != null) result.add(c);
        }
        return result;
    }

    private List<Candidate> blendWeightedRoundRobin(RequestContext ctx, List<QueueV2> queues, int pageSize) {
        Candidate[] slots = new Candidate[pageSize];
        Set<String> used = new HashSet<>();
        Map<String, Double> weights = new LinkedHashMap<>();
        Map<String, Double> accumWeights = new LinkedHashMap<>();
        Map<String, Integer> inserted = new LinkedHashMap<>();
        Map<String, Integer> cursor = new LinkedHashMap<>();
        Map<String, Candidate> lastPicked = new LinkedHashMap<>();

        double totalWeight = 0;
        for (QueueV2 q : queues) {
            double w = q.fillWeight;
            weights.put(q.name, w);
            totalWeight += w;
            inserted.put(q.name, 0);
            cursor.put(q.name, 0);
        }

        // Accumulated weights for weighted selection
        double acc = 0;
        for (QueueV2 q : queues) {
            acc += weights.get(q.name);
            accumWeights.put(q.name, acc);
        }

        // Round-robin with weights
        for (int pos = 0; pos < pageSize; pos++) {
            boolean placed = false;

            // Try to place from queues in weighted order
            for (QueueV2 q : queues) {
                if (inserted.get(q.name) >= q.maxInserts) continue;
                if (q.required && inserted.get(q.name) == 0) {
                    // Required queue - find any available
                    Candidate next = nextUnused(q, cursor, used);
                    if (next != null && canInsert(ctx, next, pos, Arrays.asList(slots))) {
                        Candidate copy = next.copy();
                        copy.tag("mix_queue", q.name);
                        copy.tag("queue_type", q.type.name());
                        slots[pos] = copy;
                        used.add(copy.itemId());
                        inserted.merge(q.name, 1, Integer::sum);
                        lastPicked.put(q.name, copy);
                        placed = true;
                        break;
                    }
                }
            }

            if (!placed) {
                // Weight-proportional selection
                double r = Math.random() * totalWeight;
                for (QueueV2 q : queues) {
                    if (inserted.get(q.name) >= q.maxInserts) continue;
                    if (r < accumWeights.get(q.name)) {
                        Candidate next = nextUnused(q, cursor, used);
                        if (next != null && canInsert(ctx, next, pos, Arrays.asList(slots))) {
                            Candidate copy = next.copy();
                            copy.tag("mix_queue", q.name);
                            copy.tag("queue_type", q.type.name());
                            slots[pos] = copy;
                            used.add(copy.itemId());
                            inserted.merge(q.name, 1, Integer::sum);
                            lastPicked.put(q.name, copy);
                        }
                        break;
                    }
                }
            }
        }

        List<Candidate> result = new ArrayList<>();
        for (Candidate c : slots) {
            if (c != null) result.add(c);
        }
        return result;
    }

    private List<Candidate> blendPriorityFill(RequestContext ctx, List<QueueV2> queues, int pageSize) {
        Candidate[] slots = new Candidate[pageSize];
        Set<String> used = new HashSet<>();

        // Sort queues by priority
        List<QueueV2> sorted = new ArrayList<>(queues);
        sorted.sort((a, b) -> {
            // Required first, then by fill weight
            if (a.required != b.required) return a.required ? -1 : 1;
            return Double.compare(b.fillWeight, a.fillWeight);
        });

        Map<String, Integer> cursor = new LinkedHashMap<>();
        for (QueueV2 q : sorted) {
            cursor.put(q.name, 0);
        }

        // Fill each queue's fixed positions
        for (QueueV2 q : sorted) {
            for (int pos : q.fixedPositions) {
                if (pos < 0 || pos >= pageSize || slots[pos] != null) continue;
                Candidate next = nextUnused(q, cursor, used);
                if (next != null && canInsert(ctx, next, pos, Arrays.asList(slots))) {
                    Candidate copy = next.copy();
                    copy.tag("mix_queue", q.name);
                    slots[pos] = copy;
                    used.add(copy.itemId());
                }
            }
        }

        // Fill remaining slots
        for (int pos = 0; pos < pageSize; pos++) {
            if (slots[pos] != null) continue;

            for (QueueV2 q : sorted) {
                if (inserted(q.name, slots) >= q.maxInserts) continue;
                Candidate next = nextUnused(q, cursor, used);
                if (next != null && canInsert(ctx, next, pos, Arrays.asList(slots))) {
                    Candidate copy = next.copy();
                    copy.tag("mix_queue", q.name);
                    copy.tag("queue_type", q.type.name());
                    slots[pos] = copy;
                    used.add(copy.itemId());
                    break;
                }
            }
        }

        List<Candidate> result = new ArrayList<>();
        for (Candidate c : slots) {
            if (c != null) result.add(c);
        }
        return result;
    }

    private List<Candidate> blendOptimal(RequestContext ctx, List<QueueV2> queues, int pageSize) {
        // Simplified optimal blending: greedy with quality score
        Candidate[] slots = new Candidate[pageSize];
        Set<String> used = new HashSet<>();
        Map<String, Integer> cursor = new LinkedHashMap<>();

        for (QueueV2 q : queues) {
            cursor.put(q.name, 0);
        }

        for (int pos = 0; pos < pageSize; pos++) {
            Candidate best = null;
            QueueV2 bestQueue = null;
            double bestScore = Double.NEGATIVE_INFINITY;

            for (QueueV2 q : queues) {
                if (inserted(q.name, slots) >= q.maxInserts) continue;

                int idx = cursor.get(q.name);
                while (idx < q.candidates.size()) {
                    Candidate c = q.candidates.get(idx);
                    if (used.contains(c.itemId())) {
                        idx++;
                        continue;
                    }

                    if (!canInsert(ctx, c, pos, Arrays.asList(slots))) {
                        idx++;
                        continue;
                    }

                    // Score: relevance * weight * diversity factor
                    double score = c.score() * q.fillWeight;
                    if (pos > 0 && slots[pos - 1] != null) {
                        // Penalize same category
                        String cat1 = c.tag("category");
                        String cat2 = slots[pos - 1].tag("category");
                        if (cat1 != null && cat1.equals(cat2)) {
                            score *= 0.8;
                        }
                    }

                    if (score > bestScore) {
                        bestScore = score;
                        best = c;
                        bestQueue = q;
                        cursor.put(q.name, idx + 1);
                    }
                    break;
                }
            }

            if (best != null) {
                Candidate copy = best.copy();
                copy.tag("mix_queue", bestQueue.name);
                copy.tag("queue_type", bestQueue.type.name());
                slots[pos] = copy;
                used.add(copy.itemId());
            }
        }

        List<Candidate> result = new ArrayList<>();
        for (Candidate c : slots) {
            if (c != null) result.add(c);
        }
        return result;
    }

    private int inserted(String queueName, Candidate[] slots) {
        int count = 0;
        for (Candidate c : slots) {
            if (c != null && queueName.equals(c.tag("mix_queue"))) {
                count++;
            }
        }
        return count;
    }

    private boolean canInsert(RequestContext ctx, Candidate candidate, int position, List<Candidate> slots) {
        // Apply all filters
        for (QueueFilter filter : safetyFilter != null ? List.of(safetyFilter, frequencyCap) : List.of(frequencyCap)) {
            if (!filter.canInsert(ctx, candidate, position, slots)) {
                return false;
            }
        }
        return true;
    }

    private Candidate nextUnused(QueueV2 q, Map<String, Integer> cursor, Set<String> used) {
        int i = cursor.getOrDefault(q.name, 0);
        while (i < q.candidates.size()) {
            Candidate c = q.candidates.get(i);
            i++;
            cursor.put(q.name, i);
            if (!used.contains(c.itemId())) {
                return c;
            }
        }
        cursor.put(q.name, i);
        return null;
    }

    private void renumber(List<Candidate> result) {
        for (int i = 0; i < result.size(); i++) {
            result.get(i).rank(i);
        }
    }

    public BudgetPacer budgetPacer() {
        return budgetPacer;
    }

    public FrequencyCap frequencyCap() {
        return frequencyCap;
    }

    public SafetyFilter safetyFilter() {
        return safetyFilter;
    }

    public static final class Builder {
        private int defaultPageSize = 20;
        private BlendStrategy strategy = BlendStrategy.WEIGHTED_ROUND_ROBIN;
        private final List<QueueV2> queues = new ArrayList<>();
        private BudgetPacer budgetPacer;
        private FrequencyCap frequencyCap;
        private SafetyFilter safetyFilter;

        public Builder defaultPageSize(int pageSize) {
            this.defaultPageSize = pageSize;
            return this;
        }

        public Builder strategy(BlendStrategy strategy) {
            this.strategy = strategy;
            return this;
        }

        public Builder addQueue(QueueV2 queue) {
            this.queues.add(queue);
            return this;
        }

        public Builder budgetPacer(BudgetPacer budgetPacer) {
            this.budgetPacer = budgetPacer;
            return this;
        }

        public Builder frequencyCap(int dailyLimit, int hourlyLimit) {
            this.frequencyCap = new FrequencyCap(dailyLimit, hourlyLimit);
            return this;
        }

        public Builder safetyFilter(SafetyFilter safetyFilter) {
            this.safetyFilter = safetyFilter;
            return this;
        }

        public MixRankStageV2 build() {
            MixRankStageV2 stage = new MixRankStageV2(defaultPageSize, strategy,
                    budgetPacer, frequencyCap, safetyFilter);
            for (QueueV2 q : queues) {
                stage.addQueue(q);
            }
            return stage;
        }
    }
}
