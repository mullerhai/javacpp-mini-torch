/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.rl.env;

import org.bytedeco.pytorch.rl.StepResult;

/**
 * Simplified trading environment for algorithmic trading RL.
 *
 * <p>Observations: Price change rates over a sliding window
 * <p>Actions: 0=sell, 1=hold, 2=buy
 * <p>Reward: Portfolio return rate
 *
 * <p>This is a simplified version for quick experimentation.
 * For production use, consider {@link TradingEnv}.
 *
 * @see TradingEnv for more sophisticated trading environment
 */
public class SimpleTradingEnv implements Env {

    private float[] prices;
    private int currentIndex;
    private int windowSize = 5;
    private float balance = 1000f;
    private float shares = 0;

    // Episode tracking
    private double cumulativeReward = 0.0;
    private int stepCount = 0;

    // ==================== Constants ====================

    private static final int ACTION_SPACE_SIZE = 3; // sell, hold, buy

    // ==================== Constructor ====================

    public SimpleTradingEnv(float[] prices) {
        this.prices = prices;
    }

    // ==================== Env Interface ====================

    @Override
    public StepResult reset() {
        currentIndex = windowSize;
        balance = 1000f;
        shares = 0;
        cumulativeReward = 0.0;
        stepCount = 0;
        return StepResult.builder()
                .legacyObservation(getObservation())
                .reward(0.0)
                .terminated(false)
                .build();
    }

    @Override
    public StepResult step(int action) {
        stepCount++;

        float price = prices[currentIndex];
        float prevValue = balance + shares * price;

        // Execute action
        if (action == 2 && balance >= price) { // Buy
            shares += 1;
            balance -= price;
        } else if (action == 0 && shares > 0) { // Sell
            balance += price;
            shares -= 1;
        }

        currentIndex++;
        float currentValue = balance + shares * prices[currentIndex];
        double reward = (currentValue / prevValue) - 1.0;
        cumulativeReward += reward;

        boolean done = currentIndex >= prices.length - 1;

        return StepResult.builder()
                .legacyObservation(getObservation())
                .reward(reward)
                .terminated(done)
                .build();
    }

    @Override
    public int actionSpaceSize() {
        return ACTION_SPACE_SIZE;
    }

    @Override
    public int observationDim() {
        return windowSize;
    }

    @Override
    public double episodeReturn() {
        return cumulativeReward;
    }

    @Override
    public int episodeLength() {
        return stepCount;
    }

    // ==================== Private Methods ====================

    private float[] getObservation() {
        float[] obs = new float[windowSize];
        for (int i = 0; i < windowSize; i++) {
            obs[i] = (prices[currentIndex - windowSize + i + 1] / prices[currentIndex - windowSize + i]) - 1.0f;
        }
        return obs;
    }

    // ==================== Getters ====================

    public float getBalance() {
        return balance;
    }

    public float getShares() {
        return shares;
    }

    public float getPortfolioValue() {
        return balance + shares * prices[currentIndex];
    }
}
