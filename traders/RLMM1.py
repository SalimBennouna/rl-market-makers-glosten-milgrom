"""
Attempt 1 summary: pure Noise-only market where inventory risk is negligible. Reward is mark-to-market PnL delta; state is spread-only (three bins in the writeup). Actions are symmetric quote offsets {±2, ±15, ±50}. The agent quickly learns to pick the widest spread, producing steady, low-variance PnL and near-zero inventory; Q-values grow then plateau with the widest offset dominating. When Zero-Intelligence makers are added, PnL-only rewards become very noisy and learning turns unstable.
"""

from traders.BaseTrader import BaseTrader
from pathlib import Path
import pandas as pd
import numpy as np
import pickle


class RLMM1(BaseTrader):
    """
    Minimal tabular Q-learning market maker.

    - State: binned spread only.
    - Action: choose a single tick offset from the mid for bid/ask quotes (symmetric).
    - Reward: incremental mark-to-market PnL.

    This is intentionally simple for baseline benchmarking. It does on-policy learning inside the
    simulation (no replay).
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 wake_up_freq='10s',
                 base_size=100,
                 offsets=(3, 10, 20),
                 epsilon=0.1,
                 alpha=0.1,
                 gamma=0.95,
                 inventory_clip=1000,
                 spread_clip=50,
                 inventory_bin=100,
                 spread_bin=1,
                 inventory_penalty=0.0,
                 inventory_limit=100,
                 epsilon_half_life_hours=5,
                 log_orders=False,
                 random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.wake_up_freq = wake_up_freq
        self.base_size = base_size
        self.offsets = list(offsets)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.inventory_clip = inventory_clip
        self.spread_clip = spread_clip
        self.inventory_bin = inventory_bin
        self.spread_bin = spread_bin
        self.inventory_penalty = inventory_penalty
        self.inventory_limit = inventory_limit
        self.state = 'AWAITING_WAKEUP'
        self.step_count = 0
        # decay steps based on wake_up_freq and half-life
        try:
            wake_seconds = pd.Timedelta(self.wake_up_freq).total_seconds()
        except Exception:
            wake_seconds = 1.0
        half_life_seconds = epsilon_half_life_hours * 3600.0
        self.epsilon_decay_steps = max(1, int(np.ceil(half_life_seconds / wake_seconds)))

        # Q-table: dict mapping (state, action_idx) -> value
        self.q = {}
        self.last_state = None
        self.last_action = None
        self.last_pnl = 0
        self.cum_reward = 0

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        # Initialize last_pnl at start (no book yet)
        self.last_pnl = self._mark_to_market(None)

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if can_trade:
            self.cancelOrders()
            self.getCurrentSpread(self.symbol, depth=1)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
            if not (bid and ask):
                # Fall back to the latest known trade (or single-sided quote) so we can seed the book
                last = self.last_trade.get(self.symbol)
                if last is not None:
                    mid = last
                    spread = 0
                elif bid is not None:
                    mid = bid
                    spread = 0
                elif ask is not None:
                    mid = ask
                    spread = 0
                else:
                    self.setWakeup(currentTime + self.getWakeFrequency())
                    return
            else:
                mid = (bid + ask) / 2
                spread = ask - bid
            state = self._discretize_state(spread)

            # reward from last action
            pnl = self._mark_to_market(mid)
            reward = pnl - self.last_pnl
            self.last_pnl = pnl
            self.cum_reward += reward

            if self.last_state is not None and self.last_action is not None:
                self._update_q(self.last_state, self.last_action, reward, state)

            action_idx = self._choose_action(state)
            greedy_idx = self._greedy_action(state)
            self._place_quotes(mid, self.offsets[action_idx])

            # Log state for downstream analysis
            self._log_state(currentTime, mid, spread, state, action_idx, greedy_idx, reward)

            self.last_state = state
            self.last_action = action_idx
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'

    def _place_quotes(self, mid, offset):
        bid_price = int(mid - offset)
        ask_price = int(mid + offset)
        if self.getHoldings(self.symbol) < self.inventory_limit:
            self.placeLimitOrder(self.symbol, self.base_size, True, bid_price)
        if self.getHoldings(self.symbol) > -self.inventory_limit:
            self.placeLimitOrder(self.symbol, self.base_size, False, ask_price)

    def _discretize_state(self, spread):
        spr = int(np.clip(spread, 0, self.spread_clip))
        spr_bin = spr // self.spread_bin
        return spr_bin

    def _choose_action(self, state):
        self.step_count += 1
        eps = self.epsilon * (0.5 ** (self.step_count / self.epsilon_decay_steps))
        if self.random_state.rand() < eps:
            return self.random_state.randint(0, len(self.offsets))
        qs = [self.q.get((state, a), 0.0) for a in range(len(self.offsets))]
        return int(np.argmax(qs))

    def _greedy_action(self, state):
        qs = [self.q.get((state, a), 0.0) for a in range(len(self.offsets))]
        return int(np.argmax(qs))

    def _update_q(self, state, action, reward, next_state):
        key = (state, action)
        current_q = self.q.get(key, 0.0)
        next_q = max([self.q.get((next_state, a), 0.0) for a in range(len(self.offsets))], default=0.0)
        target = reward + self.gamma * next_q
        self.q[key] = current_q + self.alpha * (target - current_q)

    def _mark_to_market(self, mid):
        # If mid is missing, use last_trade as a fallback
        if mid is None:
            mid = self.last_trade.get(self.symbol, 0) or 0
        return self.holdings['CASH'] + self.getHoldings(self.symbol) * mid

    def _log_state(self, currentTime, mid, spread, state, action_idx, greedy_idx, reward):
        spr_bin = state
        qs = [self.q.get((state, a), 0.0) for a in range(len(self.offsets))]
        self.logEvent('STATE', {
            'time': currentTime,
            'mid': mid,
            'spread': spread,
            'inventory': self.getHoldings(self.symbol),
            'cash': self.holdings['CASH'],
            'mtm': self._mark_to_market(mid),
            'last_action': action_idx,
            'greedy_action': greedy_idx,
            'inventory_bin': None,
            'spread_bin': spr_bin,
            'reward': reward,
            'cum_reward': self.cum_reward,
            'q_max': max(qs) if qs else 0.0,
            'q_values': qs,
        })

    def cancelOrders(self):
        for _, order in list(self.orders.items()):
            self.cancelOrder(order)

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)

    def kernelStopping(self):
        # Persist Q-table for offline analysis
        if self.kernel and self.q:
            log_dir = Path("log") / self.kernel.log_dir
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                q_path = log_dir / "q_table.pkl"
                with q_path.open("wb") as f:
                    pickle.dump(self.q, f)
            except Exception:
                pass
        super().kernelStopping()
