"""
Attempt 2 summary: inventory-only reward r_t = -lambda|q_t|^p (p=2 or 3 tried) to reduce variance. State adds inventory bins to spread; actions include no-quote plus biased long/short and symmetric quotes. Environment adds Value, ZI, HBL, and Noise agents. The agent quickly learns to flatten inventory then stops quoting; rewards become quasi-stationary around zero, actions converge to no-trade, and PnL beats the baseline MM. Downside: liquidity collapses as spreads widen once the agent stops posting.
"""

from traders.BaseTrader import BaseTrader
import pandas as pd
import numpy as np


class RLMM2(BaseTrader):
    """
    Minimal tabular Q-learning market maker.

    - State: binned inventory and spread.
    - Action: no-quote, or asymmetric tick offsets (bid, ask) chosen from:
      [(None, None), (10, 20), (20, 10), (20, 20)].
    - Reward: inventory penalty only (no PnL term): -inventory_penalty * inventory^2.
      A bonus can be given to the full no-quote action to encourage patience.

    This is intentionally simple for baseline benchmarking. It does on-policy learning inside the
    simulation (no replay).
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 wake_up_freq='10s',
                 base_size=100,
                 offsets=(1, 2, 3),
                 epsilon=0.1,
                 alpha=0.1,
                 gamma=0.95,
                 inventory_clip=1000,
                 spread_clip=40,
                 inventory_bin=100,
                 spread_bin=5,
                 inventory_penalty=1.0,
                 inventory_limit=100,
                 epsilon_half_life_hours=5,
                 no_quote_bonus=50.0,
                 log_orders=False,
                 random_state=None,
                 actions=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.wake_up_freq = wake_up_freq
        self.base_size = base_size
        # Fixed action space: list of (bid_offset, ask_offset); None means no quote on that side.
        default_actions = [
            (None, None),
            (10, 20),
            (20, 10),
            (20, 20),
        ]
        self.actions = list(default_actions if actions is None else actions)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.inventory_clip = inventory_clip
        self.spread_clip = spread_clip
        self.inventory_bin = inventory_bin
        self.spread_bin = spread_bin
        self.inventory_penalty = inventory_penalty
        self.inventory_limit = inventory_limit
        self.no_quote_bonus = no_quote_bonus
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
        self.cum_reward = 0

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

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
            state = self._discretize_state(spread, self.getHoldings(self.symbol))

            # reward from last action
            reward = -self.inventory_penalty * (self.getHoldings(self.symbol) ** 2)
            if self.last_action is not None:
                last_offsets = self.actions[self.last_action]
                if last_offsets == (None, None):
                    reward += self.no_quote_bonus
            self.cum_reward += reward

            if self.last_state is not None and self.last_action is not None:
                self._update_q(self.last_state, self.last_action, reward, state)

            action_idx = self._choose_action(state)
            greedy_idx = self._greedy_action(state)
            action = self.actions[action_idx]
            greedy_action = self.actions[greedy_idx]
            self._place_quotes(mid, action)

            # Log state for downstream analysis
            self._log_state(currentTime, mid, spread, state, action_idx, greedy_idx, reward, action, greedy_action)

            self.last_state = state
            self.last_action = action_idx
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'

    def _place_quotes(self, mid, action):
        if action is None:
            return
        bid_offset, ask_offset = action
        if bid_offset is not None and self.getHoldings(self.symbol) < self.inventory_limit:
            bid_price = int(mid - bid_offset)
            self.placeLimitOrder(self.symbol, self.base_size, True, bid_price)
        if ask_offset is not None and self.getHoldings(self.symbol) > -self.inventory_limit:
            ask_price = int(mid + ask_offset)
            self.placeLimitOrder(self.symbol, self.base_size, False, ask_price)

    def _discretize_state(self, spread, inventory):
        inv = int(np.clip(inventory, -self.inventory_clip, self.inventory_clip))
        spr = int(np.clip(spread, 0, self.spread_clip))
        inv_bin = inv // self.inventory_bin
        spr_bin = spr // self.spread_bin
        return (inv_bin, spr_bin)

    def _choose_action(self, state):
        self.step_count += 1
        eps = self.epsilon * (0.5 ** (self.step_count / self.epsilon_decay_steps))
        if self.random_state.rand() < eps:
            return self.random_state.randint(0, len(self.actions))
        qs = [self.q.get((state, a), 0.0) for a in range(len(self.actions))]
        return int(np.argmax(qs))

    def _greedy_action(self, state):
        qs = [self.q.get((state, a), 0.0) for a in range(len(self.actions))]
        return int(np.argmax(qs))

    def _update_q(self, state, action, reward, next_state):
        key = (state, action)
        current_q = self.q.get(key, 0.0)
        next_q = max([self.q.get((next_state, a), 0.0) for a in range(len(self.actions))], default=0.0)
        target = reward + self.gamma * next_q
        self.q[key] = current_q + self.alpha * (target - current_q)

    def _mark_to_market(self, mid):
        # If mid is missing, use last_trade as a fallback
        if mid is None:
            mid = self.last_trade.get(self.symbol, 0) or 0
        return self.holdings['CASH'] + self.getHoldings(self.symbol) * mid

    def _log_state(self, currentTime, mid, spread, state, action_idx, greedy_idx, reward, action, greedy_action):
        inv_bin, spr_bin = state
        qs = [self.q.get((state, a), 0.0) for a in range(len(self.actions))]
        self.logEvent('STATE', {
            'time': currentTime,
            'mid': mid,
            'spread': spread,
            'inventory': self.getHoldings(self.symbol),
            'cash': self.holdings['CASH'],
            'mtm': self._mark_to_market(mid),
            'last_action': action_idx,
            'greedy_action': greedy_idx,
            'action_offsets': action,
            'greedy_offsets': greedy_action,
            'inventory_bin': inv_bin,
            'spread_bin': spr_bin,
            'reward': reward,
            'cum_reward': self.cum_reward,
            'q_max': max(qs) if qs else 0.0,
        })

    def cancelOrders(self):
        for _, order in list(self.orders.items()):
            self.cancelOrder(order)

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)
