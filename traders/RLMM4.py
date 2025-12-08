"""
Attempt 4 summary: add informed-flow awareness via state features (order-flow imbalance over recent trades and depth imbalance at the best quotes) and penalize adverse selection with a short-horizon markout term. Reward is incremental PnL minus markout penalty (scaled by alpha/markout_weight) minus inventory penalty; state still bins inventory and spread. Actions remain asymmetric/no-quote similar to RLMM2. Goal: mitigate being picked off by informed flow while managing inventory.
"""

from traders.BaseTrader import BaseTrader
import pandas as pd
import numpy as np


class RLMM4(BaseTrader):
    """
    Tabular Q-learning market maker with simple informed-flow awareness.

    - State: binned inventory, spread, recent order-flow imbalance, and top-of-book depth imbalance.
    - Action: RLMM2-style asymmetric bid/ask offsets with an explicit no-quote option.
    - Reward: incremental PnL minus inventory penalty and a short-horizon markout penalty to capture adverse selection.

    This is intentionally simple for baseline benchmarking. It does on-policy learning inside the
    simulation (no replay).
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 wake_up_freq='10s',
                 base_size=100,
                 offsets=(1, 2, 3),
                 actions=None,
                 epsilon=0.1,
                 alpha=0.1,
                 gamma=0.95,
                 ofi_lookback=50,
                 ofi_bin_size=100,
                 ofi_clip=5000,
                 depth_imbalance_bins=None,
                 markout_horizon='30s',
                 markout_weight=0.1,
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
        self.ofi_lookback = ofi_lookback
        self.ofi_bin_size = ofi_bin_size
        self.ofi_clip = ofi_clip
        self.depth_imbalance_bins = depth_imbalance_bins or [-0.5, -0.1, 0.1, 0.5]
        try:
            self.markout_horizon = pd.Timedelta(markout_horizon)
        except Exception:
            self.markout_horizon = pd.Timedelta('30s')
        self.markout_weight = markout_weight
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
        self.pending_markouts = []

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        # Initialize last_pnl at start (no book yet)
        self.last_pnl = self._mark_to_market(None)

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if can_trade:
            self.cancelOrders()
            self.getCurrentSpread(self.symbol, depth=1)
            self.getOrderStream(self.symbol, length=self.ofi_lookback)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)
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
            depth_imbalance = self._compute_depth_imbalance(bid_vol, ask_vol)
            ofi_val = self._compute_ofi()
            holdings = self.getHoldings(self.symbol)
            state = self._discretize_state(spread, holdings, ofi_val, depth_imbalance)

            # reward from last action
            pnl = self._mark_to_market(mid)
            spread_pnl = pnl - self.last_pnl
            markout_penalty, evaluated_trades = self._process_markouts(mid, currentTime)
            reward = spread_pnl - self.markout_weight * markout_penalty - self.inventory_penalty * (holdings ** 2)
            self.last_pnl = pnl
            self.cum_reward += reward

            if self.last_state is not None and self.last_action is not None:
                self._update_q(self.last_state, self.last_action, reward, state)

            action_idx = self._choose_action(state)
            greedy_idx = self._greedy_action(state)
            action = self.actions[action_idx]
            greedy_action = self.actions[greedy_idx]
            self._place_quotes(mid, action)

            # Log state for downstream analysis
            self._log_state(currentTime, mid, spread, state, action_idx, greedy_idx, reward,
                            spread_pnl, ofi_val, depth_imbalance, markout_penalty, evaluated_trades,
                            action, greedy_action)

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

    def _discretize_state(self, spread, inventory, ofi_value, depth_imbalance):
        inv = int(np.clip(inventory, -self.inventory_clip, self.inventory_clip))
        spr = int(np.clip(spread, 0, self.spread_clip))
        inv_bin = inv // self.inventory_bin
        spr_bin = spr // self.spread_bin
        ofi_bin = self._bin_ofi(ofi_value)
        depth_bin = self._bin_depth_imbalance(depth_imbalance)
        return (inv_bin, spr_bin, ofi_bin, depth_bin)

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

    def _log_state(self, currentTime, mid, spread, state, action_idx, greedy_idx, reward,
                   spread_pnl, ofi_val, depth_imbalance, markout_penalty, evaluated_trades,
                   action, greedy_action):
        inv_bin, spr_bin, ofi_bin, depth_bin = state
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
            'inventory_bin': inv_bin,
            'spread_bin': spr_bin,
            'ofi': ofi_val,
            'ofi_bin': ofi_bin,
            'depth_imbalance': depth_imbalance,
            'depth_bin': depth_bin,
            'markout_penalty': markout_penalty,
            'markout_weighted': self.markout_weight * markout_penalty,
            'markout_pending': len(self.pending_markouts),
            'markout_evaluated': evaluated_trades,
            'spread_pnl': spread_pnl,
            'action_offsets': action,
            'greedy_offsets': greedy_action,
            'reward': reward,
            'cum_reward': self.cum_reward,
            'q_max': max(qs) if qs else 0.0,
        })

    def _compute_depth_imbalance(self, bid_vol, ask_vol):
        depth_total = bid_vol + ask_vol
        if depth_total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / depth_total

    def _bin_depth_imbalance(self, imbalance):
        # Small, discrete buckets to keep tabular state compact
        return int(np.digitize([imbalance], self.depth_imbalance_bins)[0])

    def _compute_ofi(self):
        orders = self.stream_history.get(self.symbol, [])
        if not orders:
            return 0
        signed_trades = []
        for batch in orders:
            for _, details in batch.items():
                sign = 1 if details.get('is_buy_order') else -1
                for ts, qty in details.get('transactions', []):
                    signed_trades.append((ts, sign * qty))
        if not signed_trades:
            return 0
        signed_trades = sorted(signed_trades, key=lambda x: x[0])
        net = sum(v for _, v in signed_trades[-self.ofi_lookback:])
        return net

    def _bin_ofi(self, ofi_value):
        clipped = int(np.clip(ofi_value, -self.ofi_clip, self.ofi_clip))
        if self.ofi_bin_size <= 0:
            return clipped
        magnitude = abs(clipped) // self.ofi_bin_size
        return int(np.sign(clipped) * magnitude)

    def _process_markouts(self, mid, current_time):
        if mid is None:
            return 0.0, 0
        matured = []
        still_pending = []
        for trade in self.pending_markouts:
            if trade['eval_time'] <= current_time:
                matured.append(trade)
            else:
                still_pending.append(trade)
        self.pending_markouts = still_pending
        penalty = 0.0
        for trade in matured:
            move = mid - trade['price']
            signed_move = move if trade['is_buy'] else -move
            penalty += -signed_move * trade['qty']
        return penalty, len(matured)

    def orderExecuted(self, order):
        super().orderExecuted(order)
        eval_time = self.currentTime + self.markout_horizon
        fill_price = getattr(order, 'fill_price', None) or order.limit_price
        self.pending_markouts.append({
            'eval_time': eval_time,
            'is_buy': order.is_buy_order,
            'price': fill_price,
            'qty': order.quantity
        })

    def cancelOrders(self):
        for _, order in list(self.orders.items()):
            self.cancelOrder(order)

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)
