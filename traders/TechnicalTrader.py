from traders.BaseTrader import BaseTrader
import pandas as pd
import numpy as np


class TechnicalTrader(BaseTrader):
    def __init__(self, id, name, type, symbol='IBM', starting_cash=102500,
                 min_size=15, max_size=85, wake_up_freq='58s',
                 log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.qty_floor = min_size
        self.qty_cap = max_size
        self.order_size = self.random_state.randint(self.qty_floor, self.qty_cap)
        self.wake_gap = wake_up_freq
        self.mids = []
        self.fast_avg = []
        self.slow_avg = []
        self.log_orders = log_orders
        self.state = 'AWAITING_WAKEUP'

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_gap)

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if can_trade:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
            self.placeOrders(bid, ask)
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'

    def placeOrders(self, bid, ask):
        if bid and ask:
            self.mids.append((bid + ask) / 2)
            if len(self.mids) > 20:
                self.fast_avg.append(TechnicalTrader.ma(self.mids, n=20)[-1].round(2))
            if len(self.mids) > 50:
                self.slow_avg.append(TechnicalTrader.ma(self.mids, n=50)[-1].round(2))
            if len(self.fast_avg) > 0 and len(self.slow_avg) > 0:
                if self.fast_avg[-1] >= self.slow_avg[-1]:
                    self.placeLimitOrder(self.symbol, quantity=self.order_size, is_buy_order=True, limit_price=ask)
                else:
                    self.placeLimitOrder(self.symbol, quantity=self.order_size, is_buy_order=False, limit_price=bid)

    @staticmethod
    def ma(a, n=20):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
