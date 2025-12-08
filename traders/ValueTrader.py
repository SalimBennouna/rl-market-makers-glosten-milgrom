from traders.BaseTrader import BaseTrader

import numpy as np
import pandas as pd


class ValueTrader(BaseTrader):
    def __init__(self, id, name, type, symbol='IBM', starting_cash=103000, sigma_n=9500,
                 r_bar=98500, kappa=0.047, sigma_s=98000,
                 lambda_a=0.0048, log_orders=False, log_to_file=True, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, log_to_file=log_to_file, random_state=random_state)
        self.symbol = symbol
        self.sigma_n = sigma_n
        self.r_bar = r_bar
        self.kappa = kappa
        self.sigma_s = sigma_s
        self.lambda_a = lambda_a
        self.trading = False
        self.state = 'AWAITING_WAKEUP'
        self.r_t = r_bar
        self.sigma_t = 0
        self.prev_wake_time = None
        self.aggr_ratio = 0.12
        self.lot_size = self.random_state.randint(1, 6)
        self.book_depth = 2

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.oracle = self.kernel.oracle

    def kernelStopping(self):
        super().kernelStopping()
        lot_count = int(round(self.getHoldings(self.symbol), -2) / 100)
        terminal_val = self.oracle.emit_view(self.symbol, self.currentTime, sigma_n=0, random_state=self.random_state)
        net_surplus = (terminal_val * lot_count + self.holdings['CASH'] - self.starting_cash) / self.starting_cash
        self.logEvent('FINAL_VALUATION', float(net_surplus), True)

    def getWakeFrequency(self):
        return pd.Timedelta(self.random_state.randint(low=0, high=100), unit='ns')

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        self.state = 'INACTIVE'
        if not self.mkt_open or not self.mkt_close:
            return
        if not self.trading:
            self.trading = True
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            return
        gap_ns = self.random_state.exponential(scale=1.0 / self.lambda_a)
        self.setWakeup(currentTime + pd.Timedelta(f'{int(round(gap_ns))}ns'))
        if self.mkt_closed and (self.symbol not in self.daily_close_price):
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
            return
        self.cancelOrders()
        if type(self) == ValueTrader:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
        else:
            self.state = 'ACTIVE'

    def updateEstimates(self):
        noisy_view = self.oracle.emit_view(self.symbol, self.currentTime, sigma_n=self.sigma_n, random_state=self.random_state)
        if self.prev_wake_time is None:
            self.prev_wake_time = self.mkt_open
        elapsed = (self.currentTime - self.prev_wake_time) / np.timedelta64(1, 'ns')
        forward_r = (1 - (1 - self.kappa) ** elapsed) * self.r_bar
        forward_r += ((1 - self.kappa) ** elapsed) * self.r_t
        forward_var = ((1 - self.kappa) ** (2 * elapsed)) * self.sigma_t
        forward_var += ((1 - (1 - self.kappa) ** (2 * elapsed)) / (1 - (1 - self.kappa) ** 2)) * self.sigma_s
        self.r_t = (self.sigma_n / (self.sigma_n + forward_var)) * forward_r
        self.r_t += (forward_var / (self.sigma_n + forward_var)) * noisy_view
        self.sigma_t = (self.sigma_n * self.sigma_t) / (self.sigma_n + self.sigma_t)
        remaining = max(0, (self.mkt_close - self.currentTime) / np.timedelta64(1, 'ns'))
        terminal_r = (1 - (1 - self.kappa) ** remaining) * self.r_bar
        terminal_r += ((1 - self.kappa) ** remaining) * self.r_t
        terminal_r = int(round(terminal_r))
        self.prev_wake_time = self.currentTime
        return terminal_r

    def placeOrder(self):
        target_val = self.updateEstimates()
        best_bid, bid_depth, best_ask, ask_depth = self.getKnownBidAsk(self.symbol)
        if best_bid and best_ask:
            mid_price = int((best_ask + best_bid) / 2)
            spread_width = abs(best_ask - best_bid)
            if self.random_state.rand() < self.aggr_ratio:
                offset = 0
            else:
                offset = self.random_state.randint(0, max(1, int(self.book_depth * spread_width)))
            if target_val < mid_price:
                direction = False
                limit_px = best_bid + offset
            else:
                direction = True
                limit_px = best_ask - offset
        else:
            direction = bool(self.random_state.randint(0, 2))
            limit_px = target_val
        self.placeLimitOrder(self.symbol, self.lot_size, direction, limit_px)

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD':
            if msg.body['msg'] == 'QUERY_SPREAD':
                if self.mkt_closed:
                    return
                self.placeOrder()
                self.state = 'AWAITING_WAKEUP'

    def cancelOrders(self):
        if not self.orders:
            return False
        for order_id, open_order in self.orders.items():
            self.cancelOrder(open_order)
        return True
