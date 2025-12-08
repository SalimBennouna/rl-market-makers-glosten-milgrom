from traders.BaseTrader import BaseTrader

from math import sqrt
import numpy as np
import pandas as pd


class ZITrader(BaseTrader):
    def __init__(self, id, name, type, symbol='IBM', starting_cash=102000, sigma_n=950,
                 r_bar=99000, kappa=0.048, sigma_s=95000, q_max=11,
                 sigma_pv=4800000, R_min=1, R_max=245, eta=0.95,
                 lambda_a=0.0045, log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.sigma_n = sigma_n
        self.r_bar = r_bar
        self.kappa = kappa
        self.sigma_s = sigma_s
        self.q_max = q_max
        self.sigma_pv = sigma_pv
        self.R_min = R_min
        self.R_max = R_max
        self.eta = eta
        self.lambda_a = lambda_a
        self.trading = False
        self.state = 'AWAITING_WAKEUP'
        self.r_t = r_bar
        self.sigma_t = 0
        self.prev_wake_time = None
        self.theta = [int(x) for x in sorted(np.round(self.random_state.normal(loc=0, scale=sqrt(sigma_pv), size=(q_max * 2))).tolist(), reverse=True)]

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.oracle = self.kernel.oracle

    def kernelStopping(self):
        super().kernelStopping()
        lot_count = int(round(self.getHoldings(self.symbol), -2) / 100)
        if self.symbol != 'ETF':
            closing_fair = self.oracle.emit_view(self.symbol, self.currentTime, sigma_n=0, random_state=self.random_state)
        else:
            _, closing_fair = self.oracle.observePortfolioPrice(self.symbol, self.portfolio, self.currentTime, sigma_n=0, random_state=self.random_state)

        def pv_at(idx):
            idx = max(0, min(idx, len(self.theta) - 1))
            return self.theta[idx]

        if lot_count > 0:
            surplus = sum([pv_at(x + self.q_max - 1) for x in range(1, lot_count + 1)])
        elif lot_count < 0:
            surplus = -sum([pv_at(x + self.q_max - 1) for x in range(lot_count + 1, 1)])
        else:
            surplus = 0
        surplus += closing_fair * lot_count
        surplus += self.holdings['CASH'] - self.starting_cash
        self.logEvent('FINAL_VALUATION', surplus, True)

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
        if type(self) == ZITrader:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
        else:
            self.state = 'ACTIVE'

    def updateEstimates(self):
        noisy_view = self.oracle.emit_view(self.symbol, self.currentTime, sigma_n=self.sigma_n, random_state=self.random_state)
        lot_index = int(self.getHoldings(self.symbol) / 100)
        if lot_index >= self.q_max:
            go_long = False
        elif lot_index <= -self.q_max:
            go_long = True
        else:
            go_long = bool(self.random_state.randint(0, 2))
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
        lot_index += (self.q_max - 1)
        pv = self.theta[lot_index + 1 if go_long else lot_index]
        valuation = terminal_r + pv
        return valuation, go_long

    def placeOrder(self):
        valuation, go_long = self.updateEstimates()
        # Sample a positive Gaussian margin in ticks, clipped to [0, 40]
        margin_draw = self.random_state.normal(loc=0, scale=30)
        margin = int(round(min(40, max(0, margin_draw))))
        limit_price = valuation - margin if go_long else valuation + margin
        best_bid, bid_depth, best_ask, ask_depth = self.getKnownBidAsk(self.symbol)
        if go_long and ask_depth > 0:
            ask_surplus = valuation - best_ask
            if ask_surplus >= (self.eta * margin):
                limit_price = best_ask
        elif (not go_long) and bid_depth > 0:
            bid_surplus = best_bid - valuation
            if bid_surplus >= (self.eta * margin):
                limit_price = best_bid
        order_size = 1
        self.placeLimitOrder(self.symbol, order_size, go_long, limit_price)

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
