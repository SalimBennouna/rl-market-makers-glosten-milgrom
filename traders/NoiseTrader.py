from traders.BaseTrader import BaseTrader

import pandas as pd


class NoiseTrader(BaseTrader):

    def __init__(self, id, name, type, symbol='IBM', starting_cash=101500,
                 order_block=1, log_orders=False, log_to_file=True, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders,
                         log_to_file=log_to_file, random_state=random_state)
        self.symbol = symbol
        self.trading = False
        self.state = 'AWAITING_WAKEUP'
        self.prev_wake_time = None
        # Order size is configurable; default fixed at 1 unless overridden
        self.order_block = order_block if order_block is not None else 1

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.oracle = self.kernel.oracle

    def kernelStopping(self):
        super().kernelStopping()
        lot_count = int(round(self.getHoldings(self.symbol), -2) / 100)
        bid_px, _, ask_px, _ = self.getKnownBidAsk(self.symbol)
        if bid_px and ask_px:
            closing_quote = int(bid_px + ask_px) / 2
        else:
            closing_quote = self.last_trade[self.symbol]
        surplus = closing_quote * lot_count
        surplus += self.holdings['CASH'] - self.starting_cash
        surplus = float(surplus) / self.starting_cash
        self.logEvent('FINAL_VALUATION', surplus, True)

    def getWakeFrequency(self):
        return pd.Timedelta(seconds=9)

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        self.state = 'INACTIVE'
        if not self.mkt_open or not self.mkt_close:
            return
        if not self.trading:
            self.trading = True
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            return
        if self.mkt_closed and (not self.symbol in self.daily_close_price):
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
            return
        if type(self) == NoiseTrader:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
        else:
            self.state = 'ACTIVE'

    def placeOrder(self):
        trade_side = bool(self.random_state.randint(0, 2))
        bid_px, _, ask_px, _ = self.getKnownBidAsk(self.symbol)
        if trade_side and ask_px:
            self.placeLimitOrder(self.symbol, self.order_block, trade_side, ask_px)
        elif (not trade_side) and bid_px:
            self.placeLimitOrder(self.symbol, self.order_block, trade_side, bid_px)

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD':
            if msg.body['msg'] == 'QUERY_SPREAD':
                if self.mkt_closed: return
                self.placeOrder()
                self.setWakeup(currentTime + self.getWakeFrequency())
                self.state = 'AWAITING_WAKEUP'

    def cancelOrders(self):
        if not self.orders: return False
        for id, order in self.orders.items():
            self.cancelOrder(order)
        return True
