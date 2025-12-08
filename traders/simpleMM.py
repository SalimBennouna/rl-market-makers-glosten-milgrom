from traders.BaseTrader import BaseTrader
import pandas as pd


class simpleMM(BaseTrader):
    def __init__(self, id, name, type, symbol='IBM', starting_cash=101200, min_size=18, max_size=82,
                 wake_up_freq='1100ms', log_orders=False, random_state=None, inventory_limit=120):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.qty_floor = min_size
        self.qty_cap = max_size
        self.quote_size = self.qty_floor
        self.wake_gap = wake_up_freq
        self.log_orders = log_orders
        self.state = 'AWAITING_WAKEUP'
        self.prev_gap = 12
        self.inventory_cap = inventory_limit

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_gap)

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if can_trade:
            self.getCurrentSpread(self.symbol, depth=1)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            mid_px = self.last_trade[self.symbol] if self.last_trade[self.symbol] is not None else None
            bid_px, bid_vol, ask_px, ask_vol = self.getKnownBidAsk(self.symbol)
            if bid_px and ask_px:
                mid_px = int((ask_px + bid_px) / 2)
                gap = int(abs(ask_px - bid_px) / 2)
            else:
                gap = self.prev_gap
            old_order_ids = list(self.orders.keys())
            inv = self.getHoldings(self.symbol)
            if inv < self.inventory_cap:
                self.placeLimitOrder(self.symbol, self.quote_size, True, mid_px - gap)
            if inv > -self.inventory_cap:
                self.placeLimitOrder(self.symbol, self.quote_size, False, mid_px + gap)
            for oid in old_order_ids:
                if oid in self.orders:
                    self.cancelOrder(self.orders[oid])
            self._log_state(currentTime, mid_px, gap)
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'

    def cancelOrders(self):
        for _, order in self.orders.items():
            self.cancelOrder(order)

    def _log_state(self, currentTime, mid, spread):
        self.logEvent('STATE', {
            'time': currentTime,
            'mid': mid,
            'spread': spread,
            'inventory': self.getHoldings(self.symbol),
            'cash': self.holdings['CASH'],
            'mtm': self.holdings['CASH'] + (mid if mid is not None else 0) * self.getHoldings(self.symbol),
        })
