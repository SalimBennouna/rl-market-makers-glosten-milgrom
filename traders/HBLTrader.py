from traders.ZITrader import ZITrader
import numpy as np
import sys

np.set_printoptions(threshold=np.inf)


class HBLTrader(ZITrader):
    def __init__(self, id, name, type, symbol='IBM', starting_cash=101000, sigma_n=950,
                 r_bar=99500, kappa=0.047, sigma_s=98000, q_max=11, sigma_pv=4800000, R_min=1,
                 R_max=245, eta=0.96, lambda_a=0.0046, L=9, log_orders=False, random_state=None):
        super().__init__(id, name, type, symbol=symbol, starting_cash=starting_cash, sigma_n=sigma_n,
                         r_bar=r_bar, kappa=kappa, sigma_s=sigma_s, q_max=q_max, sigma_pv=sigma_pv, R_min=R_min,
                         R_max=R_max, eta=eta, lambda_a=lambda_a, log_orders=log_orders, random_state=random_state)
        self.lookback = L

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        if self.state != 'ACTIVE':
            return
        self.getOrderStream(self.symbol, length=self.lookback)
        self.state = 'AWAITING_STREAM'

    def placeOrder(self):
        if len(self.stream_history[self.symbol]) < self.lookback:
            super().placeOrder()
            return
        valuation, go_long = self.updateEstimates()
        low_px = sys.maxsize
        high_px = 0
        for snapshot in self.stream_history[self.symbol]:
            for _, entry in snapshot.items():
                px = entry['limit_price']
                if px < low_px:
                    low_px = px
                if px > high_px:
                    high_px = px
        nd = np.zeros((high_px - low_px + 1, 8))
        for snapshot in self.stream_history[self.symbol]:
            for _, entry in snapshot.items():
                px = entry['limit_price']
                if entry['is_buy_order']:
                    if entry['transactions']:
                        nd[px - low_px, 1] += 1
                    else:
                        nd[px - low_px, 3] += 1
                else:
                    if entry['transactions']:
                        nd[px - low_px, 0] += 1
                    else:
                        nd[px - low_px, 2] += 1
        if go_long:
            nd[:, [0, 1, 2]] = np.cumsum(nd[:, [0, 1, 2]], axis=0)
            nd[::-1, 3] = np.cumsum(nd[::-1, 3], axis=0)
            nd[:, 4] = np.sum(nd[:, [0, 1, 2]], axis=1)
        else:
            nd[::-1, [0, 1, 3]] = np.cumsum(nd[::-1, [0, 1, 3]], axis=0)
            nd[:, 2] = np.cumsum(nd[:, 2], axis=0)
            nd[:, 4] = np.sum(nd[:, [0, 1, 3]], axis=1)
        nd[:, 5] = np.sum(nd[:, 0:4], axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            nd[:, 6] = np.nan_to_num(np.divide(nd[:, 4], nd[:, 5]))
        if go_long:
            nd[:, 7] = nd[:, 6] * (valuation - np.arange(low_px, high_px + 1))
        else:
            nd[:, 7] = nd[:, 6] * (np.arange(low_px, high_px + 1) - valuation)
        best_idx = np.argmax(nd[:, 7])
        best_surplus = nd[best_idx, 7]
        best_px = low_px + best_idx
        if best_surplus > 0:
            self.placeLimitOrder(self.symbol, 100, go_long, int(round(best_px)))

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_STREAM':
            if msg.body['msg'] == 'QUERY_ORDER_STREAM':
                if self.mkt_closed:
                    return
                self.getCurrentSpread(self.symbol)
                self.state = 'AWAITING_SPREAD'
