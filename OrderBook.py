import sys

from Message import Message
from Order import LimitOrder

from copy import deepcopy
import pandas as pd
from pandas import json_normalize
class OrderBook:

    def __init__(self, owner, symbol):
        self.owner = owner
        self.symbol = symbol
        self.bids = []
        self.asks = []
        self.last_trade = None

        self.book_log = []
        self.quotes_seen = set()

        self.history = [{}]

        self.last_update_ts = None

        self._transacted_volume = {
            "unrolled_transactions": None,
            "self.history_previous_length": 0
        }

    def process_limit_submit(self, order):

        if order.symbol != self.symbol:
            return

        if (order.quantity <= 0) or (int(order.quantity) != order.quantity):
            return

        self.history[0][order.order_id] = {'entry_time': self.owner.currentTime,
                                           'quantity': order.quantity, 'is_buy_order': order.is_buy_order,
                                           'limit_price': order.limit_price, 'transactions': [],
                                           'modifications': [],
                                           'cancellations': []}

        cont_flag = True

        fill_log = []

        while cont_flag:
            if order.is_buy_order:
                lane = self.asks
            else:
                lane = self.bids

            if not lane:
                match_obj = None
            elif order.is_buy_order == lane[0][0].is_buy_order:
                print(f"[orderbook] mismatched sides for match check: inbound {order} vs existing {lane[0][0]}")
                match_obj = None
            elif order.is_buy_order and (order.limit_price < lane[0][0].limit_price):

                match_obj = None
            elif (not order.is_buy_order) and (order.limit_price > lane[0][0].limit_price):

                match_obj = None
            else:
                if order.quantity >= lane[0][0].quantity:

                    match_obj = lane[0].pop(0)

                    if not lane[0]:
                        del lane[0]

                else:

                    match_obj = deepcopy(lane[0][0])
                    match_obj.quantity = order.quantity

                    lane[0][0].quantity -= match_obj.quantity

                match_obj.fill_price = match_obj.limit_price

                self.history[0][order.order_id]['transactions'].append((self.owner.currentTime, order.quantity))

                for h_idx, h_orders in enumerate(self.history):
                    if match_obj.order_id not in h_orders: continue

                    self.history[h_idx][match_obj.order_id]['transactions'].append(
                        (self.owner.currentTime, match_obj.quantity))

            match_obj = deepcopy(match_obj) if match_obj else None

            if match_obj:

                note_order = deepcopy(order)
                note_order.quantity = match_obj.quantity
                note_order.fill_price = match_obj.fill_price

                order.quantity -= note_order.quantity

                self.owner.sendMessage(order.agent_id, Message({"msg": "ORDER_EXECUTED", "order": note_order}))
                self.owner.sendMessage(match_obj.agent_id,
                                       Message({"msg": "ORDER_EXECUTED", "order": match_obj}))

                fill_log.append((note_order.quantity, note_order.fill_price))

                if order.quantity <= 0:
                    cont_flag = False

            else:

                cloned_order = deepcopy(order)
                if cloned_order.is_buy_order:
                    lane = self.bids
                else:
                    lane = self.asks

                if not lane:

                    lane.append([cloned_order])
                elif not self.outranks_price(cloned_order, lane[-1][0]) and cloned_order.limit_price != lane[-1][0].limit_price:

                    lane.append([cloned_order])
                else:

                    for pos, bucket in enumerate(lane):
                        if self.outranks_price(cloned_order, bucket[0]):
                            lane.insert(pos, [cloned_order])
                            break
                        elif cloned_order.limit_price == bucket[0].limit_price:
                            lane[pos].append(cloned_order)
                            break

                self.owner.sendMessage(order.agent_id, Message({"msg": "ORDER_ACCEPTED", "order": order}))

                cont_flag = False

        if not cont_flag:

            if self.bids:
                self.owner.logEvent('BEST_BID', "{},{},{}".format(self.symbol,
                                                                  self.bids[0][0].limit_price,
                                                                  sum(x.quantity for x in self.bids[0])))

            if self.asks:
                self.owner.logEvent('BEST_ASK', "{},{},{}".format(self.symbol,
                                                                  self.asks[0][0].limit_price,
                                                                  sum(x.quantity for x in self.asks[0])))

            if fill_log:
                agg_qty = 0
                agg_price = 0
                for qty_fill, px_fill in fill_log:
                    agg_qty += qty_fill
                    agg_price += (px_fill * qty_fill)

                avg_price = int(round(agg_price / agg_qty))
                self.owner.logEvent('LAST_TRADE', "{},${:0.4f}".format(agg_qty, avg_price))

                self.last_trade = avg_price

                self.history.insert(0, {})

                self.history = self.history[:self.owner.stream_history + 1]

            if self.owner.book_freq is not None:
                snap = {'QuoteTime': self.owner.currentTime}
                for q_px, q_vol in self.snapshot_bids():
                    snap[q_px] = -q_vol
                    self.quotes_seen.add(q_px)
                for q_px, q_vol in self.snapshot_asks():
                    if q_px in snap:
                        if snap[q_px] is not None:
                            print("[orderbook] detected bid/ask collision at quote {}".format(q_px))
                    snap[q_px] = q_vol
                    self.quotes_seen.add(q_px)
                self.book_log.append(snap)
        self.last_update_ts = self.owner.currentTime

    def process_market_submit(self, order):

        if order.symbol != self.symbol:
            return

        if (order.quantity <= 0) or (int(order.quantity) != order.quantity):
            return

        walk_levels = self.snapshot_asks() if order.is_buy_order else self.snapshot_bids()

        pending_limits = {}
        remaining_qty = order.quantity
        for lvl in walk_levels:
            lvl_px, lvl_sz = lvl[0], lvl[1]
            if remaining_qty <= lvl_sz:
                pending_limits[lvl_px] = remaining_qty
                break
            else:
                pending_limits[lvl_px] = lvl_sz

                remaining_qty -= lvl_sz
                continue
        for entry in pending_limits.items():
            px_key, qty_val = entry[0], entry[1]
            built_lo = LimitOrder(order.agent_id, order.time_placed, order.symbol, qty_val, order.is_buy_order, px_key)
            self.process_limit_submit(built_lo)

    def void_limit_entry(self, order):

        if order.is_buy_order:
            book_side = self.bids
        else:
            book_side = self.asks

        if not book_side: return

        for idx_lvl, price_list in enumerate(book_side):
            if order.limit_price == price_list[0].limit_price:

                for idx_order, ord_ref in enumerate(book_side[idx_lvl]):
                    if order.order_id == ord_ref.order_id:

                        voided = book_side[idx_lvl].pop(idx_order)

                        for h_idx, h_orders in enumerate(self.history):
                            if voided.order_id not in h_orders: continue

                            self.history[h_idx][voided.order_id]['cancellations'].append(
                                (self.owner.currentTime, voided.quantity))

                        if not book_side[idx_lvl]:
                            del book_side[idx_lvl]

                        self.owner.sendMessage(order.agent_id,
                                               Message({"msg": "ORDER_CANCELLED", "order": voided}))

                        self.last_update_ts = self.owner.currentTime
                        return

    def adjust_limit_entry(self, order, new_order):

        if order.order_id != new_order.order_id: return
        book_side = self.bids if order.is_buy_order else self.asks
        if not book_side: return
        for idx_lvl, price_list in enumerate(book_side):
            if order.limit_price == price_list[0].limit_price:
                for idx_mod, ord_ref in enumerate(book_side[idx_lvl]):
                    if order.order_id == ord_ref.order_id:
                        book_side[idx_lvl][0] = new_order
                        for h_idx, h_orders in enumerate(self.history):
                            if new_order.order_id not in h_orders: continue
                            self.history[h_idx][new_order.order_id]['modifications'].append(
                                (self.owner.currentTime, new_order.quantity))
                            self.owner.sendMessage(order.agent_id,
                                                   Message({"msg": "ORDER_MODIFIED", "new_order": new_order}))
        if order.is_buy_order:
            self.bids = book_side
        else:
            self.asks = book_side
        self.last_update_ts = self.owner.currentTime

    def snapshot_bids(self, depth=sys.maxsize):
        inside = []
        for idx in range(min(depth, len(self.bids))):
            qty_sum = 0
            px = self.bids[idx][0].limit_price
            for order_obj in self.bids[idx]:
                qty_sum += order_obj.quantity
            inside.append((px, qty_sum))

        return inside

    def snapshot_asks(self, depth=sys.maxsize):
        inside = []
        for idx in range(min(depth, len(self.asks))):
            qty_sum = 0
            px = self.asks[idx][0].limit_price
            for order_obj in self.asks[idx]:
                qty_sum += order_obj.quantity
            inside.append((px, qty_sum))

        return inside

    def rolling_volume_sum(self, lookback_period='10min'):

        if self._transacted_volume["self.history_previous_length"] == 0:
            self._transacted_volume["self.history_previous_length"] = len(self.history)
            delta_hist = self.history
        elif self._transacted_volume["self.history_previous_length"] == len(self.history):
            delta_hist = {}
        else:
            hist_cut = len(self.history) - self._transacted_volume["self.history_previous_length"] - 1
            delta_hist = self.history[0:hist_cut]
            self._transacted_volume["self.history_previous_length"] = len(self.history)

        flat_hist = []
        for hist_entry in delta_hist:
            for _, hist_val in hist_entry.items():
                flat_hist.append(hist_val)

        if not flat_hist:
            delta_txn = pd.DataFrame(columns=['execution_time', 'quantity'])
        else:
            flat_df = pd.DataFrame(flat_hist, columns=[
                'entry_time', 'quantity', 'is_buy_order', 'limit_price', 'transactions', 'modifications', 'cancellations'
            ])

            exec_df = flat_df[
                flat_df['transactions'].map(lambda d: len(d)) > 0
            ]

            txn_flat = [element for list_ in exec_df['transactions'].values for element in list_]
            delta_txn = pd.DataFrame(txn_flat, columns=['execution_time', 'quantity'])
            delta_txn = delta_txn.sort_values(by=['execution_time'])
            delta_txn = delta_txn.drop_duplicates(keep='last')

        prev_txn = self._transacted_volume["unrolled_transactions"]
        if prev_txn is None:
            prev_txn = pd.DataFrame(columns=['execution_time', 'quantity'])
        combined_txn = pd.concat([prev_txn, delta_txn], ignore_index=True)
        self._transacted_volume["unrolled_transactions"] = combined_txn
        txn_cache = combined_txn

        lb_window = pd.to_timedelta(lookback_period)
        lb_start = self.owner.currentTime - lb_window
        lb_slice = txn_cache[txn_cache['execution_time'] >= lb_start]
        vol_sum = lb_slice['quantity'].sum()

        return vol_sum

    def outranks_price(self, order, peer):

        if order.is_buy_order != peer.is_buy_order:
            print(f"[orderbook] price compare across sides: {order} vs {peer}")
            return False

        if order.is_buy_order and (order.limit_price > peer.limit_price):
            return True

        if not order.is_buy_order and (order.limit_price < peer.limit_price):
            return True

        return False
