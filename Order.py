import sys
import traceback
from copy import deepcopy


class Order:
    order_id = 0
    _order_ids = set()

    def __init__(self, agent_id, time_placed, symbol, quantity, is_buy_order, order_id=None, tag=None):
        sender = agent_id
        ts = time_placed
        ticker = symbol
        qty = quantity
        side_flag = is_buy_order
        seed_order_id = order_id
        label = tag
        self.agent_id = sender
        self.time_placed = ts
        self.symbol = ticker
        self.quantity = qty
        self.is_buy_order = side_flag
        self.order_id = self.allocate_id() if not seed_order_id else seed_order_id
        Order._order_ids.add(self.order_id)
        self.fill_price = None
        self.tag = label

    def allocate_id(self):
        if Order.order_id not in Order._order_ids:
            fresh_id = Order.order_id
        else:
            Order.order_id += 1
            fresh_id = self.allocate_id()
        return fresh_id

    def snapshot(self):
        payload = deepcopy(self).__dict__
        payload['time_placed'] = self.time_placed.isoformat()
        return payload

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memodict={}):
        raise NotImplementedError


class LimitOrder(Order):

    def __init__(self, agent_id, time_placed, symbol, quantity, is_buy_order, limit_price, order_id=None, tag=None):

        super().__init__(agent_id, time_placed, symbol, quantity, is_buy_order, order_id, tag=tag)

        self.limit_price: int = limit_price

    def __str__(self):
        fill_note = f" filled {self.dollarize(self.fill_price)}" if self.fill_price else ""
        px_label = self.dollarize(self.limit_price) if abs(self.limit_price) < sys.maxsize else "MKT"
        side_label = "B" if self.is_buy_order else "S"
        tag_note = f" [{self.tag}]" if self.tag is not None else ""
        return f"[{self.agent_id}|{self.time_placed}{tag_note}] {side_label} {self.quantity} {self.symbol} @ {px_label}{fill_note}"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        cloned = LimitOrder(self.agent_id, self.time_placed, self.symbol, self.quantity, self.is_buy_order,
                            self.limit_price,
                            order_id=self.order_id,
                            tag=self.tag)
        Order._order_ids.pop()
        cloned.fill_price = self.fill_price
        return cloned

    def dollarize(self, cents_input):
        if isinstance(cents_input, list):
            return [self.dollarize(x) for x in cents_input]
        if isinstance(cents_input, int):
            return "${:0.2f}".format(cents_input / 100)
        print(f"dollarize() expects int or list, got {cents_input!r}")
        traceback.print_stack()
        sys.exit()

    def __deepcopy__(self, memodict={}):
        agent_copy = deepcopy(self.agent_id, memodict)
        placed_copy = deepcopy(self.time_placed, memodict)
        symbol_copy = deepcopy(self.symbol, memodict)
        qty_copy = deepcopy(self.quantity, memodict)
        side_copy = deepcopy(self.is_buy_order, memodict)
        px_copy = deepcopy(self.limit_price, memodict)
        id_copy = deepcopy(self.order_id, memodict)
        tag_copy = deepcopy(self.tag, memodict)
        fill_copy = deepcopy(self.fill_price, memodict)

        shadow = LimitOrder(agent_copy, placed_copy, symbol_copy, qty_copy, side_copy, px_copy,
                            order_id=id_copy, tag=tag_copy)
        shadow.fill_price = fill_copy

        return shadow


class MarketOrder(Order):

    def __init__(self, agent_id, time_placed, symbol, quantity, is_buy_order, order_id=None, tag=None):
        super().__init__(agent_id, time_placed, symbol, quantity, is_buy_order, order_id=order_id, tag=tag)

    def __str__(self):
        side_label = "B" if self.is_buy_order else "S"
        return f"[{self.agent_id}|{self.time_placed}] MKT {side_label} {self.quantity} {self.symbol}"

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        order = MarketOrder(self.agent_id, self.time_placed, self.symbol, self.quantity, self.is_buy_order,
                            order_id=self.order_id,
                            tag=self.tag)
        Order._order_ids.pop()
        order.fill_price = self.fill_price
        return order

    def __deepcopy__(self, memodict={}):
        agent_copy = deepcopy(self.agent_id, memodict)
        placed_copy = deepcopy(self.time_placed, memodict)
        symbol_copy = deepcopy(self.symbol, memodict)
        qty_copy = deepcopy(self.quantity, memodict)
        side_copy = deepcopy(self.is_buy_order, memodict)
        id_copy = deepcopy(self.order_id, memodict)
        tag_copy = deepcopy(self.tag, memodict)
        fill_copy = deepcopy(self.fill_price, memodict)

        clone = MarketOrder(agent_copy, placed_copy, symbol_copy, qty_copy, side_copy, order_id=id_copy, tag=tag_copy)
        clone.fill_price = fill_copy

        return clone

    def dollarize(self, cents_input):
        if isinstance(cents_input, list):
            return [self.dollarize(x) for x in cents_input]
        if isinstance(cents_input, int):
            return "${:0.2f}".format(cents_input / 100)
        print(f"dollarize() expects int or list, got {cents_input!r}")
        traceback.print_stack()
        sys.exit()
