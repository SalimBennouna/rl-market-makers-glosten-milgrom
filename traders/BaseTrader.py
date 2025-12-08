from Message import Message
from Order import LimitOrder, MarketOrder

import pandas as pd
from copy import deepcopy
import sys
import traceback


class BaseTrader:
    def __init__(self, id, name, type, random_state=None, starting_cash=100000, log_orders=False, log_to_file=True):
        self.id = id
        self.name = name
        self.type = type
        self.log_to_file = log_to_file
        self.random_state = random_state
        if not random_state:
            raise ValueError("A valid, seeded np.random.RandomState object is required for every agent", self.name)
        self.kernel = None
        self.currentTime = None
        self.log = []
        self.logEvent('AGENT_TYPE', type)
        self.mkt_open = None
        self.mkt_close = None
        self.log_orders = log_orders
        if log_orders is None:
            self.log_orders = False
            self.log_to_file = False
        self.starting_cash = starting_cash
        self.MKT_BUY = sys.maxsize
        self.MKT_SELL = 0
        self.holdings = {'CASH': starting_cash}
        self.orders = {}
        self.last_trade = {}
        self.daily_close_price = {}
        self.nav_diff = 0
        self.basket_size = 0
        self.known_bids = {}
        self.known_asks = {}
        self.stream_history = {}
        self.transacted_volume = {}
        self.executed_orders = []
        self.first_wake = True
        self.mkt_closed = False
        self.book = ''

    def kernelInitializing(self, kernel):
        self.kernel = kernel

    def kernelStarting(self, startTime):
        self.logEvent('STARTING_CASH', self.starting_cash, True)
        self.exchangeID = None
        for ag in self.kernel.agents:
            if getattr(ag.__class__, '__name__', '') == 'Exchange':
                self.exchangeID = ag.id
                break
        self.setWakeup(startTime)

    def kernelStopping(self):
        self.logEvent('FINAL_HOLDINGS', self.fmtHoldings(self.holdings))
        self.logEvent('FINAL_CASH_POSITION', self.holdings['CASH'], True)
        cash = self.markToMarket(self.holdings)
        self.logEvent('ENDING_CASH', cash, True)
        print("Final holdings for {}: {}.  Marked to market: {}".format(self.name, self.fmtHoldings(self.holdings), cash))
        mytype = self.type
        gain = cash - self.starting_cash
        if mytype in self.kernel.meanResultByAgentType:
            self.kernel.meanResultByAgentType[mytype] += gain
            self.kernel.agentCountByType[mytype] += 1
        else:
            self.kernel.meanResultByAgentType[mytype] = gain
            self.kernel.agentCountByType[mytype] = 1

    def kernelTerminating(self):
        if self.log and self.log_to_file:
            dfLog = pd.DataFrame(self.log)
            dfLog.set_index('EventTime', inplace=True)
            self.writeLog(dfLog)

    def wakeup(self, currentTime):
        self.currentTime = currentTime
        if self.first_wake:
            self.logEvent('HOLDINGS_UPDATED', self.holdings)
            self.first_wake = False
        if self.mkt_open is None:
            self.sendMessage(self.exchangeID, Message({"msg": "WHEN_MKT_OPEN", "sender": self.id}))
            self.sendMessage(self.exchangeID, Message({"msg": "WHEN_MKT_CLOSE", "sender": self.id}))
        return (self.mkt_open and self.mkt_close) and not self.mkt_closed

    def receiveMessage(self, currentTime, msg):
        self.currentTime = currentTime
        had_mkt_hours = self.mkt_open is not None and self.mkt_close is not None
        if msg.body['msg'] == "WHEN_MKT_OPEN":
            self.mkt_open = msg.body['data']
        elif msg.body['msg'] == "WHEN_MKT_CLOSE":
            self.mkt_close = msg.body['data']
        elif msg.body['msg'] == "ORDER_EXECUTED":
            order = msg.body['order']
            self.orderExecuted(order)
        elif msg.body['msg'] == "ORDER_ACCEPTED":
            order = msg.body['order']
            self.orderAccepted(order)
        elif msg.body['msg'] == "ORDER_CANCELLED":
            order = msg.body['order']
            self.orderCancelled(order)
        elif msg.body['msg'] == "MKT_CLOSED":
            self.marketClosed()
        elif msg.body['msg'] == 'QUERY_LAST_TRADE':
            if msg.body['mkt_closed']:
                self.mkt_closed = True
            self.queryLastTrade(msg.body['symbol'], msg.body['data'])
        elif msg.body['msg'] == 'QUERY_SPREAD':
            if msg.body['mkt_closed']:
                self.mkt_closed = True
            self.querySpread(msg.body['symbol'], msg.body['data'], msg.body['bids'], msg.body['asks'], msg.body['book'])
        elif msg.body['msg'] == 'QUERY_ORDER_STREAM':
            if msg.body['mkt_closed']:
                self.mkt_closed = True
            self.queryOrderStream(msg.body['symbol'], msg.body['orders'])
        elif msg.body['msg'] == 'QUERY_TRANSACTED_VOLUME':
            if msg.body['mkt_closed']:
                self.mkt_closed = True
            self.query_transacted_volume(msg.body['symbol'], msg.body['transacted_volume'])
        have_mkt_hours = self.mkt_open is not None and self.mkt_close is not None
        if have_mkt_hours and not had_mkt_hours:
            ns_offset = self.getWakeFrequency()
            self.setWakeup(self.mkt_open + ns_offset)

    def getLastTrade(self, symbol):
        self.sendMessage(self.exchangeID, Message({"msg": "QUERY_LAST_TRADE", "sender": self.id, "symbol": symbol}))

    def getCurrentSpread(self, symbol, depth=1):
        self.sendMessage(self.exchangeID, Message({"msg": "QUERY_SPREAD", "sender": self.id, "symbol": symbol, "depth": depth}))

    def getOrderStream(self, symbol, length=1):
        self.sendMessage(self.exchangeID, Message({"msg": "QUERY_ORDER_STREAM", "sender": self.id, "symbol": symbol, "length": length}))

    def get_transacted_volume(self, symbol, lookback_period='10min'):
        self.sendMessage(self.exchangeID, Message({"msg": "QUERY_TRANSACTED_VOLUME", "sender": self.id, "symbol": symbol, "lookback_period": lookback_period}))

    def placeLimitOrder(self, symbol, quantity, is_buy_order, limit_price, order_id=None, ignore_risk=True, tag=None):
        order = LimitOrder(self.id, self.currentTime, symbol, quantity, is_buy_order, limit_price, order_id, tag)
        if quantity > 0:
            new_holdings = self.holdings.copy()
            q = order.quantity if order.is_buy_order else -order.quantity
            if order.symbol in new_holdings:
                new_holdings[order.symbol] += q
            else:
                new_holdings[order.symbol] = q
            if not ignore_risk:
                at_risk = self.markToMarket(self.holdings) - self.holdings['CASH']
                new_at_risk = self.markToMarket(new_holdings) - new_holdings['CASH']
                if (new_at_risk > at_risk) and (new_at_risk > self.starting_cash):
                    return
            self.orders[order.order_id] = deepcopy(order)
            self.sendMessage(self.exchangeID, Message({"msg": "LIMIT_ORDER", "sender": self.id, "order": order}))
            if self.log_orders:
                self.logEvent('ORDER_SUBMITTED', order.snapshot())

    def placeMarketOrder(self, symbol, quantity, is_buy_order, order_id=None, ignore_risk=True, tag=None):
        order = MarketOrder(self.id, self.currentTime, symbol, quantity, is_buy_order, order_id)
        if quantity > 0:
            new_holdings = self.holdings.copy()
            q = order.quantity if order.is_buy_order else -order.quantity
            if order.symbol in new_holdings:
                new_holdings[order.symbol] += q
            else:
                new_holdings[order.symbol] = q
            if not ignore_risk:
                at_risk = self.markToMarket(self.holdings) - self.holdings['CASH']
                new_at_risk = self.markToMarket(new_holdings) - new_holdings['CASH']
                if (new_at_risk > at_risk) and (new_at_risk > self.starting_cash):
                    return
            self.orders[order.order_id] = deepcopy(order)
            self.sendMessage(self.exchangeID, Message({"msg": "MARKET_ORDER", "sender": self.id, "order": order}))
            if self.log_orders:
                self.logEvent('ORDER_SUBMITTED', order.snapshot())

    def cancelOrder(self, order):
        if isinstance(order, LimitOrder):
            self.sendMessage(self.exchangeID, Message({"msg": "CANCEL_ORDER", "sender": self.id, "order": order}))
            if self.log_orders:
                self.logEvent('CANCEL_SUBMITTED', order.snapshot())

    def modifyOrder(self, order, newOrder):
        self.sendMessage(self.exchangeID, Message({"msg": "MODIFY_ORDER", "sender": self.id, "order": order, "new_order": newOrder}))
        if self.log_orders:
            self.logEvent('MODIFY_ORDER', order.snapshot())

    def orderExecuted(self, order):
        if self.log_orders:
            self.logEvent('ORDER_EXECUTED', order.snapshot())
        qty = order.quantity if order.is_buy_order else -1 * order.quantity
        sym = order.symbol
        if sym in self.holdings:
            self.holdings[sym] += qty
        else:
            self.holdings[sym] = qty
        if self.holdings[sym] == 0:
            del self.holdings[sym]
        self.holdings['CASH'] -= (qty * order.fill_price)
        if order.order_id in self.orders:
            o = self.orders[order.order_id]
            if order.quantity >= o.quantity:
                del self.orders[order.order_id]
            else:
                o.quantity -= order.quantity
        self.logEvent('HOLDINGS_UPDATED', self.holdings)

    def orderAccepted(self, order):
        if self.log_orders:
            self.logEvent('ORDER_ACCEPTED', order.snapshot())

    def orderCancelled(self, order):
        if self.log_orders:
            self.logEvent('ORDER_CANCELLED', order.snapshot())
        if order.order_id in self.orders:
            del self.orders[order.order_id]

    def marketClosed(self):
        self.logEvent('MKT_CLOSED')
        self.mkt_closed = True

    def queryLastTrade(self, symbol, price):
        self.last_trade[symbol] = price
        if self.mkt_closed:
            self.daily_close_price[symbol] = self.last_trade[symbol]

    def querySpread(self, symbol, price, bids, asks, book):
        self.queryLastTrade(symbol, price)
        self.known_bids[symbol] = bids
        self.known_asks[symbol] = asks
        self.logEvent("BID_DEPTH", bids)
        self.logEvent("ASK_DEPTH", asks)
        self.logEvent("IMBALANCE", [sum([x[1] for x in bids]), sum([x[1] for x in asks])])
        self.book = book

    def queryOrderStream(self, symbol, orders):
        self.stream_history[self.symbol] = orders

    def query_transacted_volume(self, symbol, transacted_volume):
        self.transacted_volume[symbol] = transacted_volume

    def getKnownBidAsk(self, symbol, best=True):
        if best:
            bid = self.known_bids[symbol][0][0] if self.known_bids[symbol] else None
            ask = self.known_asks[symbol][0][0] if self.known_asks[symbol] else None
            bid_vol = self.known_bids[symbol][0][1] if self.known_bids[symbol] else 0
            ask_vol = self.known_asks[symbol][0][1] if self.known_asks[symbol] else 0
            return bid, bid_vol, ask, ask_vol
        bids = self.known_bids[symbol] if self.known_bids[symbol] else None
        asks = self.known_asks[symbol] if self.known_asks[symbol] else None
        return bids, asks

    def getKnownLiquidity(self, symbol, within=0.00):
        bid_liq = self.getBookLiquidity(self.known_bids[symbol], within)
        ask_liq = self.getBookLiquidity(self.known_asks[symbol], within)
        return bid_liq, ask_liq

    def getBookLiquidity(self, book, within):
        liq = 0
        for i, (price, shares) in enumerate(book):
            if i == 0:
                best = price
            if abs(best - price) <= int(round(best * within)):
                liq += shares
        return liq

    def markToMarket(self, holdings, use_midpoint=False):
        cash = holdings['CASH']
        cash += self.basket_size * self.nav_diff
        for symbol, shares in holdings.items():
            if symbol == 'CASH':
                continue
            if use_midpoint:
                bid, ask, midpoint = self.getKnownBidAskMidpoint(symbol)
                if bid is None or ask is None or midpoint is None:
                    value = self.last_trade[symbol] * shares
                else:
                    value = midpoint * shares
            else:
                value = self.last_trade[symbol] * shares
            cash += value
            self.logEvent('MARK_TO_MARKET', "{} {} @ {} == {}".format(shares, symbol, self.last_trade[symbol], value))
        self.logEvent('MARKED_TO_MARKET', cash)
        return cash

    def getHoldings(self, symbol):
        if symbol in self.holdings:
            return self.holdings[symbol]
        return 0

    def getKnownBidAskMidpoint(self, symbol):
        bid = self.known_bids[symbol][0][0] if self.known_bids[symbol] else None
        ask = self.known_asks[symbol][0][0] if self.known_asks[symbol] else None
        midpoint = int(round((bid + ask) / 2)) if bid is not None and ask is not None else None
        return bid, ask, midpoint

    def get_average_transaction_price(self):
        return round(sum(executed_order.quantity * executed_order.fill_price for executed_order in self.executed_orders) / sum(executed_order.quantity for executed_order in self.executed_orders), 2)

    def fmtHoldings(self, holdings):
        h = ''
        for k, v in sorted(holdings.items()):
            if k == 'CASH':
                continue
            h += "{}: {}, ".format(k, v)
        h += "{}: {}".format('CASH', holdings['CASH'])
        h = '{ ' + h + ' }'
        return h

    def logEvent(self, eventType, event='', appendSummaryLog=False):
        e = deepcopy(event)
        self.log.append({'EventTime': self.currentTime, 'EventType': eventType, 'Event': e})
        if appendSummaryLog and self.kernel:
            self.kernel.appendSummaryLog(self.id, eventType, e)

    def sendMessage(self, recipientID, msg, delay=0):
        self.kernel.dispatch(self.id, recipientID, msg, delay=delay)

    def setWakeup(self, requestedTime):
        self.kernel.schedule_wake(self.id, requestedTime)

    def writeLog(self, dfLog, filename=None):
        self.kernel.archive_df(self.id, dfLog, filename)

    def updateAgentState(self, state):
        self.kernel.updateAgentState(self.id, state)

    def __lt__(self, other):
        return "{}".format(self.id) < "{}".format(other.id)

    def dollarize(self, cents):
        if isinstance(cents, list):
            return [self.dollarize(x) for x in cents]
        if isinstance(cents, int):
            return "${:0.2f}".format(cents / 100)
        print("ERROR: dollarize(cents) called without int or list of ints: {}".format(cents))
        traceback.print_stack()
        sys.exit()
