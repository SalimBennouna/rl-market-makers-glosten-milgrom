from Message import Message
from OrderBook import OrderBook

import datetime as dt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
pd.set_option('display.max_rows', 500)
from scipy.sparse import dok_matrix
from tqdm import tqdm

from copy import deepcopy


class Exchange:
  def __init__(self, id, name, type, mkt_open, mkt_close, symbols, book_freq='S', wide_book=False, pipeline_delay = 40000,
               computation_delay = 1, stream_history = 0, log_orders = False, random_state = None):

    agent_id = id
    label = name
    role = type
    open_ts = mkt_open
    close_ts = mkt_close
    sym_list = symbols
    freq = book_freq
    wide_flag = wide_book
    pipe_delay = pipeline_delay
    comp_delay = computation_delay
    history_depth = stream_history
    log_flag = log_orders
    rng = random_state

    self.id = agent_id
    self.name = label
    self.type = role
    self.log_to_file = True
    self.random_state = rng
    if not rng:
      raise ValueError("A valid, seeded np.random.RandomState object is required for every exchange", self.name)
    self.kernel = None
    self.currentTime = None
    self.log = []
    self.record_event("AGENT_TYPE", role)

    self.reschedule = False
    self.mkt_open = open_ts
    self.mkt_close = close_ts
    self.pipeline_delay = pipe_delay
    self.computation_delay = comp_delay
    self.stream_history = history_depth
    self.log_orders = log_flag
    self.order_books = {}

    for sym in sym_list:
      self.order_books[sym] = OrderBook(self, sym)

    self.book_freq = freq
    self.wide_book = wide_flag

  def kernelInitializing (self, kernel):
    self.kernel = kernel
    self.oracle = self.kernel.oracle
    for sym in self.order_books:
      try:
        self.order_books[sym].last_trade = self.oracle.params['r_bar']
      except AttributeError:
        pass

  def kernelStarting (self, startTime):
    self.schedule_wake(startTime)

  def kernelStopping (self):
    pass

  def kernelTerminating (self):
    if self.log and self.log_to_file:
      df_log = pd.DataFrame(self.log)
      df_log.set_index('EventTime', inplace=True)
      self.persist_log(df_log)
    if hasattr(self.oracle, 'f_log'):
      for sym in self.oracle.f_log:
        dfFund = pd.DataFrame(self.oracle.f_log[sym])
        if not dfFund.empty:
          dfFund.set_index('FundamentalTime', inplace=True)
          self.persist_log(dfFund, filename='fundamental_{}'.format(sym))
    if self.book_freq is None: return
    for sym in self.order_books:
      t0 = dt.datetime.now()
      self.dump_order_books(sym)
      t1 = dt.datetime.now()
      print(f"[exchange] order book logging duration: {t1 - t0}")
      print("[exchange] order book archival complete")

  def wakeup (self, currentTime):
    self.currentTime = currentTime
    if self.reschedule:
      self.schedule_wake(currentTime + self.wake_interval())

  def receiveMessage(self, currentTime, msg):
    self.currentTime = currentTime
    if currentTime > self.mkt_close:
      if msg.body['msg'] in ['LIMIT_ORDER', 'MARKET_ORDER', 'CANCEL_ORDER', 'MODIFY_ORDER']:
        self.relay(msg.body['sender'], Message({"msg": "MKT_CLOSED"}))
        return
      elif 'QUERY' in msg.body['msg']:
        pass
      else:
        self.relay(msg.body['sender'], Message({"msg": "MKT_CLOSED"}))
        return

    if msg.body['msg'] in ['LIMIT_ORDER', 'MARKET_ORDER', 'CANCEL_ORDER', 'MODIFY_ORDER']:
      if self.log_orders: self.record_event(msg.body['msg'], msg.body['order'].snapshot())
    else:
      self.record_event(msg.body['msg'], msg.body['sender'])

    if msg.body['msg'] == "WHEN_MKT_OPEN":
      self.relay(msg.body['sender'], Message({"msg": "WHEN_MKT_OPEN", "data": self.mkt_open}))
    elif msg.body['msg'] == "WHEN_MKT_CLOSE":
      self.relay(msg.body['sender'], Message({"msg": "WHEN_MKT_CLOSE", "data": self.mkt_close}))
    elif msg.body['msg'] == "QUERY_LAST_TRADE":
      sym = msg.body['symbol']
      if sym not in self.order_books:
        return
      self.relay(msg.body['sender'], Message({"msg": "QUERY_LAST_TRADE", "symbol": sym,
                                                    "data": self.order_books[sym].last_trade,
                                                    "mkt_closed": True if currentTime > self.mkt_close else False}))
    elif msg.body['msg'] == "QUERY_SPREAD":
      sym = msg.body['symbol']
      depth = msg.body['depth']
      if sym not in self.order_books:
        return
      self.relay(msg.body['sender'], Message({"msg": "QUERY_SPREAD", "symbol": sym, "depth": depth,
                                                    "bids": self.order_books[sym].snapshot_bids(depth),
                                                    "asks": self.order_books[sym].snapshot_asks(depth),
                                                    "data": self.order_books[sym].last_trade,
                                                    "mkt_closed": True if currentTime > self.mkt_close else False,
                                                    "book": ''}))
    elif msg.body['msg'] == "QUERY_ORDER_STREAM":
      sym = msg.body['symbol']
      length = msg.body['length']
      if sym not in self.order_books:
        return
      self.relay(msg.body['sender'], Message({"msg": "QUERY_ORDER_STREAM", "symbol": sym, "length": length,
                                                    "mkt_closed": True if currentTime > self.mkt_close else False,
                                                    "orders": self.order_books[sym].history[1:length + 1]
                                                    }))
    elif msg.body['msg'] == 'QUERY_TRANSACTED_VOLUME':
      sym = msg.body['symbol']
      lookback = msg.body['lookback_period']
      if sym not in self.order_books:
        return
      self.relay(msg.body['sender'], Message({"msg": "QUERY_TRANSACTED_VOLUME", "symbol": sym,
                                                    "transacted_volume": self.order_books[sym].rolling_volume_sum(lookback),
                                                    "mkt_closed": True if currentTime > self.mkt_close else False
                                                    }))
    elif msg.body['msg'] == "LIMIT_ORDER":
      order = msg.body['order']
      if order.symbol not in self.order_books:
        return
      self.order_books[order.symbol].process_limit_submit(deepcopy(order))
    elif msg.body['msg'] == "MARKET_ORDER":
      order = msg.body['order']
      if order.symbol not in self.order_books:
        return
      self.order_books[order.symbol].process_market_submit(deepcopy(order))
    elif msg.body['msg'] == "CANCEL_ORDER":
      order = msg.body['order']
      if order.symbol not in self.order_books:
        return
      self.order_books[order.symbol].void_limit_entry(deepcopy(order))
    elif msg.body['msg'] == 'MODIFY_ORDER':
      order = msg.body['order']
      new_order = msg.body['new_order']
      if order.symbol not in self.order_books:
        return
      self.order_books[order.symbol].adjust_limit_entry(deepcopy(order), deepcopy(new_order))

  def wake_interval(self):
    return pd.Timedelta(self.computation_delay)

  def dump_order_books(self, symbol):
    def book_log_to_df(book):
      quotes = sorted(list(book.quotes_seen))
      log_len = len(book.book_log)
      quote_idx_dict = {quote: idx for idx, quote in enumerate(quotes)}
      quotes_times = []

      S = dok_matrix((log_len, len(quotes)), dtype=int)

      for i, row in enumerate(tqdm(book.book_log, desc="Processing orderbook log")):
        quotes_times.append(row['QuoteTime'])
        for quote, vol in row.items():
          if quote == "QuoteTime":
            continue
          S[i, quote_idx_dict[quote]] = vol

      S = S.tocsc()
      df = pd.DataFrame.sparse.from_spmatrix(S, columns=quotes)
      df.insert(0, 'QuoteTime', quotes_times, allow_duplicates=True)
      return df

    def get_quote_range_iterator(s):
      forbidden_values = [0, 19999900]
      quotes = sorted(s)
      for val in forbidden_values:
        try: quotes.remove(val)
        except ValueError:
          pass
      return quotes

    book = self.order_books[symbol]

    if book.book_log:

      print(f"[exchange] archiving order book for {symbol} ...")
      dfLog = book_log_to_df(book)
      dfLog.set_index('QuoteTime', inplace=True)
      dfLog = dfLog[~dfLog.index.duplicated(keep='last')]
      dfLog.sort_index(inplace=True)

      if str(self.book_freq).isdigit() and int(self.book_freq) == 0:
        quotes = get_quote_range_iterator(dfLog.columns.unique())

        if not self.wide_book:
          filledIndex = pd.MultiIndex.from_product([dfLog.index, quotes], names=['time', 'quote'])
          dfLog = dfLog.stack()
          dfLog = dfLog.reindex(filledIndex)

        filename = f'ORDERBOOK_{symbol}_FULL'

      else:
        dfLog = dfLog.resample(self.book_freq).ffill()
        dfLog.sort_index(inplace=True)

        # pandas >=2.0 uses inclusive instead of closed
        try:
          time_idx = pd.date_range(self.mkt_open, self.mkt_close, freq=self.book_freq, closed='right')
        except TypeError:
          time_idx = pd.date_range(self.mkt_open, self.mkt_close, freq=self.book_freq, inclusive='right')
        dfLog = dfLog.reindex(time_idx, method='ffill')
        dfLog.sort_index(inplace=True)

        if not self.wide_book:
          dfLog = dfLog.stack()
          dfLog.sort_index(inplace=True)

          quotes = get_quote_range_iterator(dfLog.index.get_level_values(1).unique())

          filledIndex = pd.MultiIndex.from_product([time_idx, quotes], names=['time', 'quote'])
          dfLog = dfLog.reindex(filledIndex)

        filename = f'ORDERBOOK_{symbol}'

      self.persist_log(dfLog, filename)

  def record_event (self, eventType, event = '', appendSummaryLog = False):
    e = deepcopy(event)
    self.log.append({ 'EventTime' : self.currentTime, 'EventType' : eventType,
                      'Event' : e })
    if appendSummaryLog: self.kernel.appendSummaryLog(self.id, eventType, e)

  def relay (self, recipientID, msg, delay = 0):
    self.kernel.dispatch(self.id, recipientID, msg, delay = delay)

  def schedule_wake (self, requestedTime):
    self.kernel.schedule_wake(self.id, requestedTime)

  def persist_log (self, dfLog, filename=None):
    self.kernel.archive_df(self.id, dfLog, filename)

  def set_state (self, state):
    self.kernel.updateAgentState(self.id, state)

  def logEvent(self, eventType, event='', appendSummaryLog=False):
    return self.record_event(eventType, event, appendSummaryLog)

  def sendMessage(self, recipientID, msg, delay=0):
    return self.relay(recipientID, msg, delay)

  def setWakeup(self, requestedTime):
    return self.schedule_wake(requestedTime)

  def writeLog(self, dfLog, filename=None):
    return self.persist_log(dfLog, filename)

  def updateAgentState(self, state):
    return self.set_state(state)

  def getWakeFrequency(self):
    return self.wake_interval()

  def dollarize(self, cents):
    if isinstance(cents, list):
      return [self.dollarize(x) for x in cents]
    if isinstance(cents, int):
      return "${:0.2f}".format(cents / 100)
    return cents

  def __lt__(self, other):
    return ("{}".format(self.id) <
            "{}".format(other.id))
