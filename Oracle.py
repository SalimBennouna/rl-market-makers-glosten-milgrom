
import datetime as dt
import numpy as np
import pandas as pd

from math import exp, sqrt

class Oracle:

  def __init__(self, mkt_open, mkt_close, symbol, params):
    stamp_init = dt.datetime.now()
    self.mkt_open = mkt_open
    self.mkt_close = mkt_close
    self.symbol = symbol
    self.params = params
    self.f_log = {symbol: [{ 'FundamentalTime' : mkt_open, 'FundamentalValue' : params['r_bar'] }]}
    self.r = {symbol: (mkt_open, params['r_bar'])}
    self.megashocks = {symbol: [self._shock_future(mkt_open)]}
    stamp_end = dt.datetime.now()

  def _guard_symbol(self, symbol):
    if symbol != self.symbol:
      raise ValueError(f"Oracle configured for symbol {self.symbol}, received {symbol}")

  def evolve_to_stamp(self, ts, v_adj, symbol, pt, pv):
    self._guard_symbol(symbol)
    cfg = self.params
    delta_steps = int((ts - pt) / np.timedelta64(1, 'ns'))
    mu = cfg['r_bar']
    gamma = cfg['kappa']
    theta = cfg['fund_vol']
    v = cfg['random_state'].normal(
      loc=mu + (pv - mu) * (exp(-gamma * delta_steps)),
      scale=((theta) / (2 * gamma)) * (1 - exp(-2 * gamma * delta_steps))
    )
    v += v_adj
    v = max(0, v)
    v = int(round(v))
    self.r[symbol] = (ts, v)
    self.f_log[symbol].append({ 'FundamentalTime' : ts, 'FundamentalValue' : v })
    return v

  def _shock_future(self, base_time):
    rng = self.params['random_state']
    delta_ns = rng.exponential(scale=1.0 / self.params['megashock_lambda_a'])
    shock_time = base_time + pd.Timedelta(delta_ns, unit='ns')
    shock_val = rng.normal(loc=self.params['megashock_mean'], scale=sqrt(self.params['megashock_var']))
    shock_val = shock_val if rng.randint(2) == 0 else -shock_val
    return { 'MegashockTime' : shock_time, 'MegashockValue' : shock_val }

  def walk_series(self, currentTime, symbol):
    self._guard_symbol(symbol)
    pt, pv = self.r[symbol]
    if currentTime <= pt:
      return pv
    mst = self.megashocks[symbol][-1]['MegashockTime']
    msv = self.megashocks[symbol][-1]['MegashockValue']
    while mst < currentTime:
      v = self.evolve_to_stamp(mst, msv, symbol, pt, pv)
      pt, pv = mst, v
      next_ms = self._shock_future(pt)
      mst, msv = next_ms['MegashockTime'], next_ms['MegashockValue']
      self.megashocks[symbol].append(next_ms)
    v = self.evolve_to_stamp(currentTime, 0, symbol, pt, pv)
    return v

  def emit_view(self, symbol, currentTime, sigma_n = 1000, random_state = None):
    self._guard_symbol(symbol)
    if currentTime >= self.mkt_close:
      r_t = self.walk_series(self.mkt_close - pd.Timedelta('1ns'), symbol)
    else:
      r_t = self.walk_series(currentTime, symbol)
    if sigma_n == 0:
      obs = r_t
    else:
      obs = int(round(random_state.normal(loc=r_t, scale=sqrt(sigma_n))))
    return obs
