
import argparse
import numpy as np
import pandas as pd
import datetime as dt

from SimulationCore import SimulationCore
from Order import LimitOrder
from Oracle import Oracle

from Exchange import Exchange
from traders.NoiseTrader import NoiseTrader
from traders.ValueTrader import ValueTrader
from traders.ZITrader import ZITrader
from traders.HBLTrader import HBLTrader
from traders.simpleMM import simpleMM
from traders.RLMM0 import RLMM0
from traders.RLMM1 import RLMM1
from traders.RLMM2 import RLMM2
from traders.RLMM3 import RLMM3
from traders.RLMM4 import RLMM4
from traders.RLMM5 import RLMM5
from traders.TechnicalTrader import TechnicalTrader

cli_parser = argparse.ArgumentParser(description='Simulation configuration.')

cli_parser.add_argument('-c',
                        '--config',
                        required=True,
                        help='Name of config file to execute')
cli_parser.add_argument('--start-time',
                        default='09:30:00',
                        type=str,
                        help='Starting time of simulation (timedelta string, e.g., HH:MM:SS).'
                        )
cli_parser.add_argument('--end-time',
                        default='11:30:00',
                        type=str,
                        help='Ending time of simulation (timedelta string, e.g., up to 168:00:00).'
                        )
cli_parser.add_argument('-l',
                        '--log_dir',
                        default=None,
                        help='Log directory name (default: unix timestamp at program start)')
cli_parser.add_argument('-s',
                        '--seed',
                        type=int,
                        default=None,
                        help='numpy.random.seed() for simulation')

cli_parser.add_argument('--mm-type',
                        choices=['none', 'simple', 'rl_baseline', 'rl_tabular', 'RLMM0', 'RLMM1', 'RLMM2', 'RLMM3', 'RLMM4', 'RLMM5'],
                        default='none',
                        help='Which market maker class to use (or none).')
cli_parser.add_argument('--mm-wake-up-freq',
                        type=str,
                        default='10S'
                        )
cli_parser.add_argument('--mm-size',
                        type=int,
                        default=10,
                        help='Fixed size for market maker orders')

cli_parser.add_argument('--fund-vol',
                        type=float,
                        default=5e-9,
                        help='Volatility of fundamental time series.'
                        )
cli_parser.add_argument('--noise-count', type=int, default=0)
cli_parser.add_argument('--noise-cash', type=int, default=10000000)
cli_parser.add_argument('--noise-order-block', type=int, default=1,
                        help='Fixed order size for noise traders (set None for random in code).')
cli_parser.add_argument('--value-count', type=int, default=5)
cli_parser.add_argument('--value-cash', type=int, default=10000000)
cli_parser.add_argument('--value-sigma-n', type=float, default=None)
cli_parser.add_argument('--value-r-bar', type=float, default=None)
cli_parser.add_argument('--value-kappa', type=float, default=None)
cli_parser.add_argument('--value-lambda-a', type=float, default=None)
cli_parser.add_argument('--mm-inventory-limit', type=int, default=100)
cli_parser.add_argument('--zi-count', type=int, default=0)
cli_parser.add_argument('--zi-cash', type=int, default=10000000)
cli_parser.add_argument('--zi-sigma-n', type=float, default=10000)
cli_parser.add_argument('--zi-sigma-s', type=float, default=None)
cli_parser.add_argument('--zi-kappa', type=float, default=None)
cli_parser.add_argument('--zi-r-bar', type=float, default=None)
cli_parser.add_argument('--zi-q-max', type=int, default=10)
cli_parser.add_argument('--zi-sigma-pv', type=float, default=5e4)
cli_parser.add_argument('--zi-R-min', type=float, default=0)
cli_parser.add_argument('--zi-R-max', type=float, default=100)
cli_parser.add_argument('--zi-eta', type=float, default=1.0)
cli_parser.add_argument('--zi-lambda-a', type=float, default=1e-12)
cli_parser.add_argument('--hbl-count', type=int, default=0)
cli_parser.add_argument('--momentum-count', type=int, default=0)
cli_parser.add_argument('--momentum-cash', type=int, default=10000000)
cli_parser.add_argument('--momentum-min-size', type=int, default=1)
cli_parser.add_argument('--momentum-max-size', type=int, default=10)
cli_parser.add_argument('--momentum-wake', type=str, default='20s')

cli_args, remaining_args = cli_parser.parse_known_args()

log_dir = cli_args.log_dir
seed = cli_args.seed
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

start_delta = pd.to_timedelta(cli_args.start_time)
end_delta = pd.to_timedelta(cli_args.end_time)

exch_log_flag = True
order_log_flag = True
book_freq = '20S'

start_wall = dt.datetime.now()
print(f"[config] start_wall={start_wall}")
print(f"[config] seed={seed}\n")

base_day = pd.to_datetime('20010101')
open_ts = base_day + start_delta
close_ts = base_day + end_delta
actor_count, actor_pool, actor_labels = 0, [], []

sym_code = 'AAPL'
cash_start = 10000000
fund_base = 100000
kappa_val = np.log(2) / pd.to_timedelta("30min").value
noise_sigma = fund_base / 1000
lambda_rate = 7e-11

symbol_params = {'r_bar': fund_base,
                 'kappa': kappa_val,
                 'sigma_s': 0,
                 'fund_vol': cli_args.fund_vol,
                 'megashock_lambda_a': 2.7e-18,
                 'megashock_mean': 1e3,
                 'megashock_var': 5e4,
                 'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}
symbols = {sym_code: {'fund_vol': cli_args.fund_vol}}

oracle = Oracle(open_ts, close_ts, sym_code, symbol_params)

history_depth = 25000

actor_pool.extend([Exchange(id=0,
                             name="EXCHANGE_AGENT",
                             type="Exchange",
                             mkt_open=open_ts,
                             mkt_close=close_ts,
                             symbols=[sym_code],
                             log_orders=exch_log_flag,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=history_depth,
                             book_freq=book_freq,
                             wide_book=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
actor_labels.extend("Exchange")
actor_count += 1

noise_ct = cli_args.noise_count
noise_mkt_open = base_day + pd.to_timedelta("00:10:00")
noise_mkt_close = base_day + pd.to_timedelta("23:50:00")
actor_pool.extend([NoiseTrader(id=j,
                          name="NoiseTrader {}".format(j),
                          type="NoiseTrader",
                          symbol=sym_code,
                          starting_cash=cli_args.noise_cash,
                          order_block=cli_args.noise_order_block,
                          log_orders=order_log_flag,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(actor_count, actor_count + noise_ct)])
actor_count += noise_ct
actor_labels.extend(['NoiseTrader'])

num_value = cli_args.value_count
actor_pool.extend([ValueTrader(id=j,
                          name="ValueTrader {}".format(j),
                          type="ValueTrader",
                          symbol=sym_code,
                          starting_cash=cli_args.value_cash,
                          sigma_n=cli_args.value_sigma_n if cli_args.value_sigma_n is not None else noise_sigma,
                          r_bar=cli_args.value_r_bar if cli_args.value_r_bar is not None else fund_base,
                          kappa=cli_args.value_kappa if cli_args.value_kappa is not None else kappa_val,
                          lambda_a=cli_args.value_lambda_a if cli_args.value_lambda_a is not None else lambda_rate,
                          log_orders=order_log_flag,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(actor_count, actor_count + num_value)])
actor_count += num_value
actor_labels.extend(['ValueTrader'])

num_mm_agents = 0 if cli_args.mm_type == 'none' else 1

def build_market_maker(idx, actor_id):
    rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))

    if cli_args.mm_type == 'simple':
        mm_wake = '3s'
        return simpleMM(id=actor_id,
                                name="SIMPLE_MM_{}".format(idx),
                                type='simpleMM',
                                symbol=sym_code,
                                starting_cash=cash_start,
                                min_size=cli_args.mm_size,
                                max_size=cli_args.mm_size,
                                wake_up_freq=mm_wake,
                                inventory_limit=cli_args.mm_inventory_limit,
                                log_orders=order_log_flag,
                                random_state=rstate)

    mm_class_map = {
        'RLMM0': RLMM0,
        'RLMM1': RLMM1,
        'RLMM2': RLMM2,
        'RLMM3': RLMM3,
        'RLMM4': RLMM4,
        'RLMM5': RLMM5,
    }

    if cli_args.mm_type in ['rl_baseline', 'rl_tabular'] or cli_args.mm_type in mm_class_map:
        mm_key = 'RLMM0' if cli_args.mm_type in ['rl_baseline', 'rl_tabular'] else cli_args.mm_type
        mm_cls = mm_class_map[mm_key]
        return mm_cls(id=actor_id,
                     name="RL_MM{}_{}".format(mm_key[-1], idx),
                     type=mm_key,
                     symbol=sym_code,
                     starting_cash=cash_start,
                     wake_up_freq=cli_args.mm_wake_up_freq,
                     base_size=cli_args.mm_size,
                     offsets=[1, 2, 3],
                     epsilon=1.0,
                     alpha=0.1,
                     gamma=0.95,
                     inventory_clip=100,
                     spread_clip=40,
                     inventory_bin=10,
                     spread_bin=5,
                     inventory_penalty=1.0,
                     inventory_limit=100,
                     log_orders=order_log_flag,
                     random_state=rstate)

    raise ValueError(f"Unknown mm_type {cli_args.mm_type}")

actor_pool.extend([build_market_maker(idx, j)
               for idx, j in enumerate(range(actor_count, actor_count + num_mm_agents))])
actor_count += num_mm_agents
actor_labels.extend([cli_args.mm_type] * num_mm_agents)

num_zi_agents = cli_args.zi_count
actor_pool.extend([ZITrader(id=j,
                                     name="ZI_TRADER_{}".format(j),
                                     type="ZITrader",
                                     symbol=sym_code,
                                     starting_cash=cli_args.zi_cash,
                                     sigma_n=cli_args.zi_sigma_n,
                                     sigma_s=cli_args.zi_sigma_s if cli_args.zi_sigma_s is not None else symbols[sym_code]['fund_vol'],
                                     kappa=cli_args.zi_kappa if cli_args.zi_kappa is not None else kappa_val,
                                     r_bar=cli_args.zi_r_bar if cli_args.zi_r_bar is not None else fund_base,
                                     q_max=cli_args.zi_q_max,
                                     sigma_pv=cli_args.zi_sigma_pv,
                                     R_min=cli_args.zi_R_min,
                                     R_max=cli_args.zi_R_max,
                                     eta=cli_args.zi_eta,
                                     lambda_a=cli_args.zi_lambda_a,
                                     log_orders=order_log_flag,
                                     random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                               dtype='uint64')))
               for j in range(actor_count, actor_count + num_zi_agents)])
actor_count += num_zi_agents
actor_labels.extend(['ZITrader'])

num_hbl_agents = cli_args.hbl_count
actor_pool.extend([HBLTrader(id=j,
                                            name="HBL_TRADER_{}".format(j),
                                            type="HBLTrader",
                                            symbol=sym_code,
                                            starting_cash=cli_args.zi_cash,
                                            sigma_n=cli_args.zi_sigma_n,
                                            sigma_s=cli_args.zi_sigma_s if cli_args.zi_sigma_s is not None else symbols[sym_code]['fund_vol'],
                                            kappa=cli_args.zi_kappa if cli_args.zi_kappa is not None else kappa_val,
                                            r_bar=cli_args.zi_r_bar if cli_args.zi_r_bar is not None else fund_base,
                                            q_max=cli_args.zi_q_max,
                                            sigma_pv=cli_args.zi_sigma_pv,
                                            R_min=cli_args.zi_R_min,
                                            R_max=cli_args.zi_R_max,
                                            eta=cli_args.zi_eta,
                                            lambda_a=cli_args.zi_lambda_a,
                                            L=2,
                                            log_orders=order_log_flag,
                                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                      dtype='uint64')))
               for j in range(actor_count, actor_count + num_hbl_agents)])
actor_count += num_hbl_agents
actor_labels.extend(['HBLTrader'])

num_momentum_agents = cli_args.momentum_count

actor_pool.extend([TechnicalTrader(id=j,
                             name="TECHNICAL_TRADER_{}".format(j),
                             type="TechnicalTrader",
                             symbol=sym_code,
                             starting_cash=cli_args.momentum_cash,
                             min_size=cli_args.momentum_min_size,
                             max_size=cli_args.momentum_max_size,
                             wake_up_freq=cli_args.momentum_wake,
                             log_orders=order_log_flag,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))
               for j in range(actor_count, actor_count + num_momentum_agents)])
actor_count += num_momentum_agents
actor_labels.extend("TechnicalTrader")

kernel = SimulationCore("MMCompare Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                  dtype='uint64')))

kernelStartTime = base_day
kernelStopTime = close_ts + pd.to_timedelta('00:01:00')

defaultComputationDelay = 50

kernel.launch(agents=actor_pool,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              defaultComputationDelay=defaultComputationDelay,
              oracle=oracle,
              log_dir=cli_args.log_dir)

simulation_end_time = dt.datetime.now()
print(f"[config] end_wall={simulation_end_time}")
print(f"[config] duration={simulation_end_time - start_wall}")
