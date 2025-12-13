# Reinforcement Learning Market Makers in a Glosten-Milgrom Environment

This repository implements a discrete-event limit order book (LOB) simulator with multiple trader types, including tabular reinforcement-learning (RL) market makers, in a Glosten–Milgrom–style information environment. It was developed as part of the Columbia University course ORCSE4529 (Reinforcement Learning) taught by Prof. Shipra Agrawal, and is used to compare RL market-making policies against classical baselines.

**The full project report is available here: [`Reinforcement Learning Market Makers in a Glosten-Milgrom Environment - Bennouna, Jelenkovic, Karam, Souaid.pdf`](./Reinforcement Learning Market Makers in a Glosten-Milgrom Environment - Bennouna, Jelenkovic, Karam, Souaid.pdf).**

---

## Overview

This project examines whether a simple, uninformed RL agent can operate as a market maker in a Glosten–Milgrom asymmetric-information environment, where informed traders observe noisy signals of a latent fundamental while the market maker cannot. The question is not whether the market maker can be profitable (it cannot, in expectation), but whether reinforcement learning can **minimize unavoidable losses** while providing liquidity.

The simulator includes:

- A latent fundamental process (Ornstein–Uhlenbeck + jump shocks).
- Informed traders (Value, ZI, HBL agents).
- Uninformed traders (Noise, Technical).
- A heuristic market maker.
- Several RL market makers (RLMM0–RLMM4).
- An event-driven kernel implementing Glosten–Milgrom–style asynchronous interactions.

---

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate glosten-milgrom-rl
```

---

# Configuration & Entrypoint

From repository root:

```bash
python main.py -c config
```

- **`config.py`** — builds the oracle, exchange, and trader population; parses CLI arguments; launches simulations.
- **`main.py`** — wrapper for running a configuration module.

---

# Configuration Arguments

### General
- **`--log-dir`** — Directory where all simulation logs will be written (default: auto-generated timestamped folder).
- **`--seed`** — Random seed for reproducibility; if omitted, a time-based seed is used.
- **`--start-time`, `--end-time`** — Market session boundary times (e.g., `09:00`, `16:00`).
- **`--fund-vol`** — Volatility parameter of the latent fundamental value process in the Oracle.

---

### Market Maker Parameters
- **`--mm-type`** — Which market maker to include:  
  `{none, simple, RLMM0, RLMM1, RLMM2, RLMM3, RLMM4}`.
- **`--mm-size`** — Quote size (number of shares) for the market maker’s limit orders.
- **`--mm-wake-up-freq`** — How often the MM wakes (e.g., `"10s"` for every 10 seconds).
- **`--mm-inventory-limit`** — Hard cap on absolute inventory; prevents runaway positions.

---

### Noise Traders
- **`--noise-count`** — Number of uninformed noise traders.
- **`--noise-cash`** — Initial cash per noise trader.
- **`--noise-order-block`** — Base order size for noise-trader market orders.

---

### Value Traders (Informed)
- **`--value-count`** — Number of Value-based informed traders.
- **`--value-sigma-n`** — Noise level in their oracle observation.
- **`--value-r-bar`** — Long-run mean used in the Value trader’s OU belief update.
- **`--value-kappa`** — Belief mean-reversion speed.
- **`--value-lambda-a`** — Arrival/wake intensity of Value traders (controls how often they trade).

---

### Zero-Intelligence (ZI) & HBL Traders
- **`--zi-count`** — Number of ZI traders.
- **ZI valuation/noise/private-value/surplus parameters:**
  - **`--zi-sigma-n`** — Observation noise of their oracle signal.
  - **`--zi-sigma-s`** — Belief stationary variance; influences valuation uncertainty.
  - **`--zi-kappa`** — Belief mean-reversion parameter.
  - **`--zi-r-bar`** — Long-run mean used in ZI belief update.
  - **`--zi-q-max`** — Max inventory target for ZI agents.
  - **`--zi-R-min`, `--zi-R-max`** — Minimum/maximum surplus tolerance for limit order prices.
  - **`--zi-lambda-a`** — Arrival rate (wake frequency).
- **`--hbl-count`** — Number of HBL agents (a belief-learning extension of ZI).

---

### Technical Traders (Momentum-based)
- **`--momentum-count`** — Number of Technical traders.
- **`--momentum-cash`** — Initial cash allocation.
- **`--momentum-min-size`, `--momentum-max-size`** — Range of market-order sizes they submit.
- **`--momentum-wake`** — Deterministic wake interval (e.g., one trade every N seconds).---

# Example Experiments

### Noise-only market with RLMM1
```bash
python main.py -c config \
  --mm-type RLMM1 \
  --noise-count 50 \
  --value-count 0 \
  --zi-count 0 \
  --log-dir logs/rlmm1_noise_only
```

### Mixed market with RLMM2
```bash
python main.py -c config \
  --mm-type RLMM2 \
  --noise-count 50 \
  --zi-count 20 \
  --value-count 10 \
  --log-dir logs/rlmm2_mixed
```

### Baseline heuristic market maker
```bash
python main.py -c config \
  --mm-type simple \
  --noise-count 50 \
  --log-dir logs/simplemm_baseline
```

---

# Extending the Project

To create a new agent:

1. Create a class in `traders/` inheriting from `BaseTrader`.
2. Implement:
   ```python
   kernelStarting(self, startTime)
   wakeup(self, currentTime)
   receiveMessage(self, currentTime, msg)
   getWakeFrequency(self)
   ```
3. Use BaseTrader utilities:
   - `placeLimitOrder`, `placeMarketOrder`, `cancelOrder`
   - `getKnownBidAsk`, `getOrderStream`, `getLastTrade`
   - `logEvent`
4. Register your agent in `config.py`.

---

# Calibration Parameters

Full calibration tables are provided in **`Report.pdf`**, Appendix A.
