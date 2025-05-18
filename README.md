# Genetic Programming for Pre‑Uptrend Pattern Discovery

> **Goal** Automatically evolve human‑readable trading rules that tend to appear *just before* an uptrend, where an **uptrend** is formally defined as `EMA 12 > EMA 50` on the very next bar.

---

## 1. Project Snapshot

| Item           | Details                                                                                                                             |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Target Pattern | Conditions that often occur **one bar** before `EMA 12` crosses above `EMA 50`.                                                     |
| Technique      | Symbolic Regression / Classification via **Genetic Programming (GP)** using the [DEAP](https://github.com/DEAP/deap) framework.     |
| Data           | OHLCV candles + technical indicators in a tidy CSV or Parquet file. Any timeframe works (15 min, 1 h, daily…), but must be uniform. |

*The end product is a concise logic tree such as:*
`(RSI(7) < 32) AND (Close > SMA(20)) AND (MACD_Hist < 0)`
…which you can drop into your own back‑test or live system.

---

## 2. How It Works

1. **Data Ingestion**: Load candles, forward‑fill gaps, compute EMA 12 & EMA 50.
2. **Label Generation**: Create the boolean column `IsUpcomingUptrend` such that

   ```text
   IsUpcomingUptrend[t] = (EMA12[t+1] > EMA50[t+1])
   ```
3. **Primitive Set**: Expose mathematics (`+, −, ×, ÷, max, min`), comparison ops, and indicator functions (RSI, ATR, Bollinger Bands, etc.).
4. **GP Evolution**: 

   * PopulationSize individuals → evaluate on train set.
   * Fitness = **F1 Score** (harmonising precision & recall) with 10‑bar look‑ahead penalty (optional).
   * Elitism preserves the top `EliteSize` rules.
   * Standard genetic operators: subtree crossover, point mutation, size limits to avoid bloat.
5. **Validation**: After `Generations` cycles, champion rules are measured on an unseen test split and via walk‑forward validation.
6. **Export**: Winning expression trees are converted to plain Python, NumPy, or Pine Script.

---
