# GP-Uptrend-Discovery

*Genetic Programming proof-of-concept that automatically rediscovers the classic “EMA 12 > EMA 50” up-trend rule, using Python’s `gplearn`.*

---

## 🎯 Objective
Demonstrate how a lightweight Genetic Programming (GP) pipeline can evolve trading rules and validate them by independently converging on a **known** rule:

> **Up-trend ⇢ 12-period EMA is higher than 50-period EMA.**

Once the pipeline is proven, you can swap in additional indicators or price series and let GP hunt for new alpha-generating logic.

---

