# Genetic Programming Market Uptrend Predictor

## 🎯 Project Overview

This project uses **Genetic Programming (GP)** to discover early signals that predict market uptrends before they actually happen. Think of it like teaching a computer to evolve its own trading strategies by learning from historical market patterns.

### The Core Idea
- **Goal**: Predict when EMA 12 will cross above EMA 50 (indicating an uptrend) before it actually happens
- **Method**: Use genetic programming to evolve mathematical expressions that can spot early warning signs
- **Why**: If we can detect uptrends 5-10 days early, we could potentially enter positions before the crowd notices

### Simple Example
Instead of waiting for this to happen:
```
Day 10: EMA 12 crosses above EMA 50 → Everyone buys → Price already moved up
```

We want to detect patterns like:
```
Day 5: Our evolved algorithm spots unusual volume + price patterns
Day 6: Algorithm says "uptrend coming soon"
Day 10: EMA 12 crosses above EMA 50 → We're already positioned
```

## 🧬 How Genetic Programming Works Here

1. **Population**: Start with random mathematical expressions (like DNA)
2. **Evolution**: The best-performing expressions "breed" to create new ones
3. **Selection**: Expressions that correctly predict uptrends survive
4. **Mutation**: Small random changes keep exploring new patterns
5. **Repeat**: Over generations, we evolve better prediction formulas

## 📊 Target Definition

**Uptrend Signal**: When EMA 12 > EMA 50
- EMA = Exponential Moving Average
- This is a common technical indicator traders use
- We want to predict this crossover 3-10 days before it happens
