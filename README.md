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

## 📁 Project Structure

### Core Files

- **`main.py`**: Entry point that orchestrates the entire GP workflow - downloads data, cleans it, labels it, runs evolution for 200 generations with population of 3000

- **`DataDownloader.py`**: Downloads market data from Yahoo Finance with support for chunked hourly data retrieval to work around API limitations

- **`DataCleaning.py`**: Cleans and validates market data by checking OHLC relationships, removing outliers, and handling missing values

- **`DataLabel.py`**: Labels data with prediction targets by identifying EMA crossovers and marking signals 3-10 days before they occur

- **`GPFramework.py`**: High-level orchestrator that provides a clean API to initialize and configure the entire GP framework

### GP Framework Components (`GP_Framework/`)

- **`FitnessEvaluator.py`**: Evaluates GP individuals using F1 score with early detection bonus - rewards accurate predictions that come days before the actual crossover

- **`GeneticOperators.py`**: Defines genetic operators (crossover, mutation, selection) that create new individuals during evolution

- **`PopulationManager.py`**: Manages GP population creation and initialization using DEAP framework

- **`PrimitiveSetBuilder.py`**: Builds the primitive set defining all available functions and terminals for constructing trading signals

### Technical Indicators (`GP_Indicators/`)

- **`BasicIndicators.py`**: Standard technical indicators (EMA, RSI, MACD, Bollinger Bands, ATR, Momentum) wrapped for GP use

- **`MomentumIndicators.py`**: Calculates momentum across multiple timeframes (hourly, daily, weekly) and their alignment

- **`MultiTimeframe.py`**: Calculates indicators on different timeframes to enable multi-timeframe analysis

- **`PositionEncoding.py`**: Encodes price position relative to timeframe ranges and moving averages for context

### GP Primitives (`GP_Primitives/`)

- **`ComparisonPrimitives.py`**: Comparison and logical operators (>, <, AND, OR, IF-THEN-ELSE) for conditional logic

- **`MathPrimitives.py`**: Basic mathematical operations (protected division, min/max, sqrt, log, trig functions)

- **`TradingPrimitives.py`**: Trading-specific operations like crossovers, lags, and price changes for pattern detection

