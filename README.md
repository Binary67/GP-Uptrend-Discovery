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

## 🚀 Strongly Typed GP Implementation Tasks

### Phase 1: Type System Design
- [ ] Define core type hierarchy (float, series, bool, int)
- [ ] Design type promotion rules (e.g., float + series = series)
- [ ] Create type checking utilities and validators
- [ ] Document type system specifications

### Phase 2: Primitive Set Refactoring
- [ ] Modify PrimitiveSetBuilder to use DEAP's PrimitiveSetTyped
- [ ] Add type registration system for primitives
- [ ] Update primitive registration to include input/output types
- [ ] Create typed terminal generation (typed constants, variables)

### Phase 3: Primitive Type Annotations
#### GP_Primitives/
- [ ] Add type signatures to MathPrimitives (Add, Sub, Mul, Div, etc.)
- [ ] Create separate typed versions for scalar vs series operations
- [ ] Update ComparisonPrimitives with bool return types
- [ ] Type TradingPrimitives (Buy, Sell signals)

#### GP_Indicators/
- [ ] Type BasicIndicators (EMA, SMA, etc.) - series input/output
- [ ] Type MomentumIndicators (RSI, MACD, etc.)
- [ ] Type MultiTimeframe operations
- [ ] Type PositionEncoding functions

### Phase 4: Genetic Operators Update
- [ ] Implement type-aware crossover (only swap compatible subtrees)
- [ ] Create typed mutation operators respecting type constraints
- [ ] Update bloat control for typed trees
- [ ] Add type-safe subtree selection methods

### Phase 5: Population Management
- [ ] Modify tree generation to use typed primitive set
- [ ] Implement type-aware genHalfAndHalf/genFull/genGrow
- [ ] Update individual creation with type constraints
- [ ] Ensure initial population type correctness

### Phase 6: Fitness Evaluation
- [ ] Remove runtime type checking in EvaluateIndividual
- [ ] Update compilation to leverage type guarantees
- [ ] Simplify error handling (type errors impossible)
- [ ] Optimize evaluation with type information

### Phase 7: Testing & Validation
- [ ] Create unit tests for typed primitives
- [ ] Test type-aware genetic operators
- [ ] Validate type system with edge cases
- [ ] Performance comparison (typed vs untyped)

### Phase 8: Documentation & Migration
- [ ] Document new type system usage
- [ ] Create migration guide from untyped to typed
- [ ] Update examples with typed GP
- [ ] Add type system design rationale

### Estimated Effort
- **Total Tasks**: ~35 major items
- **Complexity**: High - requires careful design to maintain expressiveness
- **Benefits**: 
  - Eliminate runtime type errors
  - Faster evolution (invalid trees never generated)
  - Clearer primitive semantics
  - Better code maintainability


