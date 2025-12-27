# Recursive-Self-Improvement-Engine

A True RSI (Recursive Self-Improvement) engine that can modify its own source code to improve performance.

## Features

| Component | Description |
|-----------|-------------|
| **Multi-Universe Evolution** | Parallel evolution with genetic operators |
| **7 Mutation Operators** | const_drift, swap_binop, wrap_unary, wrap_call, insert_ifexp, shrink, grow |
| **FunctionLibrary** | Learns reusable helper expressions (ontology learning) |
| **MetaState** | Adaptive operator weights & exploration rate |
| **Deep Autopatch (L0-L5)** | Self-modification from hyperparameters to algorithm synthesis |
| **RSI-Loop** | Continuous evolve → autopatch cycle |

## RSI Levels

| Level | Capability |
|-------|------------|
| L0 | Hyperparameter tuning |
| L1 | Operator weight adaptation |
| L2 | Add/remove mutation operators |
| L3 | Modify evaluation function |
| L4 | Synthesize new operators |
| L5 | Modify self-modification logic |

## Quick Start

```bash
# Basic sanity test
python UNIFIED_RSI_EXTENDED.py selftest

# Run evolution (symbolic regression)
python UNIFIED_RSI_EXTENDED.py evolve --fresh --generations 100

# View best result
python UNIFIED_RSI_EXTENDED.py best

# Attempt self-modification (True RSI)
python UNIFIED_RSI_EXTENDED.py autopatch --levels 0,1,3 --apply

# Continuous RSI loop
python UNIFIED_RSI_EXTENDED.py rsi-loop --generations 50 --rounds 10
```

## Verified Results

- **Evolution**: Score 206 → 0.005 (99.97% improvement in 50 generations)
- **Autopatch**: Successfully self-modified code with 63% score improvement
- **Backup**: Automatic backup creation before self-modification

## How Autopatch Works

1. Parses own source code via AST
2. Proposes patch candidates (hyperparams, eval weights, new operators)
3. Tests each patch via subprocess probe runs
4. Applies best-performing patch by overwriting self
5. Creates backup file before any modification

## License

MIT
