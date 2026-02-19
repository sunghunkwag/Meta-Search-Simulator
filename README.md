# Meta-Search Simulator

An experimental neuro-symbolic framework for autonomous algorithm discovery through constraint synthesis, grammar-guided evolution, and adaptive meta-optimization.

## Overview

This repository implements a simulation framework for studying meta-search and self-modifying algorithmic structures under constrained, artificial task environments.

The system combines symbolic reasoning (constraint synthesis) with neural evolution (Tesseract Engine) under a metacognitive routing layer, enabling adaptive problem-solving across multiple difficulty levels.

## Features

- **Symbolic Reasoning**: Mathematical pattern extraction (linear, quadratic, modulo)
- **Neuro-Evolution**: Tesseract-based concept space exploration via NumPy autoencoder + GMM
- **Metacognitive Routing**: Dynamic symbolic/neural reasoning allocation based on task complexity
- **Grammar-Guided GP**: Strategy selection and hole-filling via evolutionary search
- **Autonomous Curriculum**: Self-adjusting task difficulty based on rolling success rate
- **Performance-Driven Optimization**: Hill-climbing hyperparameter tuning layer
- **Void-Logic Synthesis**: Experimental exploration of non-human-readable expression structures
- **Checkpoint & Rollback**: Automatic state preservation and collapse recovery

## Architecture

```
InventionMetaController
├── MetacognitiveController   (Task Complexity Routing)
├── ConstraintSynthesizer     (Symbolic Logic: linear / quadratic / modulo / void)
├── EvolutionarySearcher      (Grammar-Guided GP + Hole Filling)
├── TesseractEngine           (Neuro-Evolution: AE + GMM + ES Operators)
├── PerformanceSelfImprover   (Hyperparameter Hill-Climbing)
└── CheckpointManager         (State Preservation + Rollback)
```

## Installation

```bash
git clone https://github.com/sunghunkwag/Meta-Search-Simulator.git
cd Meta-Search-Simulator
pip install -r requirements.txt
```

## Usage

```bash
# Self-test (2 generations)
python UNIFIED_RSI_EXTENDED.py selftest

# Infinite evolution loop
python UNIFIED_RSI_EXTENDED.py evolve

# Resilience test (poison strategy injection)
python UNIFIED_RSI_EXTENDED.py resilience-test
```

## File Structure

```
Meta-Search-Simulator/
├── UNIFIED_RSI_EXTENDED.py   # Main system (RSI engine)
├── numpy_tesseract.py        # Neuro-evolution engine (NumPy-only)
├── atomic_factory.py         # AST node factory primitives
├── requirements.txt          # Dependencies
├── LICENSE
└── README.md
```

## Experimental Status

> ⚠️ Research prototype. Not production-ready.

- Multiprocessing stability on Windows may require `spawn` start method
- Void-logic synthesis is empirically exploratory
- Hyperparameter optimization layer approximates meta-optimization, not full code self-modification
- Requires `numpy` only — no deep learning framework dependency

## License

MIT — see [LICENSE](LICENSE)
