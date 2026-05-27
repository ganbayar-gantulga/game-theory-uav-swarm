# Game-Theoretic Target Assignment for UAV Swarms

> **Bachelor's Thesis** · Hohai University · School of Artificial Intelligence and Automation · 2026

This repository contains the MATLAB implementation of a UAV swarm target assignment framework using **non-cooperative game theory (Nash Equilibrium)** solved via **Genetic Algorithm** — comparing a Basic GA (BGA) against an Improved GA (IGA) with four adaptive enhancements.

---

## Overview

In multi-UAV strike missions, assigning each UAV to an optimal target is a high-dimensional combinatorial optimization problem. This work models the attacker–defender interaction as a **non-cooperative matrix game** and finds the **mixed-strategy Nash Equilibrium** using evolutionary computation.

**Scenario:** 5 attacking UAVs × 5 defending targets · 5×5 payoff matrix

---

## Results

| Metric | BGA (Chapter 3) | IGA (Chapter 4) | Improvement |
|---|---|---|---|
| Nash Distance (NE) | 0.0120 | **0.0002** | ↓ 98.3% |
| Attacker Payoff E[UA] | 0.9144 | 0.7851 | — |
| Defender Payoff E[UD] | 0.4207 | **0.6252** | ↑ 48.6% |
| Attacker Deviations | 0 / 200 | 0 / 200 | ✅ |
| Defender Deviations | 0 / 200 | 0 / 200 | ✅ |
| Nash Status | CONFIRMED | **CONFIRMED** | — |

> Both algorithms use a fixed random seed (`rng(42)`) for reproducibility.

---

## Key Improvements in IGA

The Improved Genetic Algorithm introduces four adaptive mechanisms over the Basic GA:

| # | Improvement | Description |
|---|---|---|
| 1 | **Adaptive Mutation Rate** `pm(t)` | Stagnation-driven: triggers when no improvement for 20 generations |
| 2 | **Cosine Annealing σ(t)** | Adaptive mutation std dev; expands ×1.8 on stagnation |
| 3 | **Hybrid SBX Crossover** | Combines arithmetic crossover with SBX (η=15, prob=0.50) |
| 4 | **Adaptive Utility Weight** `α₂(t)` | Dynamically rebuilds payoff matrix; evolves 0.20 → 0.95 |

---

## Simulation Parameters

```
Scenario      : n=5 UAVs (attacker)  ×  m=5 targets (defender)
Environment   : γ = 1.0 (ideal sensor conditions)

Target values:
  Strategic importance  Sj = [1.60, 1.40, 1.20, 1.80, 1.00]
  Threat level          Tj = [1.40, 1.60, 1.80, 1.20, 1.50]
  Mobility factor       Mj = [1.00, 0.80, 1.20, 0.60, 1.10]
  Weights               w  = [0.4, 0.4, 0.2]

Resource consumption: β1=0.3 (distance), β2=0.4 (weapon), β3=0.3 (time)
```

| Parameter | BGA | IGA |
|---|---|---|
| Population size | 200 | 300 |
| Max generations | 500 | 2000 |
| Crossover prob `pc` | 0.80 | 0.80 |
| Mutation rate `pm` | 0.10 (fixed) | Adaptive [0.05, 0.50] |
| Mutation std `σ` | 0.18 (fixed) | Cosine annealing [0.02, 0.35] |
| NE convergence threshold | 3×10⁻⁴ | 1×10⁻⁴ |

---

## Visualization Output

Each script produces a **6-panel figure**:

1. Fitness convergence curve (Nash distance vs. generation)
2. Nash Equilibrium strategy distribution (X\* attacker, Y\* defender)
3. Attacker deviation verification (200 random deviation samples)
4. Defender deviation verification (200 random deviation samples)
5. Population diversity evolution (std dev tracking)
6. α₂(t) evolution trajectory *(IGA only; BGA shows fixed line)*

---

## Repository Structure

```
game-theory-uav-swarm/
├── BGA_code.m        # Chapter 3 · Basic Genetic Algorithm
├── IGA_code.m        # Chapter 4 · Improved Genetic Algorithm
├── results/          # Simulation output figures
│   ├── bga_results.png
│   └── iga_results.png
└── README.md
```

---

## Requirements

- **MATLAB R2021a** or later
- No additional toolboxes required (uses base MATLAB functions only)
- OS: Windows / macOS / Linux

---

## How to Run

```matlab
% Clone the repo and open MATLAB
% Navigate to the project folder:
cd('path/to/game-theory-uav-swarm')

% Run Basic GA (Chapter 3) — ~10–30 seconds
run('BGA_code.m')

% Run Improved GA (Chapter 4) — ~60–180 seconds
run('IGA_code.m')
```

> Run each script independently. Do not run both simultaneously.  
> Expected output will be printed to the Command Window alongside the figure.

---

## Thesis Chapter Mapping

| File | Chapter | Topic |
|---|---|---|
| `BGA_code.m` | Chapter 3 | Nash Equilibrium via Basic GA |
| `IGA_code.m` | Chapter 4 | Nash Equilibrium via Improved GA |
| *(mathematical framework)* | Chapter 2 | Game-theoretic modeling (Equations 2.1–2.16) |

---

## Author

**Ganbayar Gantulga** · AI Engineering · Hohai University  
Supervisor: Lecturer Wen Liangdong · School of AI and Automation  
Contact: ganbaayaar0623@gmail.com · [GitHub](https://github.com/ganbayar-gantulga)
