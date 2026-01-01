# GRAM — Geometric Regularized Associative Memory

GRAM (Geometric Regularized Associative Memory) is an associative-memory framework
for robust retrieval and open-set recognition in high-dimensional spaces.

The method uses geometric regularization, field-based scoring, and controlled
annealing to stabilize recall and guide exploration near decision boundaries,
without relying on a single global scalar objective.

From a theoretical perspective, GRAM’s geometric regularization can also be
interpreted through a gravitational lens: memory states are attracted along a
learned representation manifold, where local curvature and field strength guide
retrieval dynamics.

Under this view, GRAM behaves as a gravitational manifold learner, performing
stable recall from multi-objective spaces by minimizing local curvature rather
than optimizing a global loss. The framework supports semi-supervised operation
and physics-structured inference on learned manifolds.

 — core algorithm by Derek Devine.
# GRAM — Geometric Regularized Associative Memory

Developed by **Derek M. Devine** (2025).

This repository contains the core GRAM algorithm and supporting research code for
structured associative memory, open-set retrieval, and controlled exploration of
ambiguous states.

GRAM is intended as a research framework for experimentation, extension, and
collaborative development.

See `LICENSE` for usage terms and `CITATION.cff` for citation guidance.


GRAM/
  __init__.py            # to treat as a GRAM package

  gram.py               # core GRAM/HOSAM implementation
                         # - SPR/gamma, gamma_factor
                         # - φ-scheduled bit flips (pre-denoise + NEAR)
                         # - jammed PMF multi-index
                         # - tau_mode (linear/phi), use_phi_thresholds
                         # - evaluate(), corrupt_bits()

  gram_recursive.py      # higher-level wrappers:
                         # - GRAMTime
                         # - GRAMCoarseFine
                         # - GRAMFeatureSelector
                         # - GRAMFrontierEstimator

  gram_variants.py       # (optional convenience)
                         # - alias: from hosam import HOSAM as GRAM
                         # - def GRAM_plain(...)
                         # - def GRAM_phiJ(...)
                         # - def GRAM_phiFull(...)

  gram_robustness_experiment.py
                         # - test 0% to 50% bit-flip robustness
                         # - prints TPR/FPR/time_ms across seeds

  frontier_examples.ipynb
                         # - notebook to:
                         #   * load Sweep_Updated.csv
                         #   * fit frontier
                         #   * compute d_front
                         #   * compare GRAM vs baselines vs φ-variants

  README.md              # high-level description:
                         # - what GRAM is (Gravitationally Regularized AM)
                         # - how it relates to GID, SR, UCEG
                         # - how to run robustness / frontier experimentsGRAM/

            
**GRAM** (Gravitationally Regularized Associative Memory) is a
highly structured associative memory model that integrates:

- **Gravitational Information Dynamics (GID)**  
- **Structured Randomness (SR)** via φ-annealed stochastic flux  
- **Spectral diagnostics** (SPR, γ) of the Hebbian field  
- **Jammed-PMF neighborhood sampling** from CA saturation  
- **Multi-channel certification** (energy gap, signatures, minhash)

GRAM generalizes classical associative memories (Hopfield, MHN, DAM, etc.)
into a *gravitationally regularized, recursively annealed* architecture
that is both **robust under noise** and **aligned with multi-objective
frontiers** (for example, Cost–SSR Pareto curves).

---

## Conceptual summary
In my GID / SR framework, system evolution is governed by a flux law:
\[
R_\phi(m_i) = \Delta A + \nabla_A E,
\]
where:
- \(\Delta A\) is actual change in the state,
- \(\nabla_A E\) is the curvature (gradient of an effective potential),
- \(R_\phi\) is a **structured stochastic flux**, scaled by the inverse golden ratio \(\phi^{-1}\)
  and modulated by local margins, spectral gaps, and adjacency structure.

GRAM implements this law at the level of associative recall:
- The Hebbian matrix \(W\) and score \(S'(v) = \tfrac12 v^\top W v\) provide **curvature**.
- Hamming distance, margins \(m_i = v_i (Wv)_i\), and the empirical \(\bar{\Delta S'}\)
  define a **stochastic noise scale**.
- φ-annealed guided dither (bit flips with \(K_t = \phi\_K(K_0, t)\)) supplies **structured flux**.
- Multi-channel cert (score gap, ones-count, pair signatures, minhash) enforces a
  **flux budget**: only when deviations are consistent with structured randomness
  is a recall accepted as KNOWN (otherwise NEAR-UNKNOWN / UNKNOWN).

GRAM (Gravitationally Regularized Associative Memory) is an associative memory model that combines Hebbian fields with gravitational information dynamics: SPR/γ-based diagnostics, Feigenbaum–golden φ-annealed guided dither, jammed-PMF neighborhood sampling, and multi-channel certification of matches. GRAM generalizes Hopfield-style recall into a gravitationally regularized, open-set, manifold-aware selectors that maiintain the structure of the manifold, which is a N-D space that is dependent on the nnumber of variables N, being optimized in the model.

GRAM isn’t the first ever physics-based AI but it’s likely the first associative memory / retrieval algorithm that explicitly combines saturation PMFs (jammed neighbor saturation leading to exponentially less likelihood of updates occuring -> saturation convergence in distribution), Feigenbaum / φ-criticality, structured randomness, spectral diagnostics (γ) and manifold-preserving behavior (curvature matching) in a single coherent physics framework.  Making GRAM is a new class of energy-based associative memory; a Geometry-Preserving Associative Algorithm.
Simulated annealing is physics-inspired search, while GRAM is physics-structured retrieval and inference on a learned manifold.

Simulated annealing vs GRAM: similarities and differences
Simulated annealing (SA):
•Generic global optimization heuristic.
•Has a temperature T, that is slowly lowered.
At each step:
•It proposes a new state -- usually local random perturbation
•Accepts it with probability depending on ΔE and Temperature.
It just explores and doesn’t have explicit training in the ML sense
GRAM:
•Takes a learned manifold (pattern bank / memory)
Uses:
•a gravitational metric to measure distance to patterns
•GID-based structured randomness (Rφ(m)) for updates, using this scaling law: 
•jammed PMF to propose structured perturbations
•spectral diagnostics (γ) to adapt annealing
•and multichannel certification / gating
•GRAM’s job is not to solve arbitrary optimization problems over an abstract space like SA does, instead GRAM was designed to recall manifold-consistent states in a learned representation.
SA and GRAM are related in spirit (both do annealed stochastic search), but: SA is a generic heuristic, while GRAM is a physics-structured machine for manifold-aware memory / selection.

DEFINTION:
GRAM: A New Class of Gravitational, Geometry-Preserving Associative Algorithms
The GRAM family (GRAM, GRAM-FS, GRAM-φFull, GRAM-FeigCooling) forms a fundamentally new class of associative algorithms defined by gravitational recall, multi-objective manifold preservation, and physics-inspired feature selection.
Where classical associative memories (Hopfield, BAM, MHN) operate by maximizing similarity, GRAM instead performs gravitational descent on an information manifold. Each stored pattern is treated as a mass embedded in a Pareto surface (Cost–SSR–SCR), and queries descend toward regions of high geometric and physical consistency rather than simple nearest neighbors. This allows GRAM to replicate not only the values of a multi-objective frontier, but its curvature, slope, and global topology—something no classical AM or supervised learner achieves.
The GRAM-FS extension further integrates recursive (which may or may not be percolation-based-this is still under investigation)  feature selection, retaining only the subset of features that stabilize manifold geometry.  It performs feature selection by training on a subset of the data and then iteratively removing a feature, and refitting to the manifold and checking to see how much error was introduced in the process.  In this way the most important features that preserve that sub-manifold space are identified and quantified.  Unlike Random Forest importances or L1 regularization, GRAM-FS selects features because they preserve the physical structure of the manifold, not because they maximize predictive variance.
GRAM’s open-set mechanism introduces three cognitive states—KNOWN, NEAR-UNKNOWN, UNKNOWN—making GRAM the first associative memory with formal geometric uncertainty.
Finally, GRAM-FeigCooling adds Feigenbaum-modulated annealing to the retrieval rule, dramatically improving manifold reconstruction in low-data regimes. At low N values, GRAM-FeigCooling achieves R² ≈ TRUE, RMSE and curvature errors far below MHN, and global manifold alignment unmatched by any existing AM or supervised model.
In total, GRAM is the first algorithm class to unify:
1.	Associative memory
2.	Gravitational dynamics
3.	Manifold preservation
4.	Physics-based feature selection
5.	Open-set recognition   
This places GRAM in a unique region of algorithmic space—not a variant of Hopfield, SOM, or backprop, but a new synthesis with capabilities none of those frameworks possess.  GRAM is not a classical associative memory at all.  
GRAM is a gravitational manifold learner that performs stable recall from multi-objective spaces by minimizing local curvature rather than a global scalar objective.  It’s semi-supervised, and performs physics-structured retrieval and inference on a learned manifold.
It excels when the task involves:
1.	High-dimensional data embedded on a lower-dimensional manifold
2.	Trade-offs (frontiers, Pareto surfaces, energy landscapes)
3.	Chaotic / noisy environments where stability matters
4.	Non-symmetric dynamics (unlike Hopfield, which requires symmetric weights)
5.	Multi-modal or multi-objective recall
USE CASES: 
GRAM will be used most likely in the following applications:
1.	Radiology, MRI, EEG 
2.	Complex systems (disease modeling, climate, irrigation, ecological modeling, fluids, turbulence)
3.	AI: next-generation associative memories
4.	Biomedical multi-objective modeling
5.	Neuroscience memory and perception models
6.	Aerospace/NASA design frontiers
7.	Genomics/proteomics manifold modeling
8.	Economics & risk frontier modeling
9.	Inverse problems and missing data
Anywhere the world has a manifold, not a list of points, GRAM is relevant.
For example with respect to medical radiological imaging data: 
Classical AMs (Hopfield, MHN) and Transformers struggle for multiple reasons, among some of them, because: medical signals lie on complex frontiers (cost vs sensitivity vs genomic markers vs spatial anatomy), noise is highly structured (like structured randomness theory), outliers are common, missing-data imputation must maintain structure (not just produce a low error), and manifold collapse (mean-optimization) is dangerous.  

In contrast GRAM preserves the manifold in the presence of missing daa (applicable to partial radiology scans, missing MRI slices, sparse EEG channels).  It has noise robust pattern completion using GRAM’s φ-annealing which is more powerful than Hopfield’s energy descent.  It includes Multi-objective decision support (balancing cancer sensitivity vs specificity vs toxicity), with stable recall in highly variable presentation of data that may be the same underlying pathology but whose etiology varies from patient to patient.  It also has reliability modeling inherent in the design, as it performs best in multi-objective scenarios, and improves in relative performance compared to other algorithms as the number of variables modeled increases.  Its most compelling use case is Radiological anomaly retrieval under structured noise. This is where Hopfield networks previously tried — and failed — to replace CNNs.  Transformers struggle because they optimize predictive loss, not geometric stability.
Hopfield-like systems fail due to lack of multi-objective logic.  But, GRAM, with jammed-PMF + NEAR recursion, is actually suited for it.

THE FUTURE OF GEOMETRY PRESERVING ASSOCIATIVE ALGORITHMS 
In summary, GRAM is a strict generalization of energy-based Ams, that’s stable under massive noise, can reconstruct multi-objective manifolds, not just fixed points, avoids mode collapse, and supports open-ended recall (unknown → NEAR → fallback).  Modern Hopfield Networks (MHN) fail at: manifold geometry, missing data, multi-objective tradeoffs, open-set uncertainty, but GRAM succeeds in all four.
Given this it’s very likely GRAM and its sucessors will replace MHN and Hopfield in: structured recall tasks, multi-modal retrieval, high-reliability recall settings, and generally in any domain where the geometry matters.  Transformers won’t replace this because they operate on sequence/logit spaces, not geometric manifolds.
The motto of GRAM is: 
Stability arises not from suppressing randomness but from matching its scale to the energy gradient through weighted feedback. This generalizes the notion of gravitational equilibrium to the domain of associative learning; using Structured Randomness and Gravitational Information Dynamics.  And demonstrates that when stochastic perturbations are guided, they can be harnessed to stabilize and enhance complex systems.
In the context of BAM frameworks, I previously used a guided two–bit controller restored stable recall in regimes that would otherwise exhibit attractor collapse, directly validating the theoretical premise of gravitational annealing I’d developed in previous studies of SOM.  These concepts were then integrated into GRAM, which harnesses the emergent balance between order and randomness and suggests there are universal scaling law(s) for self–organization.  GRAM is the first demonstration of these techniques (and potential laws) being modeled and harnessed.

## Installation & Dependencies

### Python version

GRAM has been developed and tested primarily with:

- **Python 3.9–3.12**

Earlier versions may work but are not guaranteed.

### Core dependencies

Minimal runtime dependencies for the core GRAM engine (`gram.py`):

- `numpy` (>= 1.22)

Optional but recommended:

- `numba` (for JIT speedups of Hamming distance and S' computations)

To install the minimal stack:
```bash
pip install numpy

To enable JIT acceleration:
pip install numba

Frontier / analysis extras
For frontier fitting, plotting, and notebooks you will also want:
•	pandas (for sweep data like Sweep_Updated.csv in the colab llinked below)
•	matplotlib (for figures)
•	jupyter / notebook or jupyterlab (for exploration)

pip install pandas matplotlib jupyter

General Colab link that contains extra benchmarking scripts for the multi-energy frontier fitting proof of concept/use case:
https://colab.research.google.com/drive/1CTk4bNcqYMX_h1c4onQuJCSglXSMsFAs?usp=sharing


Using GRAM from this repo
If you clone the repo:

git clone https://github.com/DDevine-source/GRAM.git
cd GRAM

pip install -e .

from hosam import HOSAM as GRAM
from gram_variants import GRAM_plain, GRAM_phiFull
from gram_recursive import GRAMFrontierEstimator

Quick start
1. Basic GRAM usage

At the top of evey script:
import numpy as np
from hosam import HOSAM as GRAM
from gram_variants import GRAM_plain, GRAM_phiFull #optional, as well as adding the other GRAM variants here
from gram_recursive import GRAMFrontierEstimator #optional
------------------------
# Example code:
# patterns_bits01: [P, N] array of {0,1} patterns
patterns_bits01 = np.random.randint(0, 2, size=(256, 64), dtype=np.uint8)

gram = GRAM(patterns_bits01)

query_bits = np.random.randint(0, 2, size=(64,), dtype=np.uint8)
decision, idx, meta = gram.query(query_bits)
print(decision, idx)


## License

This project is licensed under the BSD 3-Clause License. See the `LICENSE` file for details.

## Citation

If you use GRAM in academic work, please cite it using the metadata in `CITATION.cff`.

