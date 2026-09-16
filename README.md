# Uncertainty-Aware Structural Assessment

Computational research on **measurement uncertainty, mechanics-based corrosion assessment, and uncertainty propagation** in damaged infrastructure.

This repository supports my M.Sc. research in Structural Engineering at the University of Alberta and focuses on propagating in-line inspection (ILI) uncertainty through **RSTRENG** and **PSQR** assessment workflows rather than applying uncertainty only after a deterministic capacity estimate has already been produced.

## Research focus

The implementation includes two related analysis paths:

1. **RSTRENG profile-level Monte Carlo analysis**
   - Reconstructs the river-bottom profile from a two-dimensional corrosion grid.
   - Perturbs measured corrosion depth and axial sizing within specified uncertainty bounds.
   - Repeats the full RSTRENG effective-area search for every Monte Carlo realization.
   - Tracks failure-pressure and governing-defect statistics together with convergence diagnostics.

2. **Nested PSQR uncertainty propagation**
   - Samples ILI depth, feature length, and feature width uncertainty at the level where those uncertainties enter the assessment.
   - Coherently rescales the full corrosion grid for sampled feature dimensions rather than perturbing every nominal cell independently.
   - Generates many plausible PSQR paths inside each outer ILI-uncertainty realization.
   - Uses depth-weighted path initiation and axially ordered profiles.
   - Evaluates the PSQR lower-tail pressure statistic, effective corrosion dimensions, and Axial Alignment Factor (AAF).

## PSQR implementation details

The current implementation incorporates the following audited revisions:

- **Strip-based AAF:** whole-anomaly RSTRENG is compared with circumferential `6t × L` strip assessments; the minimum strip-to-whole pressure ratio defines AAF.
- **Interaction window:** candidate PSQR transitions use a centered window of `±max(6t, 25.4 mm)`, corresponding to a total width of `max(12t, 50.8 mm)`.
- **Feature-level sizing uncertainty:** reported length and width uncertainty are sampled once per outer realization and used to coherently rescale the grid.
- **Depth-weighted starting locations:** deeper reported cells are more likely to initiate a plausible path while all positive-depth cells remain possible.
- **Reproducibility safeguards:** checkpoint signatures include the algorithm version, source-code SHA-256, input-file metadata, model parameters, uncertainty settings, sample counts, and random seed.
- **Checkpointing and restart:** long production runs can resume without silently reusing results from an incompatible algorithm or input configuration.

The PSQR workflow intentionally focuses on **P5 and effective corrosion dimensions** rather than reporting a probability of failure from the PSQR path.

## Input format

The analysis reads a **headerless Excel matrix** representing a two-dimensional ILI corrosion-depth grid.

- Finite cells represent reported corrosion measurements.
- Depth values are interpreted as **percent of wall thickness**.
- `NaN` cells represent locations outside the reported feature or unavailable grid cells.
- Nominal axial and circumferential cell dimensions are supplied separately in millimetres.

The restricted inspection dataset used in the industry-linked research is not distributed through this repository. A synthetic demonstration grid is included so the public workflow can be exercised without proprietary data.

## Reproducibility defaults in the research script

The current PSQR implementation uses configurable parameters, with research defaults including:

```text
Depth sizing bound       : ±0.07 t
Length sizing bound      : ±7 mm
Width sizing bound       : ±9 mm
Confidence/certainty     : 80%
Nominal axial cell size  : 5 mm
Nominal grid width       : 5 mm
PSQR profiles / outer MC : configurable
Random seed              : 2026
```

Material properties, operating pressure where applicable, Monte Carlo sample counts, convergence settings, and worker count are also configurable.

## Output

The workflow produces numerical summaries and publication-oriented diagnostics including:

- nominal and Monte Carlo pressure statistics;
- PSQR P5 distributions;
- effective depth and effective length statistics;
- AAF and selected assessment-method diagnostics;
- outer Monte Carlo convergence;
- inner plausible-profile-count convergence;
- checkpoints, run metadata, logs, and reproducibility signatures;
- vector PDF and high-resolution PNG figures.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── CITATION.cff
├── data/
│   └── README.md
├── src/
│   └── psqr_rstreng_uq.py
└── examples/
    └── make_synthetic_example.py
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick public example

The repository includes a small **synthetic** corrosion-grid generator that is not derived from industry inspection data:

```bash
python examples/make_synthetic_example.py
```

This creates `example1.xlsx` in the repository root using the input format expected by the research script.

## Running the analysis

The research implementation is interactive when executed directly. Before running, review the material properties and uncertainty parameters in the execution block.

```bash
python src/psqr_rstreng_uq.py
```

For long PSQR studies, begin with a small convergence run before launching the full production sample count.

## Related research

This repository is part of the thesis:

> **Seyyed Mohammad Mojtahed Haeri.** *Uncertainty Propagation in Burst-Pressure Assessment of Corroded Pipelines using Engineering and Machine Learning Methods.* M.Sc. thesis, University of Alberta, 2026.

Related conference paper:

> **S. M. M. Haeri**, S. Attia, K. Shahzad, M. Meleka, S. Abtahi, and S. Adeeb, “Integrating ILI Measurement Uncertainty into the RSTRENG Method via Monte Carlo Simulation,” *ASME Pressure Vessels & Piping Conference (PVP 2026)*, Paper PVP2026-183160, Anaheim, California, 2026.

## Data and confidentiality

No raw proprietary industry inspection dataset is distributed through this repository. Public examples are synthetic or based on data that are explicitly redistributable.

## License

A software license has **not yet been assigned**. Until licensing and research/IP permissions are finalized, the repository should not be interpreted as granting reuse rights beyond those provided by applicable law and GitHub's terms.
