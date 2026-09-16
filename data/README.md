# Data

The research implementation expects a headerless Excel matrix containing a two-dimensional corrosion-depth grid.

## Expected representation

- Each finite cell contains corrosion depth as **percent of nominal wall thickness**.
- `NaN` cells indicate locations outside the reported feature or unavailable cells.
- Axial and circumferential cell dimensions are configured separately in the analysis script.

## Public-data policy

The restricted industry ILI grids used in the associated graduate research are **not distributed here**. Do not add proprietary inspection files, confidential measurements, or derived datasets that would expose restricted information.

The repository includes a synthetic demonstration-grid generator under `examples/`. Any additional public example should be synthetic or explicitly redistributable and should document its provenance.
