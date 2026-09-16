"""Create a small synthetic corrosion-depth grid for demonstration purposes.

Values are corrosion depth as percent of nominal wall thickness. NaN values mark
cells outside the synthetic feature. The output is intentionally synthetic and is
not derived from proprietary inspection data.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    grid = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, 12.0, 18.0, 24.0, 29.0, 25.0, 20.0, 15.0, np.nan, np.nan, np.nan],
            [np.nan, 10.0, 17.0, 26.0, 35.0, 43.0, 39.0, 31.0, 22.0, 14.0, np.nan, np.nan],
            [np.nan, 13.0, 22.0, 34.0, 46.0, 55.0, 51.0, 40.0, 29.0, 18.0, 11.0, np.nan],
            [np.nan, np.nan, 18.0, 28.0, 39.0, 48.0, 44.0, 35.0, 24.0, 16.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 16.0, 25.0, 31.0, 29.0, 23.0, 17.0, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=float,
    )

    output = Path(__file__).resolve().parents[1] / "example1.xlsx"
    pd.DataFrame(grid).to_excel(output, header=False, index=False)
    print(f"Synthetic example written to: {output}")


if __name__ == "__main__":
    main()
