"""
RSTRENG and PSQR uncertainty-propagation workflows for corroded-pipeline
failure-pressure assessment.

Current PSQR implementation highlights:
1. Published strip-based Axial Alignment Factor (AAF): whole-anomaly RSTRENG is
   compared with circumferential 6t x L strip assessments, and the minimum
   strip/whole pressure ratio defines AAF.
2. Candidate interaction window in physical units of +/-max(6t, 25.4 mm),
   corresponding to 12t total with a minimum total width of 2 in.
3. Checkpoint signatures include an explicit algorithm version and source-code
   SHA-256 hash so incompatible prior results cannot be silently reused.
4. Per-realization records include AAF, governing-strip information,
   whole-anomaly RSTRENG pressure, selected assessment method, and fallback
   status.
5. The PSQR path focuses on P5 and effective corrosion dimensions rather than
   probability-of-failure reporting.
6. Feature-level ILI length/width uncertainty, depth-weighted path initiation,
   and axial ordering of generated plausible profiles are retained explicitly.

The module also includes the RSTRENG Monte Carlo workflow used in the associated
research.
"""

import numpy as np
import pandas as pd
import math
import os
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import matplotlib
# Use a non-interactive backend so production runs can save figures without
# requiring a graphical display.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, beta, truncnorm, gaussian_kde
import time
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D

# ============================================================================
# PLOT STYLE CONFIGURATION FOR JOURNAL PAPERS
# ============================================================================
# The palette below is color-blind friendly and remains distinguishable in print.
# All publication figures are saved as vector PDF plus 600-dpi PNG.
JOURNAL_DOUBLE_WIDTH = 7.20   # inches, typical double-column width
JOURNAL_SINGLE_WIDTH = 3.50   # inches, typical single-column width

# Production-run defaults. Edit only if you intentionally want different behavior.
DEFAULT_CHECKPOINT_EVERY = 100
DEFAULT_MAX_METHOD1_IN_MEMORY = 250_000
DEFAULT_N_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 1))
DEFAULT_OUTPUT_ROOT = 'PSQR_OUTPUT'
ALGORITHM_VERSION = 'PSQR_UQ_v3_full_AAF_correct_window_no_POF_2026-09-10'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9.0,
    'axes.labelsize': 9.5,
    'axes.titlesize': 9.5,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'legend.fontsize': 8.0,
    'legend.frameon': False,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4.0,
    'ytick.major.size': 4.0,
    'xtick.minor.size': 2.5,
    'ytick.minor.size': 2.5,
    'figure.dpi': 160,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'savefig.facecolor': 'white',
    'lines.linewidth': 1.6,
    'lines.markersize': 4.5,
})

COLORS = {
    'primary': '#0072B2',      # blue
    'secondary': '#009E73',    # green
    'tertiary': '#56B4E9',     # light blue
    'safe': '#009E73',
    'failure': '#D55E00',      # vermillion
    'nominal': '#CC79A7',      # purple
    'map': '#E69F00',          # orange
    'text': '#222222',
    'grid': '#D9D9D9',
    'hist_fill': '#DDEBF7',
    'type1': '#0072B2',
    'type2': '#E69F00',
    'gray': '#666666',
}


def style_axis(ax, grid_axis='both'):
    """Apply one consistent, restrained journal style to an axis."""
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=False, right=False)
    if grid_axis in ('both', 'x', 'y'):
        ax.grid(True, which='major', axis=grid_axis,
                color=COLORS['grid'], linewidth=0.55, alpha=0.70)
    else:
        ax.grid(False)
    ax.grid(False, which='minor')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def save_journal_figure(fig, basename, close=False, output_dir=None):
    """Save a figure as vector PDF and 600-dpi PNG.

    For PSQR production runs, ``output_dir`` keeps figures beside checkpoints,
    logs, and numerical outputs. RSTRENG calls that omit it retain the original
    behavior and write to the current directory.
    """
    out = Path(output_dir) if output_dir is not None else Path('.')
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{basename}.pdf', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(out / f'{basename}.png', dpi=600, bbox_inches='tight', facecolor='white')
    if close:
        plt.close(fig)


# ============================================================================
# COMMON FUNCTIONS
# ============================================================================

def bulging_factor(L, D, t):
    L2Dt = (L**2) / (D * t)
    if L2Dt <= 50:
        return math.sqrt(1 + 0.6275*L2Dt - 0.003375*L2Dt**2)
    else:
        return 3.3 + 0.032*L2Dt


def compute_RSF_with_defect_dims(depths_mm, lengths_mm, diameter, thickness):
    """
    Enhanced RSF computation that returns:
    RSF, critical average depth, critical length, bulging factor type.
    """
    min_rsf = float('inf')
    critical_d_avg = 0
    critical_L_comb = 0
    critical_bulging_type = 0

    for p in range(len(depths_mm)):
        for q in range(p, len(depths_mm)):
            L_comb = np.sum(lengths_mm[p:q+1])
            A_comb = np.sum(depths_mm[p:q+1] * lengths_mm[p:q+1])

            if L_comb == 0:
                continue

            d_avg = A_comb / L_comb

            if d_avg >= thickness:
                continue

            L2Dt = (L_comb**2) / (diameter * thickness)

            if L2Dt <= 50:
                M = math.sqrt(1 + 0.6275*L2Dt - 0.003375*L2Dt**2)
                bulging_type = 1
            else:
                M = 3.3 + 0.032*L2Dt
                bulging_type = 2

            denom = 1 - (d_avg / (M * thickness))

            if denom <= 0:
                continue

            rsf = (1 - d_avg/thickness) / denom

            if rsf < min_rsf and rsf > 0:
                min_rsf = rsf
                critical_d_avg = d_avg
                critical_L_comb = L_comb
                critical_bulging_type = bulging_type

    if min_rsf == float('inf'):
        return 0.001, 0, 0, 0
    else:
        return min_rsf, critical_d_avg, critical_L_comb, critical_bulging_type


def burst_pressure(t, D, sy, RSF):
    if RSF <= 0:
        return 0.001
    return (2 * t * (sy + 68.9476) / D) * RSF


def sample_truncated_normal(mu, sigma, lower, upper):
    """Sample from truncated normal distribution."""
    if sigma <= 0:
        return mu
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)


# ============================================================================
# RSTRENG MONTE CARLO CLASS
# ============================================================================

class RSTRENGMonteCarlo:
    """
    Comprehensive Monte Carlo Analysis for RSTRENG with defect dimension tracking
    and convergence analysis using industry-standard absolute error uncertainty.
    """

    def __init__(self, excel_path, material_params, operating_pressure,
                 depth_absolute_error_pct_t=0.07,
                 length_absolute_error_mm=7.0,
                 confidence_level=0.80,
                 nominal_cell_length=5):

        self.excel_path = excel_path
        self.material_params = material_params
        self.operating_pressure = operating_pressure
        self.depth_absolute_error_pct_t = depth_absolute_error_pct_t
        self.length_absolute_error_mm = length_absolute_error_mm
        self.confidence_level = confidence_level
        self.nominal_cell_length = nominal_cell_length

        self.z_score = norm.ppf((1 + confidence_level) / 2)
        t = material_params['thickness']
        self.depth_std_mm = (depth_absolute_error_pct_t * t) / self.z_score
        self.length_std_mm = length_absolute_error_mm / self.z_score
        self.depth_absolute_error_mm = depth_absolute_error_pct_t * t

        self.results = {}
        self.convergence_results = {}
        self.traditional_results = {}

    def generate_river_bottom_profile_from_df(self, df, cell_lengths_grid):
        """Generate RSTRENG profile from DataFrame with variable cell lengths."""
        depth_percentages = []
        row_indices = []
        cell_lengths = []

        for col in range(df.shape[1]):
            col_data = df.iloc[:, col]
            valid_depths = col_data.dropna()

            if not valid_depths.empty:
                max_idx_pos = valid_depths.astype(float).values.argmax()
                max_idx = valid_depths.index[max_idx_pos]
                max_pct = min(valid_depths.loc[max_idx], 100)

                depth_percentages.append(max_pct)
                row_indices.append(max_idx)
                cell_lengths.append(cell_lengths_grid.iloc[max_idx, col])

        return np.array(cell_lengths), np.array(depth_percentages), row_indices

    def bulging_factor(self, L, D, t):
        return bulging_factor(L, D, t)

    def compute_RSF_with_defect_dims(self, depths_mm, lengths_mm, diameter, thickness):
        return compute_RSF_with_defect_dims(depths_mm, lengths_mm, diameter, thickness)

    def burst_pressure(self, t, D, sy, RSF):
        return burst_pressure(t, D, sy, RSF)

    def sample_truncated_normal(self, mu, sigma, lower, upper):
        return sample_truncated_normal(mu, sigma, lower, upper)

    def compute_nominal_rstreng(self):
        """Compute RSTRENG failure pressure without uncertainties."""
        df_original = pd.read_excel(self.excel_path, header=None)

        t = self.material_params['thickness']
        D = self.material_params['diameter']
        sy = self.material_params['yield_strength']

        df_cell_lengths = pd.DataFrame(
            np.full(df_original.shape, self.nominal_cell_length),
            index=df_original.index,
            columns=df_original.columns
        )

        lengths, depths_pct, _ = self.generate_river_bottom_profile_from_df(
            df_original,
            df_cell_lengths
        )

        depths_mm = depths_pct * t / 100

        rsf, d_avg, L_comb, bulging_type = self.compute_RSF_with_defect_dims(
            depths_mm,
            lengths,
            D,
            t
        )

        p_burst = self.burst_pressure(t, D, sy, rsf)

        return p_burst, d_avg, L_comb, rsf

    def compute_std_traditional_method(self):
        """Traditional error propagation."""
        print("\nComputing traditional method (error propagation) with industry-standard absolute errors...")

        nominal_pressure, nominal_d_avg, nominal_L_comb, nominal_rsf = self.compute_nominal_rstreng()

        t = self.material_params['thickness']
        D = self.material_params['diameter']
        sy = self.material_params['yield_strength']

        delta = 1e-6

        def compute_pressure_for_defect(d_avg, L_comb):
            M = self.bulging_factor(L_comb, D, t)

            if M <= 0 or (1 - d_avg/(M * t)) <= 0:
                return 0.001

            rsf = (1 - d_avg/t) / (1 - d_avg/(M * t))

            return self.burst_pressure(t, D, sy, rsf)

        P0 = compute_pressure_for_defect(nominal_d_avg, nominal_L_comb)

        P_d_avg_plus = compute_pressure_for_defect(
            nominal_d_avg + delta,
            nominal_L_comb
        )
        dP_dd_avg = (P_d_avg_plus - P0) / delta

        P_L_comb_plus = compute_pressure_for_defect(
            nominal_d_avg,
            nominal_L_comb + delta
        )
        dP_dL_comb = (P_L_comb_plus - P0) / delta

        std_d_avg = self.depth_std_mm
        std_L_comb = self.length_std_mm

        std_pressure_traditional = math.sqrt(
            (dP_dd_avg ** 2) * (std_d_avg ** 2) +
            (dP_dL_comb ** 2) * (std_L_comb ** 2)
        )

        num_samples = 10000

        pressure_samples = []
        d_avg_samples = []
        L_comb_samples = []

        for i in range(num_samples):
            d_avg_sample = np.random.normal(nominal_d_avg, std_d_avg)
            L_comb_sample = np.random.normal(nominal_L_comb, std_L_comb)

            d_avg_sample = max(0.001, min(d_avg_sample, t * 0.95))
            L_comb_sample = max(1.0, L_comb_sample)

            pressure = compute_pressure_for_defect(d_avg_sample, L_comb_sample)

            pressure_samples.append(pressure)
            d_avg_samples.append(d_avg_sample)
            L_comb_samples.append(L_comb_sample)

        std_pressure_mc_traditional = np.std(pressure_samples)
        mean_pressure_traditional = np.mean(pressure_samples)

        traditional_results = {
            'nominal_pressure': nominal_pressure,
            'nominal_d_avg': nominal_d_avg,
            'nominal_L_comb': nominal_L_comb,
            'std_d_avg': std_d_avg,
            'std_L_comb': std_L_comb,
            'std_pressure_error_prop': std_pressure_traditional,
            'std_pressure_mc_traditional': std_pressure_mc_traditional,
            'mean_pressure_traditional': mean_pressure_traditional,
            'pressure_samples_traditional': pressure_samples,
            'd_avg_samples_traditional': d_avg_samples,
            'L_comb_samples_traditional': L_comb_samples,
            'dP_dd_avg': dP_dd_avg,
            'dP_dL_comb': dP_dL_comb,
            'method': 'Industry-Standard Absolute Error'
        }

        print(f"Traditional Method Results (Industry-Standard):")
        print(f"  Nominal burst pressure: {nominal_pressure:.2f} MPa")
        print(f"  Nominal critical depth: {nominal_d_avg:.2f} mm")
        print(f"  Nominal critical length: {nominal_L_comb:.1f} mm")
        print(f"  Std of depth (absolute error): {std_d_avg:.4f} mm")
        print(f"  Std of length (absolute error): {std_L_comb:.3f} mm")
        print(f"  Partial derivative ∂P/∂d_avg: {dP_dd_avg:.4f}")
        print(f"  Partial derivative ∂P/∂L_comb: {dP_dL_comb:.4f}")
        print(f"  Std of pressure (error propagation): {std_pressure_traditional:.2f} MPa")
        print(f"  Std of pressure (MC traditional): {std_pressure_mc_traditional:.2f} MPa")
        print(f"  Mean pressure (traditional MC): {mean_pressure_traditional:.2f} MPa")

        self.traditional_results = traditional_results

        return traditional_results

    def safe_kde_contour(self, x_data, y_data, xx, yy):
        """Safe KDE computation with error handling for singular matrices."""
        try:
            if len(x_data) < 5:
                return None

            data = np.vstack([x_data, y_data])

            if np.std(x_data) < 1e-10 or np.std(y_data) < 1e-10:
                return None

            kde = gaussian_kde(data)
            positions = np.vstack([xx.ravel(), yy.ravel()])
            zz = np.reshape(kde(positions).T, xx.shape)

            return zz

        except (np.linalg.LinAlgError, ValueError):
            return None

    def run_monte_carlo(self, num_samples=10000, generate_plots=False, fit_distributions=True):
        """Enhanced Monte Carlo Simulation with defect dimension tracking."""
        if generate_plots:
            print(f"\n{'='*60}")
            print(f"RUNNING MONTE CARLO ANALYSIS: {num_samples} SAMPLES")
            print(f"{'='*60}")
            print(f"Uncertainty Parameters (Industry Standard):")
            print(f"  • Depth: ±{self.depth_absolute_error_pct_t:.3f}t "
                  f"(±{self.depth_absolute_error_mm:.2f} mm) at "
                  f"{self.confidence_level*100:.0f}% certainty")
            print(f"  • Length: ±{self.length_absolute_error_mm:.1f} mm at "
                  f"{self.confidence_level*100:.0f}% certainty")
            print(f"  • Z-score for {self.confidence_level*100:.0f}% confidence: {self.z_score:.3f}")
            print(f"  • Depth std dev: {self.depth_std_mm:.3f} mm")
            print(f"  • Length std dev: {self.length_std_mm:.3f} mm")
        else:
            print(f"Running {num_samples} samples...", end=" ")

        df_original = pd.read_excel(self.excel_path, header=None)

        t = self.material_params['thickness']
        D = self.material_params['diameter']
        sy = self.material_params['yield_strength']

        nominal_pressure, nominal_d_avg, nominal_L_comb, nominal_rsf = self.compute_nominal_rstreng()

        non_nan_mask = ~df_original.isna()
        non_nan_cells = list(zip(*np.where(non_nan_mask)))
        original_depths = [df_original.iloc[row, col] for row, col in non_nan_cells]
        columns_with_data = set(col for row, col in non_nan_cells)

        depth_params = []

        for depth in original_depths:
            if depth == 0:
                depth_params.append({'mu': 0, 'sigma': 0})
            else:
                sigma_pct = (self.depth_std_mm / t) * 100
                depth_params.append({
                    'mu': depth,
                    'sigma': sigma_pct,
                    'lower': 0,
                    'upper': 100
                })

        cell_length_lower = 0.1
        cell_length_upper = 50.0

        k_mean = 0.993
        k_std = k_mean * 0.035
        k_lower = 0.8
        k_upper = 1.1
        k_range = k_upper - k_lower
        k_norm_mean = (k_mean - k_lower) / k_range
        k_norm_var = (k_std / k_range) ** 2
        k_alpha = k_norm_mean * (k_norm_mean * (1 - k_norm_mean) / k_norm_var - 1)
        k_beta = (1 - k_norm_mean) * (k_norm_mean * (1 - k_norm_mean) / k_norm_var - 1)

        burst_pressures = np.zeros(num_samples)
        limit_states = np.zeros(num_samples)
        limit_states_no_zeros = np.zeros(num_samples)

        critical_d_avg_mm = np.zeros(num_samples)
        critical_L_comb_mm = np.zeros(num_samples)
        critical_d_avg_pct = np.zeros(num_samples)
        RSF_values = np.zeros(num_samples)
        bulging_factor_types = np.zeros(num_samples)

        depth_samples = np.zeros(num_samples)
        cell_length_samples = np.zeros(num_samples)
        k_values = np.zeros(num_samples)

        zero_burst_cases = []

        iterator = tqdm(range(num_samples), desc="Processing samples") if generate_plots else range(num_samples)

        for i in iterator:
            k_sample = beta.rvs(k_alpha, k_beta, loc=k_lower, scale=k_range)
            k_values[i] = k_sample
            map_value = k_sample * self.operating_pressure

            df_sampled_depths = df_original.copy().astype(float)

            depth_sum = 0
            depth_count = 0

            for idx, (row, col) in enumerate(non_nan_cells):
                params = depth_params[idx]

                if params['mu'] == 0:
                    df_sampled_depths.iloc[row, col] = 0
                else:
                    sampled_value = self.sample_truncated_normal(
                        params['mu'],
                        params['sigma'],
                        lower=params['lower'],
                        upper=params['upper']
                    )

                    df_sampled_depths.iloc[row, col] = sampled_value
                    depth_sum += sampled_value
                    depth_count += 1

            if depth_count > 0:
                depth_samples[i] = depth_sum / depth_count

            df_cell_lengths = pd.DataFrame(
                np.full(df_original.shape, self.nominal_cell_length, dtype=float),
                index=df_original.index,
                columns=df_original.columns
            )

            cell_length_sum = 0
            cell_length_count = 0

            for col in columns_with_data:
                if self.length_std_mm > 0:
                    cell_length_sample = self.sample_truncated_normal(
                        self.nominal_cell_length,
                        self.length_std_mm,
                        lower=cell_length_lower,
                        upper=cell_length_upper
                    )
                else:
                    cell_length_sample = self.nominal_cell_length

                df_cell_lengths.iloc[:, col] = cell_length_sample
                cell_length_sum += cell_length_sample
                cell_length_count += 1

            if cell_length_count > 0:
                cell_length_samples[i] = cell_length_sum / cell_length_count

            try:
                lengths, depths_pct, _ = self.generate_river_bottom_profile_from_df(
                    df_sampled_depths,
                    df_cell_lengths
                )

                depths_mm = depths_pct * t / 100

                if np.any(depths_mm >= t):
                    p_burst = 0.001
                    rsf = 0.001

                    critical_d_avg_mm[i] = t
                    critical_L_comb_mm[i] = np.sum(lengths)
                    critical_d_avg_pct[i] = 100
                    RSF_values[i] = 0.001
                    bulging_factor_types[i] = 0

                    zero_burst_cases.append({
                        'sample': i,
                        'reason': 'depth_exceeds_thickness'
                    })

                else:
                    rsf, d_avg_mm, L_comb_mm, bulging_type = self.compute_RSF_with_defect_dims(
                        depths_mm,
                        lengths,
                        D,
                        t
                    )

                    p_burst = self.burst_pressure(t, D, sy, rsf)

                    critical_d_avg_mm[i] = d_avg_mm
                    critical_L_comb_mm[i] = L_comb_mm
                    critical_d_avg_pct[i] = (d_avg_mm / t) * 100
                    RSF_values[i] = rsf
                    bulging_factor_types[i] = bulging_type

                    if p_burst <= 0:
                        zero_burst_cases.append({
                            'sample': i,
                            'reason': 'non_positive_burst_pressure'
                        })
                        p_burst = 0.001

                burst_pressures[i] = p_burst

                if p_burst <= map_value:
                    limit_states[i] = 1

                if p_burst > 0.01 and p_burst <= map_value:
                    limit_states_no_zeros[i] = 1

            except Exception as e:
                burst_pressures[i] = np.nan
                limit_states[i] = np.nan
                limit_states_no_zeros[i] = np.nan
                critical_d_avg_mm[i] = np.nan
                critical_L_comb_mm[i] = np.nan
                critical_d_avg_pct[i] = np.nan
                RSF_values[i] = np.nan
                bulging_factor_types[i] = np.nan

        valid_mask = ~np.isnan(burst_pressures)

        pof_with_zeros = np.mean(limit_states[valid_mask])

        no_zeros_mask = valid_mask & (burst_pressures > 0.01)

        if np.sum(no_zeros_mask) > 0:
            pof_without_zeros = np.mean(limit_states_no_zeros[no_zeros_mask])
        else:
            pof_without_zeros = 0.0

        valid_pressures = burst_pressures[valid_mask]
        valid_k_values = k_values[valid_mask]

        valid_d_avg_mm = critical_d_avg_mm[valid_mask]
        valid_L_comb_mm = critical_L_comb_mm[valid_mask]
        valid_d_avg_pct = critical_d_avg_pct[valid_mask]
        valid_RSF = RSF_values[valid_mask]
        valid_bulging_types = bulging_factor_types[valid_mask]

        if len(valid_pressures) > 0:
            mean_burst = np.mean(valid_pressures)
            std_burst = np.std(valid_pressures)
            min_burst = np.min(valid_pressures)
            max_burst = np.max(valid_pressures)
            mean_k = np.mean(valid_k_values)

            mean_d_avg_mm = np.mean(valid_d_avg_mm)
            std_d_avg_mm = np.std(valid_d_avg_mm)
            mean_L_comb_mm = np.mean(valid_L_comb_mm)
            std_L_comb_mm = np.std(valid_L_comb_mm)
            mean_d_avg_pct = np.mean(valid_d_avg_pct)
            mean_RSF = np.mean(valid_RSF)

            count_type1 = np.sum(valid_bulging_types == 1)
            count_type2 = np.sum(valid_bulging_types == 2)
            pct_type1 = count_type1 / len(valid_bulging_types) * 100
            pct_type2 = count_type2 / len(valid_bulging_types) * 100

        else:
            mean_burst = std_burst = min_burst = max_burst = mean_k = np.nan
            mean_d_avg_mm = std_d_avg_mm = mean_L_comb_mm = std_L_comb_mm = np.nan
            mean_d_avg_pct = mean_RSF = np.nan
            count_type1 = count_type2 = pct_type1 = pct_type2 = 0

        self.results = {
            'POF_with_zeros': pof_with_zeros,
            'POF_without_zeros': pof_without_zeros,
            'nominal_pressure': nominal_pressure,
            'nominal_d_avg': nominal_d_avg,
            'nominal_L_comb': nominal_L_comb,
            'nominal_rsf': nominal_rsf,
            'mean_burst': mean_burst,
            'std_burst': std_burst,
            'mean_d_avg_mm': mean_d_avg_mm,
            'std_d_avg_mm': std_d_avg_mm,
            'mean_L_comb_mm': mean_L_comb_mm,
            'std_L_comb_mm': std_L_comb_mm,
            'mean_RSF': mean_RSF,
            'critical_d_avg_mm': critical_d_avg_mm,
            'critical_L_comb_mm': critical_L_comb_mm,
            'critical_d_avg_pct': critical_d_avg_pct,
            'RSF_values': RSF_values,
            'bulging_factor_types': bulging_factor_types,
            'bulging_type_stats': {
                'count_type1': count_type1,
                'count_type2': count_type2,
                'pct_type1': pct_type1,
                'pct_type2': pct_type2
            },
            'burst_pressures': burst_pressures,
            'depth_samples': depth_samples,
            'cell_length_samples': cell_length_samples,
            'k_values': k_values,
            'valid_mask': valid_mask,
            'zero_burst_cases': zero_burst_cases,
            'samples_used': num_samples
        }

        if fit_distributions:
            self.fit_dimension_distributions()

        if generate_plots:
            self.create_comprehensive_plots()

        return self.results

    def fit_dimension_distributions(self):
        """
        Fit probability distributions to RSTRENG outputs:
            - burst pressure
            - critical average depth
            - critical combination length

        Store results in:
            self.results['distribution_fits']
        """
        if not self.results:
            print("No results available. Run Monte Carlo first.")
            return None

        res = self.results
        valid_mask = res['valid_mask']

        fit_results = {
            'rstreng_burst_pressure': fit_best_distribution(
                res['burst_pressures'][valid_mask],
                label='RSTRENG - Burst Pressure',
                validation_sample_size=np.sum(valid_mask),
                random_state=201
            ),
            'rstreng_depth': fit_best_distribution(
                res['critical_d_avg_mm'][valid_mask],
                label='RSTRENG - Critical Effective Depth',
                validation_sample_size=np.sum(valid_mask),
                random_state=202
            ),
            'rstreng_length': fit_best_distribution(
                res['critical_L_comb_mm'][valid_mask],
                label='RSTRENG - Critical Effective Length',
                validation_sample_size=np.sum(valid_mask),
                random_state=203
            )
        }

        self.results['distribution_fits'] = fit_results
        return fit_results

    def create_distribution_fit_plots(self):
        """
        Create distribution fitting and validation plots for RSTRENG results.
        """
        if not self.results:
            print("No results available. Run Monte Carlo first.")
            return

        res = self.results
        valid_mask = res['valid_mask']

        pressures = res['burst_pressures'][valid_mask]
        d_avg = res['critical_d_avg_mm'][valid_mask]
        L_comb = res['critical_L_comb_mm'][valid_mask]

        fits = res.get('distribution_fits', None)
        if fits is None:
            fits = self.fit_dimension_distributions()

        if fits is None:
            return

        # Figure: fitted distribution overlays
        fig_fit, axes_fit = plt.subplots(1, 3, figsize=(14, 4.5))
        fig_fit.suptitle('RSTRENG Automatic Best-Fit Distributions', fontsize=14, fontweight='bold')

        plot_distribution_fit_on_axis(
            axes_fit[0],
            pressures,
            fits['rstreng_burst_pressure'],
            '(a) Burst Pressure: Empirical + Best Fit',
            'Burst Pressure (MPa)',
            hist_color=COLORS['primary'],
            fit_color=COLORS['failure']
        )

        plot_distribution_fit_on_axis(
            axes_fit[1],
            d_avg,
            fits['rstreng_depth'],
            '(b) Critical Depth: Empirical + Best Fit',
            'Critical Average Depth (mm)',
            hist_color=COLORS['primary'],
            fit_color=COLORS['failure']
        )

        plot_distribution_fit_on_axis(
            axes_fit[2],
            L_comb,
            fits['rstreng_length'],
            '(c) Critical Length: Empirical + Best Fit',
            'Critical Combination Length (mm)',
            hist_color=COLORS['secondary'],
            fit_color=COLORS['failure']
        )

        plt.tight_layout()
        plt.savefig('rstreng_best_fit_distributions.pdf', dpi=600, bbox_inches='tight')
        plt.savefig('rstreng_best_fit_distributions.png', dpi=600, bbox_inches='tight')
        plt.show()

        # Figure: validation by sampling from fitted distributions
        fig_val, axes_val = plt.subplots(1, 3, figsize=(14, 4.5))
        fig_val.suptitle('RSTRENG Validation: Original MC Data vs Samples from Best-Fit Distribution',
                         fontsize=14, fontweight='bold')

        plot_distribution_validation_on_axis(
            axes_val[0],
            pressures,
            fits['rstreng_burst_pressure'],
            '(a) Burst Pressure Validation',
            'Burst Pressure (MPa)'
        )

        plot_distribution_validation_on_axis(
            axes_val[1],
            d_avg,
            fits['rstreng_depth'],
            '(b) Critical Depth Validation',
            'Critical Average Depth (mm)'
        )

        plot_distribution_validation_on_axis(
            axes_val[2],
            L_comb,
            fits['rstreng_length'],
            '(c) Critical Length Validation',
            'Critical Combination Length (mm)'
        )

        plt.tight_layout()
        plt.savefig('rstreng_distribution_validation.pdf', dpi=600, bbox_inches='tight')
        plt.savefig('rstreng_distribution_validation.png', dpi=600, bbox_inches='tight')
        plt.show()

        print_distribution_fit_summary(fits)

    def create_comprehensive_plots(self):
        """Create professional journal-quality visualization plots."""
        if not self.results:
            print("No results available. Run Monte Carlo simulation first.")
            return

        print("\nGenerating professional visualizations for publication...")

        results = self.results
        valid_mask = results['valid_mask']

        plot_pressures = results['burst_pressures'][valid_mask]
        plot_k_values = results['k_values'][valid_mask]
        plot_d_avg_mm = results['critical_d_avg_mm'][valid_mask]
        plot_L_comb_mm = results['critical_L_comb_mm'][valid_mask]
        plot_RSF = results['RSF_values'][valid_mask]
        plot_bulging_types = results['bulging_factor_types'][valid_mask]

        safe = plot_pressures > (plot_k_values * self.operating_pressure)
        fail = plot_pressures <= (plot_k_values * self.operating_pressure)

        fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
        fig1.suptitle('Monte Carlo Analysis Results', fontsize=14, fontweight='bold', y=0.98)

        ax = axes1[0, 0]
        ax.hist(plot_pressures, bins=40, alpha=0.8,
                color=COLORS['primary'], density=True,
                edgecolor='white', linewidth=0.5)

        mean_map = np.mean(plot_k_values) * self.operating_pressure

        ax.axvline(mean_map, color=COLORS['map'], linestyle='--', linewidth=2,
                   label=f'Mean MAP: {mean_map:.1f} MPa')
        ax.axvline(results['nominal_pressure'], color=COLORS['nominal'], linestyle='-', linewidth=2,
                   label=f'Nominal: {results["nominal_pressure"]:.1f} MPa')
        ax.set_xlabel('Burst Pressure (MPa)', fontweight='bold')
        ax.set_ylabel('Probability Density', fontweight='bold')
        ax.set_title('(a) Burst Pressure Distribution')
        ax.legend()

        ax = axes1[0, 1]
        ax.scatter(plot_d_avg_mm[safe], plot_L_comb_mm[safe],
                   c=COLORS['safe'], alpha=0.6, s=8,
                   label=f'Safe (n={np.sum(safe):,})')
        ax.scatter(plot_d_avg_mm[fail], plot_L_comb_mm[fail],
                   c=COLORS['failure'], alpha=0.8, s=12,
                   label=f'Failure (n={np.sum(fail):,})')
        ax.scatter(results['nominal_d_avg'], results['nominal_L_comb'],
                   c=COLORS['nominal'], marker='*', s=150,
                   label='Nominal', edgecolors='black', linewidth=1)
        ax.set_xlabel('Critical Average Depth (mm)', fontweight='bold')
        ax.set_ylabel('Critical Combination Length (mm)', fontweight='bold')
        ax.set_title('(b) Critical Defect Dimensions')
        ax.legend()

        ax = axes1[1, 0]
        ax.scatter(plot_d_avg_mm, plot_pressures,
                   c=COLORS['primary'], alpha=0.5, s=8)
        ax.axhline(results['nominal_pressure'], color=COLORS['nominal'],
                   linestyle='-', linewidth=2)
        ax.axhline(mean_map, color=COLORS['map'], linestyle='--', linewidth=2)
        ax.set_xlabel('Critical Average Depth (mm)', fontweight='bold')
        ax.set_ylabel('Burst Pressure (MPa)', fontweight='bold')
        ax.set_title('(c) Depth vs. Burst Pressure')

        ax = axes1[1, 1]
        sorted_pressures = np.sort(plot_pressures)
        cdf = np.arange(1, len(sorted_pressures)+1) / len(sorted_pressures)
        ax.plot(sorted_pressures, cdf, color=COLORS['primary'], linewidth=2)
        ax.axvline(mean_map, color=COLORS['map'], linestyle='--', linewidth=2)
        ax.axvline(results['nominal_pressure'], color=COLORS['nominal'], linestyle='-', linewidth=2)
        ax.axhline(results['POF_with_zeros'], color=COLORS['failure'], linestyle=':', linewidth=2,
                   label=f'POF = {results["POF_with_zeros"]:.4f}')
        ax.set_xlabel('Burst Pressure (MPa)', fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontweight='bold')
        ax.set_title('(d) Cumulative Distribution')
        ax.legend()

        plt.tight_layout()
        plt.savefig('figure1_main_results.pdf', dpi=600, bbox_inches='tight')
        plt.savefig('figure1_main_results.png', dpi=600, bbox_inches='tight')
        plt.show()

        # New RSTRENG distribution fitting and validation plots
        self.create_distribution_fit_plots()

    def run_convergence_analysis(self, sample_sizes=None):
        """Run convergence analysis across different sample sizes."""
        if sample_sizes is None:
            sample_sizes = [100, 500, 1000, 2000, 5000, 10000]

        print(f"\n{'='*60}")
        print("RUNNING CONVERGENCE ANALYSIS")
        print(f"{'='*60}")

        convergence_data = {
            'sample_sizes': sample_sizes,
            'POF_with_zeros': [],
            'POF_without_zeros': [],
            'mean_burst': [],
            'std_burst': [],
            'mean_d_avg_mm': [],
            'mean_L_comb_mm': [],
            'computation_time': [],
            'reliability_index': []
        }

        print(f"\n{'Samples':>10} | {'POF (zeros)':<12} | {'Mean Pressure':<12} | "
              f"{'Std Pressure':<12} | {'Time (s)':<8}")
        print("-" * 80)

        for i, n in enumerate(sample_sizes):
            start_time = time.time()

            if i == len(sample_sizes) - 1:
                results = self.run_monte_carlo(num_samples=n, generate_plots=True, fit_distributions=True)
            else:
                results = self.run_monte_carlo(num_samples=n, generate_plots=False, fit_distributions=False)

            elapsed_time = time.time() - start_time

            if results['POF_with_zeros'] > 0 and results['POF_with_zeros'] < 1:
                rel_index = -norm.ppf(results['POF_with_zeros'])
            else:
                rel_index = np.nan

            convergence_data['POF_with_zeros'].append(results['POF_with_zeros'])
            convergence_data['POF_without_zeros'].append(results['POF_without_zeros'])
            convergence_data['mean_burst'].append(results['mean_burst'])
            convergence_data['std_burst'].append(results['std_burst'])
            convergence_data['mean_d_avg_mm'].append(results['mean_d_avg_mm'])
            convergence_data['mean_L_comb_mm'].append(results['mean_L_comb_mm'])
            convergence_data['computation_time'].append(elapsed_time)
            convergence_data['reliability_index'].append(rel_index)

            pof_zeros_str = (
                f"{results['POF_with_zeros']:.2e}"
                if results['POF_with_zeros'] < 0.001
                else f"{results['POF_with_zeros']:.4f}"
            )

            print(f"{n:>10} | {pof_zeros_str:<12} | "
                  f"{results['mean_burst']:<12.1f} | "
                  f"{results['std_burst']:<12.1f} | {elapsed_time:<8.1f}")

        self.convergence_results = convergence_data

        self.print_detailed_results()

        return convergence_data

    def print_detailed_results(self):
        """Print comprehensive results summary."""
        if not self.results:
            print("No results available. Run Monte Carlo simulation first.")
            return

        results = self.results

        print(f"\n{'='*80}")
        print("COMPREHENSIVE RSTRENG MONTE CARLO ANALYSIS RESULTS")
        print(f"{'='*80}")

        print(f"\nUNCERTAINTY PARAMETERS:")
        print(f"  • Depth: ±{self.depth_absolute_error_pct_t:.3f}t "
              f"(±{self.depth_absolute_error_mm:.2f} mm)")
        print(f"  • Length: ±{self.length_absolute_error_mm:.1f} mm")
        print(f"  • Confidence Level: {self.confidence_level*100:.0f}%")
        print(f"  • Z-score: {self.z_score:.3f}")

        print(f"\nNOMINAL RSTRENG:")
        print(f"  • Burst Pressure: {results['nominal_pressure']:.2f} MPa")
        print(f"  • Critical Depth: {results['nominal_d_avg']:.2f} mm")
        print(f"  • Critical Length: {results['nominal_L_comb']:.1f} mm")
        print(f"  • RSF: {results['nominal_rsf']:.3f}")

        print(f"\nMONTE CARLO RESULTS ({results['samples_used']:,} samples):")
        print(f"  • POF including zero cases: {results['POF_with_zeros']:.6f}")
        print(f"  • POF excluding zero cases: {results['POF_without_zeros']:.6f}")

        if results['POF_with_zeros'] > 0 and results['POF_with_zeros'] < 1:
            rel_index = -norm.ppf(results['POF_with_zeros'])
            print(f"  • Reliability Index: {rel_index:.3f}")

        print(f"\nSTATISTICS:")
        print(f"  • Burst Pressure: {results['mean_burst']:.2f} ± {results['std_burst']:.2f} MPa")
        print(f"  • Critical Depth: {results['mean_d_avg_mm']:.2f} ± {results['std_d_avg_mm']:.2f} mm")
        print(f"  • Critical Length: {results['mean_L_comb_mm']:.1f} ± {results['std_L_comb_mm']:.1f} mm")

        fits = results.get('distribution_fits', None)
        if fits is not None:
            print_distribution_fit_summary(fits)



# ============================================================================
# DISTRIBUTION FITTING FUNCTIONS
# ============================================================================

import scipy.stats as stats


def clean_distribution_data(data):
    """
    Clean data before distribution fitting.
    Removes NaN and inf values.
    """
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    return data


def histogram_overlap_similarity(original, simulated, bins=50):
    """
    Histogram-overlap similarity between original data and fitted-distribution samples.

    Returns:
        similarity percentage between 0 and 100.
    """
    original = clean_distribution_data(original)
    simulated = clean_distribution_data(simulated)

    if len(original) < 5 or len(simulated) < 5:
        return np.nan

    data_min = min(np.min(original), np.min(simulated))
    data_max = max(np.max(original), np.max(simulated))

    if data_max <= data_min:
        return 100.0

    hist_original, bin_edges = np.histogram(
        original,
        bins=bins,
        range=(data_min, data_max),
        density=True
    )

    hist_simulated, _ = np.histogram(
        simulated,
        bins=bin_edges,
        density=True
    )

    bin_widths = np.diff(bin_edges)
    overlap = np.sum(np.minimum(hist_original, hist_simulated) * bin_widths)

    return max(0.0, min(100.0, overlap * 100.0))


def fit_single_distribution(data, dist_name):
    """
    Fit one candidate distribution and calculate AIC, BIC, KS statistic, and parameters.

    Candidate distributions:
        normal
        lognormal
        weibull
        gamma
        exponential
        logistic
        gumbel
    """
    data = clean_distribution_data(data)

    if len(data) < 5:
        return None

    distribution_map = {
        'normal': stats.norm,
        'lognormal': stats.lognorm,
        'weibull': stats.weibull_min,
        'gamma': stats.gamma,
        'exponential': stats.expon,
        'logistic': stats.logistic,
        'gumbel': stats.gumbel_r
    }

    dist = distribution_map[dist_name]

    try:
        # Positive-only distributions are fitted with loc fixed at zero
        # when all data are strictly positive.
        if dist_name in ['lognormal', 'weibull', 'gamma', 'exponential']:
            if np.min(data) <= 0:
                return None

            params = dist.fit(data, floc=0)
        else:
            params = dist.fit(data)

        pdf_values = dist.pdf(data, *params)
        pdf_values = np.maximum(pdf_values, 1e-300)

        log_likelihood = np.sum(np.log(pdf_values))

        k = len(params)
        n = len(data)

        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        ks_stat, ks_pvalue = stats.kstest(data, dist.cdf, args=params)

        return {
            'distribution': dist_name,
            'scipy_distribution': dist,
            'params': params,
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'n': n
        }

    except Exception:
        return None


def fit_best_distribution(data, label='data',
                          candidate_distributions=None,
                          validation_sample_size=None,
                          random_state=123):
    """
    Automatically fit several candidate distributions and select the best one by AIC.

    Validation is done by sampling from the best fitted distribution and comparing
    the sampled distribution with the original MC distribution.

    Validation metrics:
        KS two-sample statistic
        KS similarity percentage = 100 * (1 - KS statistic)
        Histogram-overlap similarity percentage
        Wasserstein distance
        Normalized Wasserstein similarity percentage
    """
    data = clean_distribution_data(data)

    if candidate_distributions is None:
        candidate_distributions = [
            'normal',
            'lognormal',
            'weibull',
            'gamma',
            'exponential',
            'logistic',
            'gumbel'
        ]

    if len(data) < 5:
        return {
            'label': label,
            'best_fit': None,
            'all_fits': [],
            'validation': None,
            'message': 'Not enough valid data points for distribution fitting.'
        }

    all_fits = []

    for dist_name in candidate_distributions:
        fit = fit_single_distribution(data, dist_name)

        if fit is not None and np.isfinite(fit['aic']):
            all_fits.append(fit)

    if len(all_fits) == 0:
        return {
            'label': label,
            'best_fit': None,
            'all_fits': [],
            'validation': None,
            'message': 'No distribution could be fitted successfully.'
        }

    all_fits = sorted(all_fits, key=lambda x: x['aic'])
    best_fit = all_fits[0]

    if validation_sample_size is None:
        validation_sample_size = len(data)

    np.random.seed(random_state)

    dist = best_fit['scipy_distribution']
    params = best_fit['params']

    try:
        simulated = dist.rvs(*params, size=validation_sample_size)
        simulated = clean_distribution_data(simulated)

        # If original data are non-negative physical dimensions,
        # remove negative simulated values for validation comparison.
        if np.min(data) >= 0:
            simulated = simulated[simulated >= 0]

        if len(simulated) < 5:
            validation = None
        else:
            if len(simulated) > len(data):
                simulated_for_comparison = np.random.choice(
                    simulated,
                    size=len(data),
                    replace=False
                )
            else:
                simulated_for_comparison = simulated

            ks_2sample_stat, ks_2sample_pvalue = stats.ks_2samp(
                data,
                simulated_for_comparison
            )

            ks_similarity_percent = max(
                0.0,
                min(100.0, (1.0 - ks_2sample_stat) * 100.0)
            )

            hist_overlap_percent = histogram_overlap_similarity(
                data,
                simulated_for_comparison,
                bins=50
            )

            wasserstein = stats.wasserstein_distance(
                data,
                simulated_for_comparison
            )

            data_range = np.max(data) - np.min(data)

            if data_range > 0:
                normalized_wasserstein_percent = max(
                    0.0,
                    min(100.0, (1.0 - wasserstein / data_range) * 100.0)
                )
            else:
                normalized_wasserstein_percent = 100.0

            validation = {
                'simulated_data': simulated_for_comparison,
                'ks_2sample_statistic': ks_2sample_stat,
                'ks_2sample_pvalue': ks_2sample_pvalue,
                'ks_similarity_percent': ks_similarity_percent,
                'histogram_overlap_similarity_percent': hist_overlap_percent,
                'wasserstein_distance': wasserstein,
                'normalized_wasserstein_similarity_percent': normalized_wasserstein_percent
            }

    except Exception:
        validation = None

    return {
        'label': label,
        'best_fit': best_fit,
        'all_fits': all_fits,
        'validation': validation,
        'message': 'Distribution fitting completed successfully.'
    }


def print_distribution_fit_summary(fit_results):
    """
    Print best distribution and validation results.
    """
    print("\n" + "=" * 80)
    print("AUTOMATIC DISTRIBUTION FITTING SUMMARY")
    print("=" * 80)

    for key, result in fit_results.items():
        print(f"\n{result['label']}")
        print("-" * len(result['label']))

        if result['best_fit'] is None:
            print(result['message'])
            continue

        best = result['best_fit']
        val = result['validation']

        print(f"Best distribution: {best['distribution']}")
        print(f"Parameters: {best['params']}")
        print(f"AIC: {best['aic']:.3f}")
        print(f"BIC: {best['bic']:.3f}")
        print(f"One-sample KS statistic: {best['ks_statistic']:.4f}")
        print(f"One-sample KS p-value: {best['ks_pvalue']:.4f}")

        print("\nCandidate ranking by AIC:")
        for rank, fit in enumerate(result['all_fits'], start=1):
            print(
                f"  {rank}. {fit['distribution']:<12s} "
                f"AIC={fit['aic']:.3f}, "
                f"BIC={fit['bic']:.3f}, "
                f"KS={fit['ks_statistic']:.4f}"
            )

        if val is not None:
            print("\nValidation by sampling from fitted distribution:")
            print(f"  KS two-sample statistic: {val['ks_2sample_statistic']:.4f}")
            print(f"  KS two-sample p-value: {val['ks_2sample_pvalue']:.4f}")
            print(f"  KS similarity: {val['ks_similarity_percent']:.2f}%")
            print(f"  Histogram-overlap similarity: {val['histogram_overlap_similarity_percent']:.2f}%")
            print(f"  Wasserstein distance: {val['wasserstein_distance']:.4f}")
            print(f"  Normalized Wasserstein similarity: {val['normalized_wasserstein_similarity_percent']:.2f}%")


def plot_distribution_fit_on_axis(ax, data, fit_result, title, xlabel,
                                  bins=40,
                                  hist_color=None,
                                  fit_color=None,
                                  show_legend=False):
    """Plot empirical density and selected fitted PDF without text collisions.

    The previous version placed a three-line statistics box and a legend in the
    same small panel. For publication figures, the annotation is now compact and
    panel legends are optional; the PSQR multi-panel figure uses one shared
    legend in its otherwise-empty sixth panel.
    """
    data = clean_distribution_data(data)

    if hist_color is None:
        hist_color = COLORS['primary']
    if fit_color is None:
        fit_color = COLORS['failure']

    ax.hist(
        data,
        bins=bins,
        density=True,
        alpha=0.70,
        color=hist_color,
        edgecolor='white',
        linewidth=0.40,
        label='Empirical data'
    )

    if fit_result['best_fit'] is not None and len(data) > 0:
        best = fit_result['best_fit']
        dist = best['scipy_distribution']
        params = best['params']

        x_min = np.min(data)
        x_max = np.max(data)

        if x_max > x_min:
            pad = 0.015 * (x_max - x_min)
            x = np.linspace(x_min - pad, x_max + pad, 600)
            y = dist.pdf(x, *params)
            ax.plot(x, y, color=fit_color, linewidth=1.8,
                    label=f"Best fit: {best['distribution']}")

        val = fit_result['validation']
        if val is not None:
            similarity_text = (
                f"{best['distribution'].capitalize()} | AIC {best['aic']:.1f}\n"
                f"KS sim. {val['ks_similarity_percent']:.1f}% | "
                f"overlap {val['histogram_overlap_similarity_percent']:.1f}%"
            )
        else:
            similarity_text = f"{best['distribution'].capitalize()} | AIC {best['aic']:.1f}"

        # Compact box at lower right avoids the tallest histogram region in most
        # distributions and never competes with a panel legend.
        ax.text(
            0.97, 0.05, similarity_text,
            transform=ax.transAxes,
            va='bottom', ha='right', fontsize=6.8,
            bbox=dict(boxstyle='round,pad=0.24', facecolor='white',
                      alpha=0.94, edgecolor=COLORS['grid'], linewidth=0.55)
        )

    ax.set_title(title, loc='left', pad=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density')
    if show_legend:
        ax.legend(loc='upper right', fontsize=7.0)
    style_axis(ax)


def plot_distribution_validation_on_axis(ax, data, fit_result, title, xlabel,
                                         bins=40, show_legend=False):
    """Compare MC data with samples from the fitted law using compact annotation."""
    data = clean_distribution_data(data)

    # Use common bin edges so overlap is visually meaningful.
    simulated = None
    if fit_result['validation'] is not None:
        simulated = clean_distribution_data(fit_result['validation']['simulated_data'])

    if simulated is not None and len(simulated) > 0 and len(data) > 0:
        lo = min(np.min(data), np.min(simulated))
        hi = max(np.max(data), np.max(simulated))
        bin_edges = np.linspace(lo, hi, bins + 1) if hi > lo else bins
    else:
        bin_edges = bins

    ax.hist(
        data,
        bins=bin_edges,
        density=True,
        alpha=0.58,
        color=COLORS['primary'],
        edgecolor='white',
        linewidth=0.40,
        label='Original MC data'
    )

    if simulated is not None and len(simulated) > 0:
        ax.hist(
            simulated,
            bins=bin_edges,
            density=True,
            alpha=0.44,
            color=COLORS['secondary'],
            edgecolor='white',
            linewidth=0.40,
            label='Fitted-distribution sample'
        )

        val = fit_result['validation']
        validation_text = (
            f"KS sim. {val['ks_similarity_percent']:.1f}% | "
            f"overlap {val['histogram_overlap_similarity_percent']:.1f}%\n"
            f"Wasserstein sim. {val['normalized_wasserstein_similarity_percent']:.1f}%"
        )
        ax.text(
            0.97, 0.05, validation_text,
            transform=ax.transAxes,
            va='bottom', ha='right', fontsize=6.8,
            bbox=dict(boxstyle='round,pad=0.24', facecolor='white',
                      alpha=0.94, edgecolor=COLORS['grid'], linewidth=0.55)
        )

    ax.set_title(title, loc='left', pad=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density')
    if show_legend:
        ax.legend(loc='upper right', fontsize=7.0)
    style_axis(ax)


# ============================================================================
# PSQR-SPECIFIC FUNCTIONS
# ============================================================================

def generate_river_bottom_profile_tiebreak(df, cell_length_mm=5):
    """Generate a conventional deepest-to-deepest river-bottom profile."""
    depth_percentages = []
    row_indices = []
    prev_row = None

    for col in range(df.shape[1]):
        col_data = df.iloc[:, col].dropna()
        if col_data.empty:
            continue

        max_depth = float(col_data.max())
        candidate_rows = col_data[col_data == max_depth].index.tolist()
        if prev_row is None or len(candidate_rows) == 1:
            selected_row = int(min(candidate_rows))
        else:
            selected_row = int(min(candidate_rows, key=lambda r: abs(r - prev_row)))

        depth_percentages.append(min(max(max_depth, 0.0), 100.0))
        row_indices.append(selected_row)
        prev_row = selected_row

    lengths = np.full(len(depth_percentages), float(cell_length_mm))
    return lengths, np.asarray(depth_percentages, dtype=float), row_indices


def rstreng_from_depth_grid(depth_grid, cell_length_mm, wall_thickness_mm,
                            diameter_mm, yield_strength_mpa):
    """Evaluate conventional RSTRENG for a 2-D grid via its river-bottom profile."""
    df = pd.DataFrame(np.asarray(depth_grid, dtype=float))
    lengths, depths_pct, row_indices = generate_river_bottom_profile_tiebreak(
        df, cell_length_mm
    )
    if len(depths_pct) == 0:
        return {
            'pressure_mpa': np.inf,
            'rsf': np.nan,
            'd_avg_mm': np.nan,
            'L_comb_mm': np.nan,
            'row_indices': [],
        }

    depths_mm = depths_pct * float(wall_thickness_mm) / 100.0
    rsf, d_avg, L_comb, _ = compute_RSF_with_defect_dims(
        depths_mm, lengths, diameter_mm, wall_thickness_mm
    )
    pressure = burst_pressure(wall_thickness_mm, diameter_mm, yield_strength_mpa, rsf)
    return {
        'pressure_mpa': float(pressure),
        'rsf': float(rsf),
        'd_avg_mm': float(d_avg),
        'L_comb_mm': float(L_comb),
        'row_indices': list(row_indices),
    }


def calculate_axial_alignment_factor(depth_grid, cell_length_mm, cell_width_mm,
                                      wall_thickness_mm, diameter_mm,
                                      yield_strength_mpa,
                                      equality_tolerance=1e-8):
    """
    Calculate the published strip-based PSQR Axial Alignment Factor (AAF).

    The public PRCI description evaluates 6t-wide x full-length-L strips,
    advancing one circumferential grid space at a time. For each strip the
    RSTRENG pressure is divided by the whole-anomaly RSTRENG pressure; AAF is
    the minimum ratio. AAF approximately equal to 1 retains RSTRENG, whereas
    AAF greater than 1 selects PSQR P5.

    Because the ILI map is cell based, 6t is represented by the nearest whole
    number of circumferential grid rows. Both target and discretized widths are
    returned so the implementation is auditable.
    """
    grid = np.asarray(depth_grid, dtype=float)
    if grid.ndim != 2:
        raise ValueError('depth_grid must be a two-dimensional array.')
    if cell_width_mm <= 0:
        raise ValueError('cell_width_mm must be positive.')

    span = get_active_grid_span(grid)
    whole = rstreng_from_depth_grid(
        grid, cell_length_mm, wall_thickness_mm, diameter_mm, yield_strength_mpa
    )
    whole_pressure = float(whole['pressure_mpa'])
    if not np.isfinite(whole_pressure) or whole_pressure <= 0:
        raise RuntimeError('Whole-anomaly RSTRENG pressure is not finite and positive.')

    target_strip_width_mm = 6.0 * float(wall_thickness_mm)
    rows_per_strip = max(
        1,
        int(np.floor(target_strip_width_mm / float(cell_width_mm) + 0.5))
    )
    actual_strip_width_mm = rows_per_strip * float(cell_width_mm)

    row_min = int(span['row_min'])
    row_max = int(span['row_max'])
    n_rows = grid.shape[0]
    strip_records = []

    for start_row in range(row_min, row_max + 1):
        end_row = min(n_rows - 1, start_row + rows_per_strip - 1)
        if end_row < start_row:
            continue

        strip_grid = np.full_like(grid, np.nan, dtype=float)
        strip_grid[start_row:end_row + 1, :] = grid[start_row:end_row + 1, :]
        strip = rstreng_from_depth_grid(
            strip_grid, cell_length_mm, wall_thickness_mm,
            diameter_mm, yield_strength_mpa
        )
        strip_pressure = float(strip['pressure_mpa'])
        if not np.isfinite(strip_pressure) or strip_pressure <= 0:
            continue

        ratio_raw = strip_pressure / whole_pressure
        if ratio_raw < 1.0 - 1e-7:
            raise RuntimeError(
                'AAF consistency check failed: a 6t strip produced a lower '
                'RSTRENG pressure than the whole anomaly. Check grid/NaN semantics.'
            )
        ratio = max(1.0, float(ratio_raw))
        strip_records.append({
            'start_row': int(start_row),
            'end_row': int(end_row),
            'pressure_mpa': strip_pressure,
            'ratio': ratio,
        })

    if not strip_records:
        raise RuntimeError('No valid 6t strips were available for the AAF calculation.')

    governing = min(strip_records, key=lambda rec: rec['ratio'])
    aaf = float(governing['ratio'])
    use_rstreng = bool(aaf <= 1.0 + float(equality_tolerance))

    return {
        'aaf': aaf,
        'use_rstreng_fallback': use_rstreng,
        'whole_rstreng_pressure_mpa': whole_pressure,
        'whole_rstreng_d_avg_mm': float(whole['d_avg_mm']),
        'whole_rstreng_L_comb_mm': float(whole['L_comb_mm']),
        'governing_strip_pressure_mpa': float(governing['pressure_mpa']),
        'governing_strip_start_row': int(governing['start_row']),
        'governing_strip_end_row': int(governing['end_row']),
        'target_strip_width_mm': float(target_strip_width_mm),
        'actual_strip_width_mm': float(actual_strip_width_mm),
        'rows_per_strip': int(rows_per_strip),
        'n_strips_evaluated': int(len(strip_records)),
    }

def depth_weighted_start_probability_map(depth_grid):
    """
    Return the PSQR starting-cell probability map.

    Each finite, positive-depth cell receives probability proportional to its
    reported corrosion depth. Hence deeper cells are more likely to become a
    starting point, but all positive-depth cells remain possible. If a grid has
    no positive depth (degenerate case), finite cells are sampled uniformly.
    """
    grid = np.asarray(depth_grid, dtype=float)
    finite = np.isfinite(grid)

    if not np.any(finite):
        raise ValueError('The PSQR depth grid contains no finite cells.')

    weights = np.zeros_like(grid, dtype=float)
    weights[finite] = np.clip(grid[finite], 0.0, None)
    total = np.sum(weights)

    if total <= 0:
        weights[finite] = 1.0
        total = np.sum(weights)

    return weights / total


def generate_psqr_profiles_from_grid(depth_grid, wall_thickness_mm, cell_width_mm,
                                     num_profiles=500):
    """
    Generate plausible PSQR paths from a two-dimensional corrosion grid.

    Starts are sampled independently with probability proportional to local
    positive corrosion depth. Path transitions retain the 0.1 proximity / 0.9
    depth weighting. Candidate cells are restricted to the published centered
    interaction window: 12t total with a minimum total width of 2 in.
    """
    depth_grid = np.asarray(depth_grid, dtype=float)
    rows, cols = depth_grid.shape
    if cell_width_mm <= 0:
        raise ValueError('cell_width_mm must be positive.')

    half_window_mm = max(6.0 * float(wall_thickness_mm), 25.4)
    half_span_cells = max(
        0,
        int(np.floor(half_window_mm / float(cell_width_mm) + 1e-12))
    )

    start_probability_map = depth_weighted_start_probability_map(depth_grid)
    start_positions = np.argwhere(start_probability_map > 0)
    start_probs = start_probability_map[start_probability_map > 0]
    start_probs = start_probs / np.sum(start_probs)

    profiles = []
    for _ in range(int(num_profiles)):
        start_idx = np.random.choice(len(start_positions), p=start_probs)
        start_row, start_col = start_positions[start_idx]
        start_row = int(start_row)
        start_col = int(start_col)
        profile = [(start_col, start_row)]

        for direction in (-1, 1):
            current_col, current_row = start_col, start_row
            while True:
                next_col = current_col + direction
                if next_col < 0 or next_col >= cols:
                    break

                candidates = []
                for row in range(rows):
                    distance_cells = abs(row - current_row)
                    distance_mm = distance_cells * float(cell_width_mm)
                    if distance_mm > half_window_mm + 1e-12:
                        continue

                    depth = depth_grid[row, next_col]
                    if not np.isfinite(depth):
                        continue

                    proximity = (half_span_cells - distance_cells) + 1
                    if proximity <= 0:
                        continue
                    candidates.append((row, float(proximity), max(float(depth), 0.0)))

                if not candidates:
                    break

                total_prox = sum(p for _, p, _ in candidates)
                total_depth = sum(d for _, _, d in candidates)
                probs = []
                for row, prox, depth in candidates:
                    if total_prox <= 0 and total_depth <= 0:
                        prob = 1.0 / len(candidates)
                    elif total_depth <= 0:
                        prob = prox / total_prox
                    elif total_prox <= 0:
                        prob = depth / total_depth
                    else:
                        prob = 0.1 * (prox / total_prox) + 0.9 * (depth / total_depth)
                    probs.append(prob)

                probs = np.asarray(probs, dtype=float)
                probs /= probs.sum()
                selected_idx = np.random.choice(len(candidates), p=probs)
                selected_row = int(candidates[selected_idx][0])
                profile.append((next_col, selected_row))
                current_col, current_row = next_col, selected_row

        profile = sorted(set(profile), key=lambda item: item[0])
        profiles.append(profile)

    return profiles

def convert_psqr_profile(profile, depth_grid, wall_thickness, cell_length):
    """Convert a plausible PSQR path into axially ordered depth/length arrays."""
    ordered_profile = sorted(profile, key=lambda item: item[0])
    depths_mm = []

    for col, row in ordered_profile:
        depth_pct = depth_grid[row, col]
        if not np.isfinite(depth_pct):
            continue
        depths_mm.append(min(max(float(depth_pct), 0.0), 100.0) * wall_thickness / 100.0)

    return np.full(len(depths_mm), float(cell_length)), np.asarray(depths_mm, dtype=float)


def get_active_grid_span(depth_grid):
    """
    Determine the rectangular grid span occupied by the reported feature.

    The span is measured from the first to the last row/column containing at
    least one finite ILI cell. Internal gaps are therefore retained in the
    physical extent instead of being collapsed.
    """
    grid = np.asarray(depth_grid, dtype=float)
    valid = np.isfinite(grid)

    if not np.any(valid):
        raise ValueError('Cannot determine feature dimensions from an empty grid.')

    rows, cols = np.where(valid)
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())

    return {
        'row_min': row_min,
        'row_max': row_max,
        'col_min': col_min,
        'col_max': col_max,
        'n_rows_span': row_max - row_min + 1,
        'n_cols_span': col_max - col_min + 1,
    }


def psqr_burst_pressure_from_grid(depth_grid, cell_length_mm, cell_width_mm,
                                  wall_thickness_mm, diameter_mm, yield_strength_mpa,
                                  num_profiles=500):
    """Run PSQR and strip-based AAF diagnostics for one corrosion-grid realization."""
    aaf_info = calculate_axial_alignment_factor(
        depth_grid, cell_length_mm, cell_width_mm,
        wall_thickness_mm, diameter_mm, yield_strength_mpa
    )

    # P5 is always computed so the two effective-dimension characterizations
    # remain defined for every outer ILI realization. AAF separately selects the
    # formal model assessment pressure (RSTRENG vs PSQR P5).
    profiles = generate_psqr_profiles_from_grid(
        depth_grid, wall_thickness_mm, cell_width_mm, num_profiles=num_profiles
    )
    pressures, d_avg_all, L_comb_all = [], [], []

    for profile in profiles:
        lengths, depths = convert_psqr_profile(
            profile, depth_grid, wall_thickness_mm, cell_length_mm
        )
        if len(depths) == 0:
            continue
        rsf, d_avg, L_comb, _ = compute_RSF_with_defect_dims(
            depths, lengths, diameter_mm, wall_thickness_mm
        )
        p_burst = burst_pressure(wall_thickness_mm, diameter_mm, yield_strength_mpa, rsf)
        pressures.append(float(p_burst))
        d_avg_all.append(float(d_avg))
        L_comb_all.append(float(L_comb))

    if not pressures:
        raise RuntimeError('PSQR generated no valid plausible-profile pressure results.')

    pressures_array = np.asarray(pressures, dtype=float)
    d_avg_array = np.asarray(d_avg_all, dtype=float)
    L_comb_array = np.asarray(L_comb_all, dtype=float)
    psqr_p5 = float(np.percentile(pressures_array, 5))
    idx_p5 = int(np.argmin(np.abs(pressures_array - psqr_p5)))
    d_avg_p5 = float(d_avg_array[idx_p5])
    L_comb_p5 = float(L_comb_array[idx_p5])

    if aaf_info['use_rstreng_fallback']:
        assessment_pressure = float(aaf_info['whole_rstreng_pressure_mpa'])
        assessment_method = 'RSTRENG'
        assessment_d_avg = float(aaf_info['whole_rstreng_d_avg_mm'])
        assessment_L_comb = float(aaf_info['whole_rstreng_L_comb_mm'])
    else:
        assessment_pressure = psqr_p5
        assessment_method = 'PSQR_P5'
        assessment_d_avg = d_avg_p5
        assessment_L_comb = L_comb_p5

    return {
        'psqr_p5_mpa': psqr_p5,
        'pressures_all_mpa': pressures_array,
        'd_avg_all_mm': d_avg_array,
        'L_comb_all_mm': L_comb_array,
        'd_avg_p5_mm': d_avg_p5,
        'L_comb_p5_mm': L_comb_p5,
        'assessment_pressure_mpa': assessment_pressure,
        'assessment_method': assessment_method,
        'assessment_d_avg_mm': assessment_d_avg,
        'assessment_L_comb_mm': assessment_L_comb,
        **aaf_info,
    }

# ============================================================================
# PSQR MONTE CARLO CLASS
# ============================================================================

# ---------------------------------------------------------------------------
# Multiprocessing worker for the OUTER PSQR Monte Carlo level.
# Defined at module level so Windows ``spawn`` can pickle it safely when the
# script is launched from Anaconda Prompt with ``python <script>.py``.
# ---------------------------------------------------------------------------
_PSQR_WORKER_CONTEXT = None


def _init_psqr_worker(context):
    global _PSQR_WORKER_CONTEXT
    _PSQR_WORKER_CONTEXT = context


def _psqr_outer_sample_worker(sample_index):
    """Compute one independent outer PSQR realization reproducibly."""
    c = _PSQR_WORKER_CONTEXT
    if c is None:
        raise RuntimeError('PSQR worker context has not been initialized.')

    seed = int((c['random_seed'] + 104729 * (int(sample_index) + 1)) % (2**32 - 1))
    np.random.seed(seed)
    depth_grid = np.array(c['depth_grid_original'], dtype=float, copy=True)

    for j, (row, col) in enumerate(c['non_nan_cells']):
        mu = c['original_depths'][j]
        if mu == 0.0:
            depth_grid[row, col] = 0.0
        else:
            depth_grid[row, col] = sample_truncated_normal(
                mu, c['sigma_pct'], lower=0.0, upper=100.0
            )

    finite_depths = depth_grid[np.isfinite(depth_grid)]
    mean_depth_pct = float(np.mean(finite_depths))
    max_depth_pct = float(np.max(finite_depths))

    feature_length_mm = sample_truncated_normal(
        c['nominal_feature_length_mm'], c['length_std_mm'],
        lower=np.finfo(float).eps, upper=np.inf
    ) if c['length_std_mm'] > 0 else c['nominal_feature_length_mm']
    feature_width_mm = sample_truncated_normal(
        c['nominal_feature_width_mm'], c['width_std_mm'],
        lower=np.finfo(float).eps, upper=np.inf
    ) if c['width_std_mm'] > 0 else c['nominal_feature_width_mm']

    cell_length_mm = c['nominal_cell_length'] * (
        feature_length_mm / c['nominal_feature_length_mm']
    )
    cell_width_mm = c['nominal_cell_width'] * (
        feature_width_mm / c['nominal_feature_width_mm']
    )

    result = psqr_burst_pressure_from_grid(
        depth_grid,
        cell_length_mm=cell_length_mm,
        cell_width_mm=cell_width_mm,
        wall_thickness_mm=c['t'],
        diameter_mm=c['D'],
        yield_strength_mpa=c['sy'],
        num_profiles=c['psqr_profiles_per_sample'],
    )

    return {
        'sample_index': int(sample_index),
        'p5': float(result['psqr_p5_mpa']),
        'assessment_pressure_mpa': float(result['assessment_pressure_mpa']),
        'used_rstreng_fallback': float(result['use_rstreng_fallback']),
        'aaf': float(result['aaf']),
        'whole_rstreng_pressure_mpa': float(result['whole_rstreng_pressure_mpa']),
        'governing_strip_pressure_mpa': float(result['governing_strip_pressure_mpa']),
        'governing_strip_start_row': float(result['governing_strip_start_row']),
        'governing_strip_end_row': float(result['governing_strip_end_row']),
        'aaf_target_strip_width_mm': float(result['target_strip_width_mm']),
        'aaf_actual_strip_width_mm': float(result['actual_strip_width_mm']),
        'assessment_d_avg_mm': float(result['assessment_d_avg_mm']),
        'assessment_L_comb_mm': float(result['assessment_L_comb_mm']),
        'd_avg_p5': float(result['d_avg_p5_mm']),
        'L_comb_p5': float(result['L_comb_p5_mm']),
        'feature_length_mm': float(feature_length_mm),
        'feature_width_mm': float(feature_width_mm),
        'cell_length_mm': float(cell_length_mm),
        'cell_width_mm': float(cell_width_mm),
        'mean_depth_pct': mean_depth_pct,
        'max_depth_pct': max_depth_pct,
        'pressures_all': np.asarray(result['pressures_all_mpa'], dtype=np.float64),
        'd_avg_all': np.asarray(result['d_avg_all_mm'], dtype=np.float64),
        'L_comb_all': np.asarray(result['L_comb_all_mm'], dtype=np.float64),
    }

class PSQRMonteCarlo:
    """
    Nested Monte Carlo uncertainty propagation for PSQR.

    OUTER STOCHASTIC LEVEL
        ILI measurement uncertainty is applied to:
          * every reported corrosion depth;
          * the reported overall axial feature length;
          * the reported overall circumferential feature width.

    INNER STOCHASTIC LEVEL
        For every uncertain ILI realization, PSQR generates an ensemble of
        plausible paths using depth-weighted starting points and the original
        depth/proximity path-transition rule. The 5th-percentile burst pressure
        of that ensemble becomes the outer-realization response.

    FEATURE-LEVEL GEOMETRY UNCERTAINTY
        Reported +/- length and width sizing accuracies describe uncertainty in
        the overall feature dimensions. They are therefore sampled once per outer
        Monte Carlo realization and used to coherently rescale the entire grid,
        rather than being applied independently to every nominal 5-mm cell.
    """

    def __init__(self, excel_path, material_params,
                 depth_absolute_error_pct_t=0.07,
                 length_absolute_error_mm=7.0,
                 width_absolute_error_mm=9.0,
                 confidence_level=0.80,
                 nominal_cell_length=5.0,
                 nominal_cell_width=5.0,
                 psqr_profiles_per_sample=500,
                 random_seed=2026,
                 n_workers=None,
                 checkpoint_every=DEFAULT_CHECKPOINT_EVERY,
                 max_method1_in_memory=DEFAULT_MAX_METHOD1_IN_MEMORY,
                 output_root=DEFAULT_OUTPUT_ROOT,
                 save_raw_method1_chunks=True):
        self.excel_path = excel_path
        self.material_params = material_params
        self.depth_absolute_error_pct_t = float(depth_absolute_error_pct_t)
        self.length_absolute_error_mm = float(length_absolute_error_mm)
        self.width_absolute_error_mm = float(width_absolute_error_mm)
        self.confidence_level = float(confidence_level)
        self.nominal_cell_length = float(nominal_cell_length)
        self.nominal_cell_width = float(nominal_cell_width)
        self.psqr_profiles_per_sample = int(psqr_profiles_per_sample)
        self.random_seed = int(random_seed)
        self.n_workers = int(DEFAULT_N_WORKERS if n_workers is None else n_workers)
        self.n_workers = max(1, self.n_workers)
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.max_method1_in_memory = max(10_000, int(max_method1_in_memory))
        self.output_root = Path(output_root)
        self.save_raw_method1_chunks = bool(save_raw_method1_chunks)
        self.current_run_dir = None
        self.current_chunk_dir = None
        self.current_log_path = None

        self.z_score = norm.ppf((1.0 + self.confidence_level) / 2.0)
        t = float(material_params['thickness'])
        self.depth_std_mm = (self.depth_absolute_error_pct_t * t) / self.z_score
        self.length_std_mm = self.length_absolute_error_mm / self.z_score
        self.width_std_mm = self.width_absolute_error_mm / self.z_score
        self.depth_absolute_error_mm = self.depth_absolute_error_pct_t * t

        self.results = {}
        self.convergence_results = {}
        self.profile_convergence_results = {}

    def _read_original_grid(self):
        df_original = pd.read_excel(self.excel_path, header=None)
        return df_original, df_original.values.astype(float)

    def _nominal_feature_geometry(self, depth_grid):
        span = get_active_grid_span(depth_grid)
        nominal_length = span['n_cols_span'] * self.nominal_cell_length
        nominal_width = span['n_rows_span'] * self.nominal_cell_width
        return span, float(nominal_length), float(nominal_width)

    def _sample_feature_geometry(self, nominal_feature_length_mm,
                                 nominal_feature_width_mm):
        """
        Sample one coherent feature length and width for an outer realization.

        The reported ILI bounds are converted to Normal standard deviations by
        sigma = bound / z, using the supplied certainty level. Only positivity
        is imposed; because the random variable is the overall feature size,
        not one 5-mm cell, this truncation avoids the strong bias that can be
        introduced by independent per-cell geometry perturbations.
        """
        if self.length_std_mm > 0:
            feature_length = sample_truncated_normal(
                nominal_feature_length_mm,
                self.length_std_mm,
                lower=np.finfo(float).eps,
                upper=np.inf,
            )
        else:
            feature_length = nominal_feature_length_mm

        if self.width_std_mm > 0:
            feature_width = sample_truncated_normal(
                nominal_feature_width_mm,
                self.width_std_mm,
                lower=np.finfo(float).eps,
                upper=np.inf,
            )
        else:
            feature_width = nominal_feature_width_mm

        # Uniform affine rescaling preserves the grid topology and maps the
        # sampled overall feature dimensions back to one coherent cell spacing.
        cell_length = self.nominal_cell_length * (
            feature_length / nominal_feature_length_mm
        )
        cell_width = self.nominal_cell_width * (
            feature_width / nominal_feature_width_mm
        )

        return (
            float(feature_length),
            float(feature_width),
            float(cell_length),
            float(cell_width),
        )

    def _build_run_signature(self, num_samples):
        """Stable signature; algorithm or source-code changes invalidate checkpoints."""
        excel_abs = str(Path(self.excel_path).resolve())
        try:
            st = os.stat(excel_abs)
            file_meta = {'path': excel_abs, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
        except OSError:
            file_meta = {'path': excel_abs}

        try:
            code_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        except Exception:
            code_sha256 = 'unavailable'

        payload = {
            'algorithm_version': ALGORITHM_VERSION,
            'source_code_sha256': code_sha256,
            'input_file': file_meta,
            'material_params': self.material_params,
            'depth_absolute_error_pct_t': self.depth_absolute_error_pct_t,
            'length_absolute_error_mm': self.length_absolute_error_mm,
            'width_absolute_error_mm': self.width_absolute_error_mm,
            'confidence_level': self.confidence_level,
            'nominal_cell_length': self.nominal_cell_length,
            'nominal_cell_width': self.nominal_cell_width,
            'psqr_profiles_per_sample': self.psqr_profiles_per_sample,
            'random_seed': self.random_seed,
            'num_samples': int(num_samples),
        }
        payload_text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(payload_text.encode('utf-8')).hexdigest(), payload

    def _prepare_run_directory(self, num_samples):
        signature, payload = self._build_run_signature(num_samples)
        run_name = (
            f"N{int(num_samples)}_P{self.psqr_profiles_per_sample}_"
            f"seed{self.random_seed}_{signature[:8]}"
        )
        run_dir = self.output_root / run_name
        chunk_dir = run_dir / 'method1_chunks'
        run_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        self.current_run_dir = run_dir
        self.current_chunk_dir = chunk_dir
        self.current_log_path = run_dir / 'psqr_run.log'

        config_path = run_dir / 'run_configuration.json'
        config_payload = dict(payload)
        config_payload.update({
            'signature': signature,
            'n_workers': self.n_workers,
            'checkpoint_every': self.checkpoint_every,
            'max_method1_in_memory': self.max_method1_in_memory,
            'save_raw_method1_chunks': self.save_raw_method1_chunks,
        })
        config_path.write_text(json.dumps(config_payload, indent=2, default=str))
        return signature, run_dir, chunk_dir

    def _log(self, message):
        """Print progress and append it to the run log."""
        text = str(message)
        print(text, flush=True)
        if self.current_log_path is not None:
            with open(self.current_log_path, 'a', encoding='utf-8') as f:
                f.write(text + '\n')

    @staticmethod
    def _allocate_outer_arrays(num_samples):
        n = int(num_samples)
        return {
            'burst_pressures_5th': np.full(n, np.nan, dtype=np.float64),
            'assessment_pressures_mpa': np.full(n, np.nan, dtype=np.float64),
            'used_rstreng_fallback': np.zeros(n, dtype=np.float64),
            'aaf_values': np.full(n, np.nan, dtype=np.float64),
            'whole_rstreng_pressures_mpa': np.full(n, np.nan, dtype=np.float64),
            'governing_strip_pressures_mpa': np.full(n, np.nan, dtype=np.float64),
            'governing_strip_start_rows': np.full(n, np.nan, dtype=np.float64),
            'governing_strip_end_rows': np.full(n, np.nan, dtype=np.float64),
            'aaf_target_strip_widths_mm': np.full(n, np.nan, dtype=np.float64),
            'aaf_actual_strip_widths_mm': np.full(n, np.nan, dtype=np.float64),
            'assessment_d_avg_mm': np.full(n, np.nan, dtype=np.float64),
            'assessment_L_comb_mm': np.full(n, np.nan, dtype=np.float64),
            'critical_d_avg_mm': np.full(n, np.nan, dtype=np.float64),
            'critical_L_comb_mm': np.full(n, np.nan, dtype=np.float64),
            'critical_d_avg_pct': np.full(n, np.nan, dtype=np.float64),
            'sampled_feature_lengths_mm': np.full(n, np.nan, dtype=np.float64),
            'sampled_feature_widths_mm': np.full(n, np.nan, dtype=np.float64),
            'sampled_cell_lengths_mm': np.full(n, np.nan, dtype=np.float64),
            'sampled_cell_widths_mm': np.full(n, np.nan, dtype=np.float64),
            'sampled_mean_depth_pct': np.full(n, np.nan, dtype=np.float64),
            'sampled_max_depth_pct': np.full(n, np.nan, dtype=np.float64),
        }

    @staticmethod
    def _new_method1_state():
        return {
            'count': 0,
            'sum_pressure': 0.0,
            'sumsq_pressure': 0.0,
            'sum_depth': 0.0,
            'sumsq_depth': 0.0,
            'sum_length': 0.0,
            'sumsq_length': 0.0,
            'reservoir_keys': np.empty(0, dtype=np.float64),
            'reservoir_pressure': np.empty(0, dtype=np.float64),
            'reservoir_depth': np.empty(0, dtype=np.float64),
            'reservoir_length': np.empty(0, dtype=np.float64),
            'reservoir_mc_index': np.empty(0, dtype=np.int64),
        }

    def _update_method1_state(self, state, pressures, depths, lengths, mc_indices,
                              deterministic_seed):
        """Exact streaming moments + uniform priority reservoir for plotting/fits."""
        p = np.asarray(pressures, dtype=np.float64)
        d = np.asarray(depths, dtype=np.float64)
        L = np.asarray(lengths, dtype=np.float64)
        m = np.asarray(mc_indices, dtype=np.int64)
        valid = np.isfinite(p) & np.isfinite(d) & np.isfinite(L)
        p, d, L, m = p[valid], d[valid], L[valid], m[valid]
        if len(p) == 0:
            return state

        state['count'] += int(len(p))
        state['sum_pressure'] += float(np.sum(p, dtype=np.float64))
        state['sumsq_pressure'] += float(np.dot(p, p))
        state['sum_depth'] += float(np.sum(d, dtype=np.float64))
        state['sumsq_depth'] += float(np.dot(d, d))
        state['sum_length'] += float(np.sum(L, dtype=np.float64))
        state['sumsq_length'] += float(np.dot(L, L))

        # Priority sampling: assign each profile an independent U(0,1) key and
        # retain the largest K. This is an unbiased uniform sample of all profiles
        # seen so far and is cheap to checkpoint.
        rng = np.random.default_rng(int(deterministic_seed))
        keys = rng.random(len(p))
        keys_all = np.concatenate([state['reservoir_keys'], keys])
        p_all = np.concatenate([state['reservoir_pressure'], p])
        d_all = np.concatenate([state['reservoir_depth'], d])
        L_all = np.concatenate([state['reservoir_length'], L])
        m_all = np.concatenate([state['reservoir_mc_index'], m])

        cap = self.max_method1_in_memory
        if len(keys_all) > cap:
            keep = np.argpartition(keys_all, -cap)[-cap:]
            state['reservoir_keys'] = keys_all[keep]
            state['reservoir_pressure'] = p_all[keep]
            state['reservoir_depth'] = d_all[keep]
            state['reservoir_length'] = L_all[keep]
            state['reservoir_mc_index'] = m_all[keep]
        else:
            state['reservoir_keys'] = keys_all
            state['reservoir_pressure'] = p_all
            state['reservoir_depth'] = d_all
            state['reservoir_length'] = L_all
            state['reservoir_mc_index'] = m_all
        return state

    @staticmethod
    def _stream_mean_std(total, sum_value, sumsq_value):
        if total <= 0:
            return np.nan, np.nan
        mean = sum_value / total
        var = max(0.0, sumsq_value / total - mean * mean)
        return float(mean), float(math.sqrt(var))

    def _save_method1_chunk(self, start_index, end_index, pressures, depths,
                            lengths, mc_indices):
        if not self.save_raw_method1_chunks:
            return None
        path = self.current_chunk_dir / (
            f'method1_{int(start_index):07d}_{int(end_index)-1:07d}.npz'
        )
        tmp = path.with_suffix('.tmp.npz')
        np.savez_compressed(
            tmp,
            pressure=np.asarray(pressures, dtype=np.float64),
            depth=np.asarray(depths, dtype=np.float64),
            length=np.asarray(lengths, dtype=np.float64),
            mc_index=np.asarray(mc_indices, dtype=np.int64),
        )
        os.replace(tmp, path)
        return path

    def _save_checkpoint(self, signature, next_index, outer, method1_state):
        path = self.current_run_dir / 'psqr_checkpoint.npz'
        tmp = self.current_run_dir / 'psqr_checkpoint.tmp.npz'
        payload = {
            'signature': np.array(signature),
            'next_index': np.array(int(next_index), dtype=np.int64),
            'method1_count': np.array(method1_state['count'], dtype=np.int64),
            'method1_sum_pressure': np.array(method1_state['sum_pressure']),
            'method1_sumsq_pressure': np.array(method1_state['sumsq_pressure']),
            'method1_sum_depth': np.array(method1_state['sum_depth']),
            'method1_sumsq_depth': np.array(method1_state['sumsq_depth']),
            'method1_sum_length': np.array(method1_state['sum_length']),
            'method1_sumsq_length': np.array(method1_state['sumsq_length']),
            'reservoir_keys': method1_state['reservoir_keys'],
            'reservoir_pressure': method1_state['reservoir_pressure'],
            'reservoir_depth': method1_state['reservoir_depth'],
            'reservoir_length': method1_state['reservoir_length'],
            'reservoir_mc_index': method1_state['reservoir_mc_index'],
        }
        payload.update(outer)
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, path)
        return path

    def _load_checkpoint(self, signature, num_samples):
        path = self.current_run_dir / 'psqr_checkpoint.npz'
        if not path.exists():
            return 0, self._allocate_outer_arrays(num_samples), self._new_method1_state()

        with np.load(path, allow_pickle=False) as z:
            saved_signature = str(z['signature'].item())
            if saved_signature != signature:
                raise RuntimeError(
                    'Checkpoint configuration does not match this run. '
                    'Use the matching input/settings or remove that checkpoint directory.'
                )
            outer = self._allocate_outer_arrays(num_samples)
            for key in outer:
                arr = np.asarray(z[key])
                if len(arr) != int(num_samples):
                    raise RuntimeError('Checkpoint sample count does not match requested run.')
                outer[key] = arr.copy()

            state = self._new_method1_state()
            state.update({
                'count': int(z['method1_count'].item()),
                'sum_pressure': float(z['method1_sum_pressure'].item()),
                'sumsq_pressure': float(z['method1_sumsq_pressure'].item()),
                'sum_depth': float(z['method1_sum_depth'].item()),
                'sumsq_depth': float(z['method1_sumsq_depth'].item()),
                'sum_length': float(z['method1_sum_length'].item()),
                'sumsq_length': float(z['method1_sumsq_length'].item()),
                'reservoir_keys': np.asarray(z['reservoir_keys']).copy(),
                'reservoir_pressure': np.asarray(z['reservoir_pressure']).copy(),
                'reservoir_depth': np.asarray(z['reservoir_depth']).copy(),
                'reservoir_length': np.asarray(z['reservoir_length']).copy(),
                'reservoir_mc_index': np.asarray(z['reservoir_mc_index']).copy(),
            })
            next_index = int(z['next_index'].item())
        return next_index, outer, state

    def _write_final_numerical_outputs(self, outer, method1_state, summary):
        """Write compact outer results + summary; complete Method-1 data remain in chunks."""
        out = self.current_run_dir
        np.savez_compressed(out / 'psqr_outer_results.npz', **outer)
        pd.DataFrame({
            'sample': np.arange(len(outer['burst_pressures_5th'])),
            'PSQR_P5_MPa': outer['burst_pressures_5th'],
            'AAF': outer['aaf_values'],
            'used_RSTRENG_fallback': outer['used_rstreng_fallback'].astype(int),
            'selected_assessment_pressure_MPa': outer['assessment_pressures_mpa'],
            'whole_anomaly_RSTRENG_pressure_MPa': outer['whole_rstreng_pressures_mpa'],
            'governing_6t_strip_pressure_MPa': outer['governing_strip_pressures_mpa'],
            'governing_6t_strip_start_row': outer['governing_strip_start_rows'],
            'governing_6t_strip_end_row': outer['governing_strip_end_rows'],
            'AAF_target_strip_width_mm': outer['aaf_target_strip_widths_mm'],
            'AAF_actual_strip_width_mm': outer['aaf_actual_strip_widths_mm'],
            'selected_assessment_depth_mm': outer['assessment_d_avg_mm'],
            'selected_assessment_length_mm': outer['assessment_L_comb_mm'],
            'P5_associated_depth_mm': outer['critical_d_avg_mm'],
            'P5_associated_length_mm': outer['critical_L_comb_mm'],
            'feature_length_mm': outer['sampled_feature_lengths_mm'],
            'feature_width_mm': outer['sampled_feature_widths_mm'],
            'cell_length_mm': outer['sampled_cell_lengths_mm'],
            'cell_width_mm': outer['sampled_cell_widths_mm'],
            'mean_depth_pct_t': outer['sampled_mean_depth_pct'],
            'max_depth_pct_t': outer['sampled_max_depth_pct'],
        }).to_csv(out / 'psqr_outer_results.csv', index=False)
        (out / 'psqr_summary.json').write_text(
            json.dumps(summary, indent=2, default=lambda x: float(x) if np.isscalar(x) else str(x))
        )

    def compute_nominal_psqr_diagnostics(self):
        """Compute reproducible nominal P5 plus AAF/RSTRENG diagnostics."""
        _, depth_grid = self._read_original_grid()
        t = self.material_params['thickness']
        D = self.material_params['diameter']
        sy = self.material_params['yield_strength']
        state = np.random.get_state()
        np.random.seed(self.random_seed + 11)
        try:
            result = psqr_burst_pressure_from_grid(
                depth_grid,
                cell_length_mm=self.nominal_cell_length,
                cell_width_mm=self.nominal_cell_width,
                wall_thickness_mm=t,
                diameter_mm=D,
                yield_strength_mpa=sy,
                num_profiles=self.psqr_profiles_per_sample,
            )
        finally:
            np.random.set_state(state)
        return result

    def compute_nominal_psqr(self):
        """Return the nominal raw PSQR P5 for backward compatibility."""
        return float(self.compute_nominal_psqr_diagnostics()['psqr_p5_mpa'])

    def fit_dimension_distributions(self):
        """Fit candidate parametric distributions to the main PSQR outputs."""
        if not self.results:
            print('No results available. Run Monte Carlo first.')
            return None

        res = self.results
        valid_mask = res['valid_mask']

        fit_results = {
            'psqr_burst_pressure': fit_best_distribution(
                res['burst_pressures_5th'][valid_mask],
                label='PSQR - 5th-Percentile Burst Pressure',
                validation_sample_size=int(np.sum(valid_mask)),
                random_state=100,
            ),
            'method1_depth': fit_best_distribution(
                res['method1_all_profile_d_avg_mm'],
                label='Method 1 - Effective Depth: All Plausible Profiles',
                validation_sample_size=len(res['method1_all_profile_d_avg_mm']),
                random_state=101,
            ),
            'method1_length': fit_best_distribution(
                res['method1_all_profile_L_comb_mm'],
                label='Method 1 - Effective Length: All Plausible Profiles',
                validation_sample_size=len(res['method1_all_profile_L_comb_mm']),
                random_state=102,
            ),
            'method2_depth': fit_best_distribution(
                res['method2_d_avg_at_5th_mm'][valid_mask],
                label='Method 2 - P5-Associated Effective Depth',
                validation_sample_size=int(np.sum(valid_mask)),
                random_state=103,
            ),
            'method2_length': fit_best_distribution(
                res['method2_L_comb_at_5th_mm'][valid_mask],
                label='Method 2 - P5-Associated Effective Length',
                validation_sample_size=int(np.sum(valid_mask)),
                random_state=104,
            ),
        }

        self.results['distribution_fits'] = fit_results
        return fit_results

    def run_monte_carlo(self, num_samples=1000, generate_plots=False,
                        fit_distributions=True):
        """Run nested PSQR ILI-uncertainty propagation with strip-based AAF diagnostics."""
        num_samples = int(num_samples)
        if num_samples <= 0:
            raise ValueError('num_samples must be positive.')

        signature, run_dir, chunk_dir = self._prepare_run_directory(num_samples)
        self._log('')
        self._log('=' * 78)
        self._log(f'PSQR NESTED MONTE CARLO | outer samples = {num_samples:,}')
        self._log('=' * 78)
        self._log(f'Algorithm version: {ALGORITHM_VERSION}')
        self._log(f'Input grid: {Path(self.excel_path).resolve()}')
        self._log(f'Run directory: {run_dir.resolve()}')
        self._log(f'CPU workers: {self.n_workers}')
        self._log(f'Checkpoint interval: {self.checkpoint_every} outer samples')
        self._log(f'Method-1 in-memory cap: {self.max_method1_in_memory:,} profiles')
        self._log(f'PSQR plausible profiles / outer sample: {self.psqr_profiles_per_sample:,}')
        self._log(
            f'ILI sizing @ {self.confidence_level*100:.0f}% certainty: '
            f'depth +/-{self.depth_absolute_error_pct_t:.3f}t, '
            f'length +/-{self.length_absolute_error_mm:.1f} mm, '
            f'width +/-{self.width_absolute_error_mm:.1f} mm'
        )

        df_original, depth_grid_original = self._read_original_grid()
        t = float(self.material_params['thickness'])
        D = float(self.material_params['diameter'])
        sy = float(self.material_params['yield_strength'])

        non_nan_mask = np.isfinite(depth_grid_original)
        non_nan_cells = [tuple(map(int, rc)) for rc in np.argwhere(non_nan_mask)]
        original_depths = [float(depth_grid_original[row, col]) for row, col in non_nan_cells]
        span_info, nominal_feature_length_mm, nominal_feature_width_mm = (
            self._nominal_feature_geometry(depth_grid_original)
        )
        sigma_pct = (self.depth_std_mm / t) * 100.0

        worker_context = {
            'depth_grid_original': depth_grid_original,
            'non_nan_cells': non_nan_cells,
            'original_depths': original_depths,
            'sigma_pct': float(sigma_pct),
            'nominal_feature_length_mm': nominal_feature_length_mm,
            'nominal_feature_width_mm': nominal_feature_width_mm,
            'length_std_mm': self.length_std_mm,
            'width_std_mm': self.width_std_mm,
            'nominal_cell_length': self.nominal_cell_length,
            'nominal_cell_width': self.nominal_cell_width,
            't': t,
            'D': D,
            'sy': sy,
            'psqr_profiles_per_sample': self.psqr_profiles_per_sample,
            'random_seed': self.random_seed,
        }

        next_index, outer, method1_state = self._load_checkpoint(signature, num_samples)
        if next_index > 0:
            self._log(
                f'Checkpoint detected: {next_index:,}/{num_samples:,} outer samples '
                f'already complete. Resuming automatically.'
            )
        else:
            self._log('No compatible checkpoint found. Starting a new run.')
        if next_index >= num_samples:
            self._log('Checkpoint already contains the complete simulation. Reusing saved results.')

        start_wall = time.time()
        processed_this_session = 0
        executor = None
        try:
            if next_index < num_samples:
                if self.n_workers > 1:
                    executor = ProcessPoolExecutor(
                        max_workers=self.n_workers,
                        mp_context=mp.get_context('spawn'),
                        initializer=_init_psqr_worker,
                        initargs=(worker_context,),
                    )
                else:
                    _init_psqr_worker(worker_context)

                for chunk_start in range(next_index, num_samples, self.checkpoint_every):
                    chunk_end = min(num_samples, chunk_start + self.checkpoint_every)
                    indices = list(range(chunk_start, chunk_end))
                    if executor is not None:
                        chunk_results = list(executor.map(_psqr_outer_sample_worker, indices, chunksize=1))
                    else:
                        chunk_results = [_psqr_outer_sample_worker(i) for i in indices]

                    chunk_pressures, chunk_depths, chunk_lengths, chunk_mc_indices = [], [], [], []
                    for r in chunk_results:
                        i = r['sample_index']
                        outer['burst_pressures_5th'][i] = r['p5']
                        outer['assessment_pressures_mpa'][i] = r['assessment_pressure_mpa']
                        outer['used_rstreng_fallback'][i] = r['used_rstreng_fallback']
                        outer['aaf_values'][i] = r['aaf']
                        outer['whole_rstreng_pressures_mpa'][i] = r['whole_rstreng_pressure_mpa']
                        outer['governing_strip_pressures_mpa'][i] = r['governing_strip_pressure_mpa']
                        outer['governing_strip_start_rows'][i] = r['governing_strip_start_row']
                        outer['governing_strip_end_rows'][i] = r['governing_strip_end_row']
                        outer['aaf_target_strip_widths_mm'][i] = r['aaf_target_strip_width_mm']
                        outer['aaf_actual_strip_widths_mm'][i] = r['aaf_actual_strip_width_mm']
                        outer['assessment_d_avg_mm'][i] = r['assessment_d_avg_mm']
                        outer['assessment_L_comb_mm'][i] = r['assessment_L_comb_mm']
                        outer['critical_d_avg_mm'][i] = r['d_avg_p5']
                        outer['critical_L_comb_mm'][i] = r['L_comb_p5']
                        outer['critical_d_avg_pct'][i] = (r['d_avg_p5'] / t) * 100.0
                        outer['sampled_feature_lengths_mm'][i] = r['feature_length_mm']
                        outer['sampled_feature_widths_mm'][i] = r['feature_width_mm']
                        outer['sampled_cell_lengths_mm'][i] = r['cell_length_mm']
                        outer['sampled_cell_widths_mm'][i] = r['cell_width_mm']
                        outer['sampled_mean_depth_pct'][i] = r['mean_depth_pct']
                        outer['sampled_max_depth_pct'][i] = r['max_depth_pct']

                        n_inner = len(r['pressures_all'])
                        if n_inner:
                            chunk_pressures.append(r['pressures_all'])
                            chunk_depths.append(r['d_avg_all'])
                            chunk_lengths.append(r['L_comb_all'])
                            chunk_mc_indices.append(np.full(n_inner, i, dtype=np.int64))

                    if chunk_pressures:
                        cp, cd, cL, cm = (np.concatenate(chunk_pressures), np.concatenate(chunk_depths),
                                          np.concatenate(chunk_lengths), np.concatenate(chunk_mc_indices))
                    else:
                        cp = cd = cL = np.empty(0, dtype=np.float64)
                        cm = np.empty(0, dtype=np.int64)

                    self._update_method1_state(
                        method1_state, cp, cd, cL, cm,
                        deterministic_seed=self.random_seed + 500_000 + chunk_start,
                    )
                    self._save_method1_chunk(chunk_start, chunk_end, cp, cd, cL, cm)
                    self._save_checkpoint(signature, chunk_end, outer, method1_state)

                    processed_this_session += chunk_end - chunk_start
                    elapsed = time.time() - start_wall
                    rate = processed_this_session / elapsed if elapsed > 0 else np.nan
                    remaining = num_samples - chunk_end
                    eta_s = remaining / rate if np.isfinite(rate) and rate > 0 else np.nan
                    eta_text = (f'{eta_s/3600:.2f} h' if np.isfinite(eta_s) and eta_s >= 3600
                                else f'{eta_s/60:.1f} min' if np.isfinite(eta_s) else 'n/a')
                    self._log(
                        f'Completed {chunk_end:,}/{num_samples:,} outer samples | '
                        f'{rate:.3f} samples/s | ETA {eta_text} | '
                        f'Method-1 profiles processed {method1_state["count"]:,}'
                    )
        except KeyboardInterrupt:
            self._log('Run interrupted by user. The last completed checkpoint is safe; rerun to resume.')
            raise
        except Exception as exc:
            self._log(f'Run stopped because of an error: {type(exc).__name__}: {exc}')
            self._log('The last completed checkpoint is safe; rerun the same configuration to resume.')
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)

        p5_values = outer['burst_pressures_5th']
        assessment_values = outer['assessment_pressures_mpa']
        aaf_values = outer['aaf_values']
        fallback_flags = outer['used_rstreng_fallback']
        critical_d_avg_mm = outer['critical_d_avg_mm']
        critical_L_comb_mm = outer['critical_L_comb_mm']
        critical_d_avg_pct = outer['critical_d_avg_pct']

        valid_mask = np.isfinite(p5_values)
        valid_p5 = p5_values[valid_mask]
        valid_assessment = assessment_values[valid_mask]
        valid_d_avg = critical_d_avg_mm[valid_mask]
        valid_L_comb = critical_L_comb_mm[valid_mask]
        valid_aaf = aaf_values[valid_mask]
        valid_fallback = fallback_flags[valid_mask] > 0.5

        total_m1 = method1_state['count']
        method1_mean_pressure, method1_std_pressure = self._stream_mean_std(
            total_m1, method1_state['sum_pressure'], method1_state['sumsq_pressure'])
        method1_mean_d_avg, method1_std_d_avg = self._stream_mean_std(
            total_m1, method1_state['sum_depth'], method1_state['sumsq_depth'])
        method1_mean_L_comb, method1_std_L_comb = self._stream_mean_std(
            total_m1, method1_state['sum_length'], method1_state['sumsq_length'])

        all_profile_pressures = method1_state['reservoir_pressure']
        all_profile_d_avg_mm = method1_state['reservoir_depth']
        all_profile_L_comb_mm = method1_state['reservoir_length']
        all_profile_mc_indices = method1_state['reservoir_mc_index']

        if len(valid_p5):
            mean_pressure = float(np.mean(valid_p5))
            std_pressure = float(np.std(valid_p5))
            mean_assessment_pressure = float(np.mean(valid_assessment))
            std_assessment_pressure = float(np.std(valid_assessment))
            mean_d_avg, std_d_avg = float(np.mean(valid_d_avg)), float(np.std(valid_d_avg))
            mean_L_comb, std_L_comb = float(np.mean(valid_L_comb)), float(np.std(valid_L_comb))
            mean_aaf, min_aaf, max_aaf = map(float, (np.mean(valid_aaf), np.min(valid_aaf), np.max(valid_aaf)))
            fallback_count = int(np.sum(valid_fallback))
            fallback_fraction = float(fallback_count / len(valid_fallback))
            nominal_diag = self.compute_nominal_psqr_diagnostics()
            nominal_p5 = float(nominal_diag['psqr_p5_mpa'])
        else:
            mean_pressure = std_pressure = mean_assessment_pressure = std_assessment_pressure = np.nan
            mean_d_avg = std_d_avg = mean_L_comb = std_L_comb = np.nan
            mean_aaf = min_aaf = max_aaf = np.nan
            fallback_count, fallback_fraction = 0, np.nan
            nominal_diag, nominal_p5 = None, np.nan

        self.results = {
            'samples_used': num_samples, 'valid_mask': valid_mask, 'random_seed': self.random_seed,
            'algorithm_version': ALGORITHM_VERSION, 'run_directory': str(run_dir.resolve()),
            'n_workers': self.n_workers, 'checkpoint_every': self.checkpoint_every,
            'burst_pressures_5th': p5_values, 'mean_5th_percentile': mean_pressure,
            'std_5th_percentile': std_pressure, 'nominal_5th_percentile': nominal_p5,
            'assessment_pressures_mpa': assessment_values,
            'mean_assessment_pressure_mpa': mean_assessment_pressure,
            'std_assessment_pressure_mpa': std_assessment_pressure,
            'aaf_values': aaf_values, 'used_rstreng_fallback': fallback_flags,
            'mean_AAF': mean_aaf, 'min_AAF': min_aaf, 'max_AAF': max_aaf,
            'rstreng_fallback_count': fallback_count,
            'rstreng_fallback_fraction': fallback_fraction,
            'whole_rstreng_pressures_mpa': outer['whole_rstreng_pressures_mpa'],
            'governing_strip_pressures_mpa': outer['governing_strip_pressures_mpa'],
            'governing_strip_start_rows': outer['governing_strip_start_rows'],
            'governing_strip_end_rows': outer['governing_strip_end_rows'],
            'aaf_target_strip_widths_mm': outer['aaf_target_strip_widths_mm'],
            'aaf_actual_strip_widths_mm': outer['aaf_actual_strip_widths_mm'],
            'nominal_AAF_diagnostics': nominal_diag,
            'critical_d_avg_mm': critical_d_avg_mm, 'critical_L_comb_mm': critical_L_comb_mm,
            'critical_d_avg_pct': critical_d_avg_pct, 'mean_d_avg_mm': mean_d_avg,
            'std_d_avg_mm': std_d_avg, 'mean_L_comb_mm': mean_L_comb, 'std_L_comb_mm': std_L_comb,
            'method2_d_avg_at_5th_mm': critical_d_avg_mm, 'method2_L_comb_at_5th_mm': critical_L_comb_mm,
            'method2_d_avg_at_5th_pct': critical_d_avg_pct, 'method2_mean_d_avg_mm': mean_d_avg,
            'method2_std_d_avg_mm': std_d_avg, 'method2_mean_L_comb_mm': mean_L_comb,
            'method2_std_L_comb_mm': std_L_comb,
            'method1_all_profile_d_avg_mm': all_profile_d_avg_mm,
            'method1_all_profile_L_comb_mm': all_profile_L_comb_mm,
            'method1_all_profile_pressures': all_profile_pressures,
            'method1_all_profile_mc_indices': all_profile_mc_indices,
            'method1_total_profiles': int(total_m1),
            'method1_in_memory_profiles': int(len(all_profile_pressures)),
            'method1_in_memory_is_reservoir': bool(total_m1 > len(all_profile_pressures)),
            'method1_raw_chunk_directory': str(chunk_dir.resolve()),
            'method1_mean_d_avg_mm': method1_mean_d_avg, 'method1_std_d_avg_mm': method1_std_d_avg,
            'method1_mean_L_comb_mm': method1_mean_L_comb, 'method1_std_L_comb_mm': method1_std_L_comb,
            'method1_mean_profile_pressure': method1_mean_pressure,
            'method1_std_profile_pressure': method1_std_pressure,
            'nominal_feature_length_mm': nominal_feature_length_mm,
            'nominal_feature_width_mm': nominal_feature_width_mm,
            'sampled_feature_lengths_mm': outer['sampled_feature_lengths_mm'],
            'sampled_feature_widths_mm': outer['sampled_feature_widths_mm'],
            'sampled_cell_lengths_mm': outer['sampled_cell_lengths_mm'],
            'sampled_cell_widths_mm': outer['sampled_cell_widths_mm'],
            'sampled_mean_depth_pct': outer['sampled_mean_depth_pct'],
            'sampled_max_depth_pct': outer['sampled_max_depth_pct'],
            'active_grid_span': span_info, 'material_params': self.material_params,
            'uncertainty_params': {
                'depth_error_pct_t': self.depth_absolute_error_pct_t,
                'depth_error_mm': self.depth_absolute_error_mm,
                'length_error_mm': self.length_absolute_error_mm,
                'width_error_mm': self.width_absolute_error_mm,
                'confidence_level': self.confidence_level,
                'depth_std_mm': self.depth_std_mm, 'length_std_mm': self.length_std_mm,
                'width_std_mm': self.width_std_mm,
                'geometry_application': 'overall feature dimensions, coherently rescaled grid',
                'starting_point': 'depth-weighted random starting cell',
                'psqr_interaction_window': 'total max(12t, 50.8 mm), centered on previous cell',
                'AAF': 'minimum RSTRENG pressure ratio over circumferential 6t x L strips',
            },
        }

        summary = {
            'algorithm_version': ALGORITHM_VERSION, 'samples_used': num_samples,
            'profiles_per_outer_sample': self.psqr_profiles_per_sample,
            'method1_total_profiles': int(total_m1),
            'method1_in_memory_profiles': int(len(all_profile_pressures)),
            'nominal_P5_MPa': nominal_p5, 'mean_P5_MPa': mean_pressure,
            'std_P5_MPa': std_pressure,
            'mean_selected_assessment_pressure_MPa': mean_assessment_pressure,
            'std_selected_assessment_pressure_MPa': std_assessment_pressure,
            'AAF_mean': mean_aaf, 'AAF_min': min_aaf, 'AAF_max': max_aaf,
            'RSTRENG_fallback_count': fallback_count,
            'RSTRENG_fallback_fraction': fallback_fraction,
            'nominal_AAF': float(nominal_diag['aaf']) if nominal_diag else np.nan,
            'nominal_selected_method': nominal_diag['assessment_method'] if nominal_diag else 'n/a',
            'nominal_whole_RSTRENG_pressure_MPa': float(nominal_diag['whole_rstreng_pressure_mpa']) if nominal_diag else np.nan,
            'method1_mean_depth_mm': method1_mean_d_avg, 'method1_std_depth_mm': method1_std_d_avg,
            'method1_mean_length_mm': method1_mean_L_comb, 'method1_std_length_mm': method1_std_L_comb,
            'P5_associated_mean_depth_mm': mean_d_avg, 'P5_associated_std_depth_mm': std_d_avg,
            'P5_associated_mean_length_mm': mean_L_comb, 'P5_associated_std_length_mm': std_L_comb,
            'run_directory': str(run_dir.resolve()),
        }
        self._write_final_numerical_outputs(outer, method1_state, summary)
        if fit_distributions:
            self.fit_dimension_distributions()
        if generate_plots:
            self.create_comprehensive_plots()
        self._log('Simulation complete.')
        self._log(f'Numerical results, figures, raw chunks, and log: {run_dir.resolve()}')
        return self.results

    def _plot_start_probability_map(self):
        """Additional methods figure: depth-weighted starting-cell probabilities."""
        _, depth_grid = self._read_original_grid()
        prob_map = depth_weighted_start_probability_map(depth_grid)

        fig, ax = plt.subplots(figsize=(JOURNAL_SINGLE_WIDTH * 1.35,
                                        JOURNAL_SINGLE_WIDTH * 1.05))
        masked = np.ma.masked_where(~np.isfinite(depth_grid), prob_map)
        im = ax.imshow(masked, origin='lower', aspect='auto', cmap='viridis')
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Starting-cell probability')
        ax.set_xlabel('Axial grid column')
        ax.set_ylabel('Circumferential grid row')
        ax.set_title('Depth-weighted PSQR starting probability', loc='left')
        style_axis(ax, grid_axis='none')
        fig.tight_layout()
        save_journal_figure(fig, 'psqr_starting_probability_map', close=True, output_dir=self.current_run_dir)

    def _plot_geometry_uncertainty_diagnostics(self):
        """Additional figure documenting the corrected feature-level geometry UQ."""
        res = self.results
        valid = res['valid_mask']
        pressures = res['burst_pressures_5th'][valid]
        Ls = res['sampled_feature_lengths_mm'][valid]
        Ws = res['sampled_feature_widths_mm'][valid]

        fig, axes = plt.subplots(2, 2, figsize=(JOURNAL_DOUBLE_WIDTH, 5.8))

        ax = axes[0, 0]
        ax.hist(Ls, bins=35, density=True, color=COLORS['primary'],
                alpha=0.72, edgecolor='white', linewidth=0.4)
        ax.axvline(res['nominal_feature_length_mm'], color=COLORS['nominal'],
                   linewidth=1.6, linestyle='--', label='Nominal feature length')
        ax.set_title('(a) Sampled overall axial length', loc='left')
        ax.set_xlabel('Feature length (mm)')
        ax.set_ylabel('Probability density')
        ax.legend()
        style_axis(ax)

        ax = axes[0, 1]
        ax.hist(Ws, bins=35, density=True, color=COLORS['secondary'],
                alpha=0.72, edgecolor='white', linewidth=0.4)
        ax.axvline(res['nominal_feature_width_mm'], color=COLORS['nominal'],
                   linewidth=1.6, linestyle='--', label='Nominal feature width')
        ax.set_title('(b) Sampled overall circumferential width', loc='left')
        ax.set_xlabel('Feature width (mm)')
        ax.set_ylabel('Probability density')
        ax.legend()
        style_axis(ax)

        ax = axes[1, 0]
        ax.scatter(Ls, pressures, s=10, alpha=0.42, color=COLORS['primary'],
                   linewidths=0)
        ax.set_title('(c) P5 response to axial sizing uncertainty', loc='left')
        ax.set_xlabel('Sampled feature length (mm)')
        ax.set_ylabel('PSQR P5 burst pressure (MPa)')
        style_axis(ax)

        ax = axes[1, 1]
        ax.scatter(Ws, pressures, s=10, alpha=0.42, color=COLORS['secondary'],
                   linewidths=0)
        ax.set_title('(d) P5 response to circumferential sizing uncertainty', loc='left')
        ax.set_xlabel('Sampled feature width (mm)')
        ax.set_ylabel('PSQR P5 burst pressure (MPa)')
        style_axis(ax)

        fig.tight_layout()
        save_journal_figure(fig, 'psqr_geometry_uncertainty_diagnostics', close=True, output_dir=self.current_run_dir)

    def _plot_nested_pressure_distributions(self):
        """
        Additional descriptive figure separating the inner plausible-profile
        distribution from the outer distribution of PSQR P5 values.
        """
        res = self.results
        inner = clean_distribution_data(res['method1_all_profile_pressures'])
        outer = clean_distribution_data(res['burst_pressures_5th'][res['valid_mask']])

        fig, axes = plt.subplots(1, 2, figsize=(JOURNAL_DOUBLE_WIDTH, 3.0))

        ax = axes[0]
        ax.hist(inner, bins=45, density=True, color=COLORS['tertiary'],
                alpha=0.74, edgecolor='white', linewidth=0.4)
        ax.set_title('(a) Inner PSQR plausible-profile pressures', loc='left')
        ax.set_xlabel('Plausible-profile burst pressure (MPa)')
        ax.set_ylabel('Probability density')
        style_axis(ax)

        ax = axes[1]
        ax.hist(outer, bins=40, density=True, color=COLORS['primary'],
                alpha=0.74, edgecolor='white', linewidth=0.4)
        ax.axvline(res['nominal_5th_percentile'], color=COLORS['nominal'],
                   linestyle='--', linewidth=1.6, label='Nominal PSQR P5')
        ax.set_title('(b) Outer distribution of PSQR P5', loc='left')
        ax.set_xlabel('PSQR P5 burst pressure (MPa)')
        ax.set_ylabel('Probability density')
        ax.legend()
        style_axis(ax)

        fig.tight_layout()
        save_journal_figure(fig, 'psqr_nested_pressure_distributions', close=True, output_dir=self.current_run_dir)

    def create_comprehensive_plots(self):
        """Generate PSQR figures centered on P5, effective dimensions, and AAF."""
        if not self.results:
            print('No results. Run Monte Carlo first.')
            return
        res = self.results
        valid = res['valid_mask']
        pressures = res['burst_pressures_5th'][valid]
        d_avg = res['critical_d_avg_mm'][valid]
        L_comb = res['critical_L_comb_mm'][valid]
        d_avg_m1 = res['method1_all_profile_d_avg_mm']
        L_comb_m1 = res['method1_all_profile_L_comb_mm']
        aaf = res['aaf_values'][valid]

        fig1, axes1 = plt.subplots(2, 2, figsize=(JOURNAL_DOUBLE_WIDTH, 5.7), constrained_layout=True)
        ax = axes1[0, 0]
        ax.hist(pressures, bins=40, alpha=0.72, color=COLORS['primary'], density=True,
                edgecolor='white', linewidth=0.40)
        ax.axvline(res['nominal_5th_percentile'], color=COLORS['nominal'], linestyle='--', linewidth=1.5,
                   label=f'Nominal P5 = {res["nominal_5th_percentile"]:.2f} MPa')
        ax.set_xlabel('PSQR P5 burst pressure (MPa)'); ax.set_ylabel('Probability density')
        ax.set_title('(a) P5 burst-pressure distribution', loc='left'); ax.legend(fontsize=7.2); style_axis(ax)

        ax = axes1[0, 1]
        ax.scatter(d_avg, L_comb, color=COLORS['primary'], alpha=0.42, s=10, linewidths=0)
        ax.set_xlabel('P5-associated effective depth (mm)'); ax.set_ylabel('P5-associated effective length (mm)')
        ax.set_title('(b) P5-associated defect dimensions', loc='left'); style_axis(ax)

        ax = axes1[1, 0]
        sorted_p = np.sort(pressures); cdf = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
        ax.plot(sorted_p, cdf, color=COLORS['primary'], linewidth=1.7)
        ax.axvline(res['nominal_5th_percentile'], color=COLORS['nominal'], linestyle='--', linewidth=1.4,
                   label='Nominal P5')
        ax.set_xlabel('PSQR P5 burst pressure (MPa)'); ax.set_ylabel('Empirical cumulative probability')
        ax.set_title('(c) Empirical CDF of P5', loc='left'); ax.legend(fontsize=7.2); style_axis(ax)

        ax = axes1[1, 1]
        ax.hist(aaf, bins=32, density=True, alpha=0.72, color=COLORS['secondary'],
                edgecolor='white', linewidth=0.40)
        ax.axvline(1.0, color=COLORS['failure'], linestyle='--', linewidth=1.4, label='AAF = 1')
        ax.set_xlabel('Axial Alignment Factor (AAF)'); ax.set_ylabel('Probability density')
        ax.set_title('(d) AAF distribution', loc='left')
        ax.legend(fontsize=7.2, title=f'RSTRENG fallback: {res["rstreng_fallback_count"]:,}')
        style_axis(ax)
        save_journal_figure(fig1, 'psqr_montecarlo_results', close=True, output_dir=self.current_run_dir)

        fig2, axes2 = plt.subplots(2, 2, figsize=(JOURNAL_DOUBLE_WIDTH, 5.7), constrained_layout=True)
        datasets = [
            (axes2[0, 0], d_avg_m1, res['method1_mean_d_avg_mm'], COLORS['primary'],
             '(a) All plausible profiles: effective depth', 'Effective average depth (mm)'),
            (axes2[0, 1], L_comb_m1, res['method1_mean_L_comb_mm'], COLORS['secondary'],
             '(b) All plausible profiles: effective length', 'Effective length (mm)'),
            (axes2[1, 0], d_avg, res['method2_mean_d_avg_mm'], COLORS['primary'],
             '(c) P5-associated profile: effective depth', 'Effective average depth (mm)'),
            (axes2[1, 1], L_comb, res['method2_mean_L_comb_mm'], COLORS['secondary'],
             '(d) P5-associated profile: effective length', 'Effective length (mm)'),
        ]
        for ax, data, mean_value, color, title, xlabel in datasets:
            ax.hist(data, bins=32, density=True, alpha=0.72, color=color, edgecolor='white', linewidth=0.40)
            ax.axvline(mean_value, color=COLORS['failure'], linestyle='--', linewidth=1.45,
                       label=f'Mean = {mean_value:.2f} mm')
            ax.set_xlabel(xlabel); ax.set_ylabel('Probability density'); ax.set_title(title, loc='left')
            ax.legend(fontsize=7.2); style_axis(ax)
        save_journal_figure(fig2, 'psqr_dimension_distributions', close=True, output_dir=self.current_run_dir)

        self._plot_start_probability_map()
        self._plot_geometry_uncertainty_diagnostics()
        self._plot_nested_pressure_distributions()

    def run_convergence_analysis(self, sample_sizes=None):
        """Convergence with respect to the number of outer Monte Carlo realizations."""
        if sample_sizes is None:
            sample_sizes = [100, 500, 1000]
        conv = {'sample_sizes': [], 'mean': [], 'std': [], 'method1_mean_d_avg_mm': [],
                'method1_mean_L_comb_mm': [], 'method2_mean_d_avg_mm': [],
                'method2_mean_L_comb_mm': [], 'AAF_mean': [],
                'RSTRENG_fallback_fraction': [], 'time': []}
        print('\nRunning PSQR outer-MC convergence analysis...')
        for n in sample_sizes:
            start = time.time()
            res = self.run_monte_carlo(num_samples=n, generate_plots=False, fit_distributions=False)
            elapsed = time.time() - start
            conv['sample_sizes'].append(int(n)); conv['mean'].append(res['mean_5th_percentile'])
            conv['std'].append(res['std_5th_percentile'])
            conv['method1_mean_d_avg_mm'].append(res['method1_mean_d_avg_mm'])
            conv['method1_mean_L_comb_mm'].append(res['method1_mean_L_comb_mm'])
            conv['method2_mean_d_avg_mm'].append(res['method2_mean_d_avg_mm'])
            conv['method2_mean_L_comb_mm'].append(res['method2_mean_L_comb_mm'])
            conv['AAF_mean'].append(res['mean_AAF'])
            conv['RSTRENG_fallback_fraction'].append(res['rstreng_fallback_fraction'])
            conv['time'].append(elapsed)
            print(f"{n:8d} samples | mean P5={res['mean_5th_percentile']:.4f} MPa | "
                  f"SD P5={res['std_5th_percentile']:.4f} MPa | mean AAF={res['mean_AAF']:.5f} | "
                  f"RSTRENG fallback={100*res['rstreng_fallback_fraction']:.2f}% | time={elapsed:.1f} s")
        self.convergence_results = conv
        out = Path(self.current_run_dir) if self.current_run_dir is not None else Path('.')
        pd.DataFrame(conv).to_csv(out / 'psqr_outer_mc_convergence.csv', index=False)
        fig, axes = plt.subplots(2, 2, figsize=(JOURNAL_DOUBLE_WIDTH, 5.6), constrained_layout=True)
        x = np.asarray(conv['sample_sizes'])
        axes[0,0].semilogx(x, conv['mean'], 's-', color=COLORS['secondary']); axes[0,0].set_xlabel('Outer Monte Carlo realizations, $N_{MC}$'); axes[0,0].set_ylabel('Mean PSQR $P_5$ (MPa)'); axes[0,0].set_title('(a) Mean $P_5$ convergence', loc='left'); style_axis(axes[0,0])
        axes[0,1].semilogx(x, conv['std'], '^-', color=COLORS['map']); axes[0,1].set_xlabel('Outer Monte Carlo realizations, $N_{MC}$'); axes[0,1].set_ylabel('SD of PSQR $P_5$ (MPa)'); axes[0,1].set_title('(b) $P_5$ standard deviation', loc='left'); style_axis(axes[0,1])
        axes[1,0].semilogx(x, conv['method1_mean_d_avg_mm'], 'o-', color=COLORS['primary'], label='Ensemble-based'); axes[1,0].semilogx(x, conv['method2_mean_d_avg_mm'], 's--', color=COLORS['secondary'], label='$P_5$-associated'); axes[1,0].set_xlabel('Outer Monte Carlo realizations, $N_{MC}$'); axes[1,0].set_ylabel('Mean effective depth (mm)'); axes[1,0].set_title('(c) Effective-depth convergence', loc='left'); axes[1,0].legend(); style_axis(axes[1,0])
        axes[1,1].semilogx(x, conv['method1_mean_L_comb_mm'], 'o-', color=COLORS['primary'], label='Ensemble-based'); axes[1,1].semilogx(x, conv['method2_mean_L_comb_mm'], 's--', color=COLORS['secondary'], label='$P_5$-associated'); axes[1,1].set_xlabel('Outer Monte Carlo realizations, $N_{MC}$'); axes[1,1].set_ylabel('Mean effective length (mm)'); axes[1,1].set_title('(d) Effective-length convergence', loc='left'); axes[1,1].legend(); style_axis(axes[1,1])
        save_journal_figure(fig, 'psqr_outer_mc_convergence_4panel', close=True, output_dir=out)
        return conv

    def run_profile_count_convergence_analysis(self, profile_counts=None, fixed_num_samples=500):
        """Convergence with respect to the inner PSQR plausible-profile count."""
        if profile_counts is None:
            profile_counts = [50, 100, 200, 500, 1000]
        profile_counts = [int(x) for x in profile_counts if int(x) > 0]
        if not profile_counts:
            raise ValueError('profile_counts must contain at least one positive integer.')
        print('\nRunning PSQR inner-profile-count convergence analysis...')
        print(f'Fixed outer Monte Carlo samples: {fixed_num_samples:,}')
        original_profile_count = self.psqr_profiles_per_sample
        conv = {'profile_counts': [], 'fixed_num_samples': [], 'mean': [], 'std': [],
                'method1_mean_d_avg_mm': [], 'method1_mean_L_comb_mm': [],
                'method2_mean_d_avg_mm': [], 'method2_mean_L_comb_mm': [],
                'AAF_mean': [], 'RSTRENG_fallback_fraction': [], 'time': []}
        try:
            for pcount in profile_counts:
                self.psqr_profiles_per_sample = pcount
                start = time.time(); res = self.run_monte_carlo(num_samples=fixed_num_samples, generate_plots=False, fit_distributions=False); elapsed = time.time() - start
                conv['profile_counts'].append(pcount); conv['fixed_num_samples'].append(int(fixed_num_samples))
                conv['mean'].append(res['mean_5th_percentile']); conv['std'].append(res['std_5th_percentile'])
                conv['method1_mean_d_avg_mm'].append(res['method1_mean_d_avg_mm']); conv['method1_mean_L_comb_mm'].append(res['method1_mean_L_comb_mm'])
                conv['method2_mean_d_avg_mm'].append(res['method2_mean_d_avg_mm']); conv['method2_mean_L_comb_mm'].append(res['method2_mean_L_comb_mm'])
                conv['AAF_mean'].append(res['mean_AAF']); conv['RSTRENG_fallback_fraction'].append(res['rstreng_fallback_fraction']); conv['time'].append(elapsed)
                print(f"{pcount:8d} profiles | mean P5={res['mean_5th_percentile']:.4f} MPa | SD P5={res['std_5th_percentile']:.4f} MPa | mean AAF={res['mean_AAF']:.5f} | RSTRENG fallback={100*res['rstreng_fallback_fraction']:.2f}% | time={elapsed:.1f} s")
        finally:
            self.psqr_profiles_per_sample = original_profile_count
        self.profile_convergence_results = conv
        out = Path(self.current_run_dir) if self.current_run_dir is not None else Path('.')
        pd.DataFrame(conv).to_csv(out / 'psqr_profile_count_convergence.csv', index=False)
        fig, axes = plt.subplots(2,2,figsize=(JOURNAL_DOUBLE_WIDTH,5.6),constrained_layout=True); x=np.asarray(conv['profile_counts'])
        axes[0,0].semilogx(x,conv['mean'],'s-',color=COLORS['secondary']); axes[0,0].set_xlabel('Plausible profiles, $N_p$'); axes[0,0].set_ylabel('Mean PSQR $P_5$ (MPa)'); axes[0,0].set_title('(a) Mean $P_5$ convergence',loc='left'); style_axis(axes[0,0])
        axes[0,1].semilogx(x,conv['std'],'^-',color=COLORS['map']); axes[0,1].set_xlabel('Plausible profiles, $N_p$'); axes[0,1].set_ylabel('SD of PSQR $P_5$ (MPa)'); axes[0,1].set_title('(b) $P_5$ standard deviation',loc='left'); style_axis(axes[0,1])
        axes[1,0].semilogx(x,conv['method1_mean_d_avg_mm'],'o-',color=COLORS['primary'],label='Ensemble-based'); axes[1,0].semilogx(x,conv['method2_mean_d_avg_mm'],'s--',color=COLORS['secondary'],label='$P_5$-associated'); axes[1,0].set_xlabel('Plausible profiles, $N_p$'); axes[1,0].set_ylabel('Mean effective depth (mm)'); axes[1,0].set_title('(c) Effective-depth convergence',loc='left'); axes[1,0].legend(); style_axis(axes[1,0])
        axes[1,1].semilogx(x,conv['method1_mean_L_comb_mm'],'o-',color=COLORS['primary'],label='Ensemble-based'); axes[1,1].semilogx(x,conv['method2_mean_L_comb_mm'],'s--',color=COLORS['secondary'],label='$P_5$-associated'); axes[1,1].set_xlabel('Plausible profiles, $N_p$'); axes[1,1].set_ylabel('Mean effective length (mm)'); axes[1,1].set_title('(d) Effective-length convergence',loc='left'); axes[1,1].legend(); style_axis(axes[1,1])
        save_journal_figure(fig,'psqr_profile_count_convergence_4panel',close=True,output_dir=out)
        return conv

    def print_detailed_results(self):
        """Print paper-oriented PSQR P5/effective-dimension and AAF diagnostics."""
        if not self.results:
            print('No results. Run Monte Carlo first.'); return
        res=self.results
        print('\n'+'='*80); print('PSQR NESTED ILI-UNCERTAINTY RESULTS'); print('='*80)
        print(f"Algorithm version: {res.get('algorithm_version','n/a')}")
        print(f"Outer Monte Carlo samples: {res['samples_used']:,}")
        print(f"Plausible profiles per outer sample: {self.psqr_profiles_per_sample:,}")
        print('\nPSQR P5:'); print(f"  Nominal PSQR P5: {res['nominal_5th_percentile']:.3f} MPa"); print(f"  Mean PSQR P5: {res['mean_5th_percentile']:.3f} MPa"); print(f"  SD PSQR P5: {res['std_5th_percentile']:.3f} MPa")
        print('\nAXIAL ALIGNMENT FACTOR (AAF):'); print(f"  Mean AAF: {res['mean_AAF']:.6f}"); print(f"  Min/Max AAF: {res['min_AAF']:.6f} / {res['max_AAF']:.6f}"); print(f"  Whole-anomaly RSTRENG fallback: {res['rstreng_fallback_count']:,}/{np.sum(res['valid_mask']):,} ({100*res['rstreng_fallback_fraction']:.3f}%)")
        nominal=res.get('nominal_AAF_diagnostics');
        if nominal:
            print(f"  Nominal AAF: {nominal['aaf']:.6f}"); print(f"  Nominal selected assessment method: {nominal['assessment_method']}"); print(f"  AAF 6t target/actual strip width: {nominal['target_strip_width_mm']:.3f}/{nominal['actual_strip_width_mm']:.3f} mm")
        print('\nMETHOD 1 -- ALL PLAUSIBLE PROFILES:'); print(f"  Total profiles: {res['method1_total_profiles']:,}"); print(f"  Mean d_avg: {res['method1_mean_d_avg_mm']:.3f} mm"); print(f"  SD d_avg: {res['method1_std_d_avg_mm']:.3f} mm"); print(f"  Mean L_comb: {res['method1_mean_L_comb_mm']:.3f} mm"); print(f"  SD L_comb: {res['method1_std_L_comb_mm']:.3f} mm")
        print('\nMETHOD 2 -- P5-ASSOCIATED REPRESENTATIVE PROFILE:'); print(f"  Number of values: {len(res['method2_d_avg_at_5th_mm']):,}"); print(f"  Mean d_avg: {res['method2_mean_d_avg_mm']:.3f} mm"); print(f"  SD d_avg: {res['method2_std_d_avg_mm']:.3f} mm"); print(f"  Mean L_comb: {res['method2_mean_L_comb_mm']:.3f} mm"); print(f"  SD L_comb: {res['method2_std_L_comb_mm']:.3f} mm")

# ============================================================================
# USER INPUT HELPER FUNCTIONS
# ============================================================================

def parse_int_list_from_user(prompt, default_values):
    """
    Ask the user for comma-separated positive integers.
    If the user presses Enter, default_values are used.
    """
    default_text = ", ".join(str(v) for v in default_values)
    raw = input(f"{prompt} [default: {default_text}]: ").strip()

    if raw == "":
        return list(default_values)

    try:
        values = [int(x.strip()) for x in raw.split(',') if x.strip()]
        values = [v for v in values if v > 0]
        if len(values) == 0:
            print("No valid positive numbers entered. Using default values.")
            return list(default_values)
        return values
    except Exception:
        print("Invalid input. Using default values.")
        return list(default_values)


def ask_positive_int(prompt, default_value):
    """
    Ask the user for one positive integer.
    If the user presses Enter, default_value is used.
    """
    raw = input(f"{prompt} [default: {default_value}]: ").strip()

    if raw == "":
        return int(default_value)

    try:
        value = int(raw)
        if value <= 0:
            print("Value must be positive. Using default value.")
            return int(default_value)
        return value
    except Exception:
        print("Invalid input. Using default value.")
        return int(default_value)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Required/recommended for safe multiprocessing on Windows/Anaconda.
    mp.freeze_support()

    MATERIAL = {
        'diameter': 609.6,      # mm
        'thickness': 6.68,      # mm
        'yield_strength': 483   # MPa
    }

    OPERATING_PRESSURE = 8.450  # MPa

    print("MONTE CARLO ANALYSIS FOR CORROSION ASSESSMENT")
    print("=" * 60)
    print("Select model:")
    print("1 - RSTRENG (river bottom profile)")
    print("2 - PSQR (probabilistic profile generation)")
    choice = input("Enter 1 or 2: ").strip()

    print("\nSelect analysis type:")
    print("a - Convergence analysis")
    print("b - Single analysis with plots")
    analysis_type = input("Enter a or b: ").strip().lower()

    if choice == '1':
        analyzer = RSTRENGMonteCarlo(
            excel_path="example1.xlsx",
            material_params=MATERIAL,
            operating_pressure=OPERATING_PRESSURE,
            depth_absolute_error_pct_t=0.07,
            length_absolute_error_mm=7.0,
            confidence_level=0.80,
            nominal_cell_length=5
        )

        if analysis_type == 'a':
            print("\nRSTRENG convergence analysis selected.")
            sample_sizes = parse_int_list_from_user(
                "Enter Monte Carlo sample sizes separated by commas",
                [100, 1000]
            )
            analyzer.run_convergence_analysis(sample_sizes=sample_sizes)
        else:
            print("\nRunning single RSTRENG analysis with plots...")
            num_samples = ask_positive_int(
                "Enter number of Monte Carlo samples",
                1000
            )
            analyzer.run_monte_carlo(num_samples=num_samples, generate_plots=True, fit_distributions=True)
            analyzer.print_detailed_results()

    elif choice == '2':
        default_profiles = 200

        if analysis_type == 'a':
            print("\nPSQR convergence analysis selected.")
            print("Choose PSQR convergence type:")
            print("1 - MC sample-size convergence: fixed number of profiles, changing number of MC samples")
            print("2 - Profile-count convergence: fixed number of MC samples, changing number of PSQR profiles")
            convergence_choice = input("Enter 1 or 2: ").strip()

            if convergence_choice == '1':
                fixed_profiles = ask_positive_int(
                    "Enter fixed number of PSQR profiles per MC sample",
                    default_profiles
                )

                analyzer = PSQRMonteCarlo(
                    excel_path="example1.xlsx",
                    material_params=MATERIAL,
                    depth_absolute_error_pct_t=0.07,
                    length_absolute_error_mm=7.0,
                    width_absolute_error_mm=9.0,
                    confidence_level=0.80,
                    nominal_cell_length=5,
                    nominal_cell_width=5,
                    psqr_profiles_per_sample=fixed_profiles
                )

                sample_sizes = parse_int_list_from_user(
                    "Enter Monte Carlo sample sizes separated by commas",
                    [100, 500, 1000]
                )

                analyzer.run_convergence_analysis(sample_sizes=sample_sizes)

            elif convergence_choice == '2':
                fixed_samples = ask_positive_int(
                    "Enter fixed number of Monte Carlo samples",
                    500
                )

                profile_counts = parse_int_list_from_user(
                    "Enter PSQR profile counts separated by commas",
                    [50, 100, 200, 500, 1000]
                )

                analyzer = PSQRMonteCarlo(
                    excel_path="example1.xlsx",
                    material_params=MATERIAL,
                    depth_absolute_error_pct_t=0.07,
                    length_absolute_error_mm=7.0,
                    width_absolute_error_mm=9.0,
                    confidence_level=0.80,
                    nominal_cell_length=5,
                    nominal_cell_width=5,
                    psqr_profiles_per_sample=profile_counts[0]
                )

                analyzer.run_profile_count_convergence_analysis(
                    profile_counts=profile_counts,
                    fixed_num_samples=fixed_samples
                )

            else:
                print("Invalid PSQR convergence choice. Exiting.")

        else:
            print("\nRunning single PSQR analysis with plots...")
            num_samples = ask_positive_int(
                "Enter number of Monte Carlo samples",
                500
            )
            profiles_per_sample = ask_positive_int(
                "Enter number of PSQR profiles per MC sample",
                default_profiles
            )

            analyzer = PSQRMonteCarlo(
                excel_path="example1.xlsx",
                material_params=MATERIAL,
                depth_absolute_error_pct_t=0.07,
                length_absolute_error_mm=7.0,
                width_absolute_error_mm=9.0,
                confidence_level=0.80,
                nominal_cell_length=5,
                nominal_cell_width=5,
                psqr_profiles_per_sample=profiles_per_sample
            )

            analyzer.run_monte_carlo(num_samples=num_samples, generate_plots=True, fit_distributions=True)
            analyzer.print_detailed_results()

    else:
        print("Invalid choice. Exiting.")
