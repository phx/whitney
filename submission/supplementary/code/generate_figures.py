#!/usr/bin/env python3
"""
Generate figures for the paper submission.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from core.field import UnifiedField
from core.constants import (
    ALPHA_VAL, Z_MASS, X, T, E,
    FIGURE_SIZE, PLOT_DPI, PLOT_STYLE,
    EXPERIMENTAL_DATA,
    g1_REF, g2_REF, g3_REF,
    E_GUT, E_PLANCK,
    SIN2_THETA_W
)
import os
import pandas as pd
import seaborn as sns
from scipy import stats
from core.errors import ValidationError

# Apply plot style settings
plt.rcParams.update(PLOT_STYLE)

def generate_field_space(save_path: Optional[str] = '../figures/field_space.pdf') -> None:
    """Generate field configuration space visualization"""
    if not save_path:
        raise ValueError("save_path must be specified")
    field = UnifiedField(alpha=ALPHA_VAL)
    
    # Create field configuration space plot
    E = np.linspace(0, 1000, 100)  # Energy scale in GeV
    # Convert symbolic expressions to numerical values
    psi = [complex(field.compute_basis_function(n=1, E=e).subs([(X, 0), (T, 0)])) for e in E]
    
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(E, np.real(psi), label='Re(Ψ)')
    plt.plot(E, np.imag(psi), label='Im(Ψ)')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Field Amplitude')
    plt.title('Fractal Field Configuration Space')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def generate_energy_scales(save_path: Optional[str] = '../figures/energy_scales.pdf') -> None:
    """Generate energy scale evolution plot"""
    if not save_path:
        raise ValueError("save_path must be specified")
    field = UnifiedField(alpha=ALPHA_VAL)
    
    # Plot coupling constant evolution
    E = np.logspace(2, 16, 100)  # From 100 GeV to 10^16 GeV
    # Convert symbolic expressions to numerical values
    g1 = [float(field.compute_coupling(1, e)) for e in E]
    g2 = [float(field.compute_coupling(2, e)) for e in E]
    g3 = [float(field.compute_coupling(3, e)) for e in E]
    
    plt.figure(figsize=FIGURE_SIZE)
    plt.semilogx(E, g1, label='g₁')
    plt.semilogx(E, g2, label='g₂')
    plt.semilogx(E, g3, label='g₃')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Coupling Strength')
    plt.title('Coupling Constant Evolution')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def generate_gauge_hierarchy(save_path: Optional[str] = '../figures/gauge_hierarchy.pdf') -> None:
    """Generate gauge group hierarchy diagram"""
    if not save_path:
        raise ValueError("save_path must be specified")
    plt.figure(figsize=FIGURE_SIZE)
    levels = ['U(1)', 'SU(2)', 'SU(3)', 'SU(5)', 'SO(10)', 'E8']
    y_pos = np.arange(len(levels))
    plt.barh(y_pos, np.arange(len(levels)) + 1)
    plt.yticks(y_pos, levels)
    plt.xlabel('Energy Scale (log GeV)')
    plt.title('Gauge Group Hierarchy')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def generate_holographic_info(save_path: Optional[str] = '../figures/holographic_info.pdf') -> None:
    """Generate holographic encoding visualization"""
    if not save_path:
        raise ValueError("save_path must be specified")
    plt.figure(figsize=FIGURE_SIZE)
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, 1, 50)
    R, Theta = np.meshgrid(r, theta)
    Z = R * np.exp(1j * Theta)
    plt.pcolormesh(R*np.cos(Theta), R*np.sin(Theta), np.abs(Z))
    plt.title('Holographic Information Encoding')
    plt.axis('equal')
    plt.colorbar(label='Information Density')
    plt.savefig(save_path)
    plt.close()

def generate_pull_plot(save_path: Optional[str] = '../figures/pull_distribution.pdf') -> None:
    """Generate pull distribution plot comparing predictions with experiments."""
    if not save_path:
        raise ValueError("save_path must be specified")
    
    # Load statistical analysis
    stats_df = pd.read_csv('../data/statistical_analysis.csv')
    
    plt.figure(figsize=FIGURE_SIZE)
    
    # Plot pull values with error bars
    y_pos = np.arange(len(stats_df['Observable']))
    plt.errorbar(stats_df['Pull_Value'], y_pos, xerr=1.0, fmt='o', 
                capsize=5, color='blue', label='Measurement')
    
    # Add reference lines
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=-2, color='r', linestyle=':', alpha=0.3)
    plt.axvline(x=2, color='r', linestyle=':', alpha=0.3)
    
    plt.yticks(y_pos, stats_df['Observable'])
    plt.xlabel('Pull Value (σ)')
    plt.title('Pull Distribution of Theoretical Predictions')
    plt.grid(True, alpha=0.3)
    
    # Add chi-square information
    chi2_text = f"χ²/DoF = {stats_df['Chi_Square'].iloc[0]:.1f}/{stats_df['DoF'].iloc[0]}"
    p_value_text = f"p-value = {stats_df['P_Value'].iloc[0]:.3f}"
    plt.text(0.95, 0.05, chi2_text + '\n' + p_value_text,
            transform=plt.gca().transAxes, ha='right', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_correlation_matrix(save_path: Optional[str] = '../figures/correlation_matrix.pdf') -> None:
    """Generate correlation matrix heatmap."""
    if not save_path:
        raise ValueError("save_path must be specified")
    
    # Load predictions
    pred_df = pd.read_csv('../data/predictions.csv')
    
    # Calculate correlation matrix
    corr_matrix = pred_df.pivot(columns='Correlation_Group', 
                               values=['Value', 'Total_Uncertainty']).corr()
    
    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                vmin=-1, vmax=1, fmt='.2f')
    
    plt.title('Observable Correlation Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_residuals_plot(save_path: Optional[str] = '../figures/residuals.pdf') -> None:
    """
    Generate residuals plot showing deviations from experimental data.
    
    This plot shows:
    1. Raw residuals (prediction - experiment)
    2. Normalized residuals (pulls)
    3. Error bands for statistical and systematic uncertainties
    """
    if not save_path:
        raise ValueError("save_path must be specified")
    
    # Load predictions and experimental data
    pred_df = pd.read_csv('../data/predictions.csv')
    stats_df = pd.read_csv('../data/statistical_analysis.csv')
    
    plt.figure(figsize=(12, 8))
    
    # Create two subplots
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    # Upper plot: Predictions vs Experimental Values
    for idx, row in stats_df.iterrows():
        obs = row['Observable']
        if obs in EXPERIMENTAL_DATA:  # Only plot observables with experimental data
            pred = pred_df[pred_df['Observable'] == obs]['Value'].values[0]
            exp_val, exp_err = EXPERIMENTAL_DATA[obs]
            
            # Plot experimental point with error bars
            ax1.errorbar(exp_val, idx, xerr=exp_err, fmt='o', color='black',
                        label='Experiment' if idx == 0 else '')
            
            # Plot prediction with error band
            pred_err = pred_df[pred_df['Observable'] == obs]['Total_Uncertainty'].values[0]
            ax1.errorbar(pred, idx, xerr=pred_err, fmt='s', color='blue',
                        label='Theory' if idx == 0 else '')
    
    ax1.set_yticks(range(len(EXPERIMENTAL_DATA)))
    ax1.set_yticklabels(list(EXPERIMENTAL_DATA.keys()))
    ax1.set_title('Comparison with Experimental Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Lower plot: Pull Distribution
    ax2.hist(stats_df['Pull_Value'], bins=20, density=True, alpha=0.6,
             color='blue', label='Pulls')
    
    # Add reference Gaussian
    x = np.linspace(-4, 4, 100)
    ax2.plot(x, stats.norm.pdf(x, 0, 1), 'r--', 
             label='Standard Normal')
    
    ax2.set_xlabel('Pull Value (σ)')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs('../figures', exist_ok=True)
    
    # Generate all figures with default paths
    for func in [generate_field_space, generate_energy_scales, 
                generate_gauge_hierarchy, generate_holographic_info,
                generate_pull_plot, generate_correlation_matrix,
                generate_residuals_plot]:
        try:
            func()
        except Exception as e:
            print(f"Error generating {func.__name__}: {e}") 