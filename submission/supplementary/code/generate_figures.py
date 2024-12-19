#!/usr/bin/env python3
"""Generate figures for the paper."""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from core.field import UnifiedField
from core.contexts import numeric_precision, field_config
from core.physics_constants import (
    ALPHA_VAL, Z_MASS, M_PLANCK,
    g1_REF, g2_REF, g3_REF
)
from core.types import FieldConfig

# Set up figures directory path
figures_dir = Path('../../../figures')  # Path from code/ to submission/figures/

def setup_plotting():
    """Set up matplotlib plotting style."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'figure.figsize': (8, 6),
        'text.usetex': False,  # Don't use LaTeX for text rendering
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica']
    })

def generate_coupling_evolution():
    """Generate gauge coupling evolution plot."""
    field = UnifiedField()
    # Energy range from M_Z to GUT scale
    E = np.logspace(np.log10(Z_MASS), 16, 1000)  # GeV
    g1 = [float(abs(field.compute_coupling(1, e))) for e in E]
    g2 = [float(abs(field.compute_coupling(2, e))) for e in E]
    g3 = [float(abs(field.compute_coupling(3, e))) for e in E]
    
    plt.figure()
    plt.plot(np.log10(E), g1, label='g1', linewidth=2)
    plt.plot(np.log10(E), g2, label='g2', linewidth=2)
    plt.plot(np.log10(E), g3, label='g3', linewidth=2)
    plt.xlabel('log10(E/GeV)')
    plt.ylabel('Gauge Couplings gi')
    plt.title('Gauge Coupling Unification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(figures_dir / 'coupling_unification.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

def generate_mass_spectrum():
    """Generate fermion mass spectrum plot."""
    field = UnifiedField()
    config = FieldConfig(
        mass=125.0,  # Higgs mass in GeV
        coupling=0.1,  # Standard coupling
        dimension=4,  # 4D spacetime
        max_level=10  # Sufficient for convergence
    )
    masses = field.compute_fermion_masses(config)
    
    particles = ['electron', 'muon', 'tau', 'up', 'charm', 'top', 'down', 'strange', 'bottom']
    mass_values = [np.log10(masses[p].value) for p in particles]
    
    plt.figure()
    colors = plt.cm.viridis(np.linspace(0, 1, len(particles)))
    plt.bar(range(len(particles)), mass_values, color=colors)
    plt.xticks(range(len(particles)), particles, rotation=45)
    plt.ylabel('$\\log_{10}(\\text{Mass}/\\text{GeV})$')
    plt.title('Fermion Mass Spectrum')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(figures_dir / 'mass_spectrum.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

def generate_dark_matter_profile():
    """Generate dark matter density profile plot."""
    field = UnifiedField()
    r = np.logspace(-2, 2, 1000)  # kpc
    rho = [field.compute_dark_matter_density(ri).value for ri in r]
    
    plt.figure()
    plt.loglog(r, rho, label='Fractal Model', linewidth=2)
    plt.xlabel('r (kpc)')
    plt.ylabel('rho(r) (GeV/cm³)')
    plt.title('Dark Matter Density Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(figures_dir / 'dark_matter_profile.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

def generate_neutrino_oscillations():
    """Generate neutrino oscillation probability plot."""
    field = UnifiedField()
    # Create config for neutrino mass calculations
    config = FieldConfig(
        mass=125.0,  # Higgs mass in GeV
        coupling=0.1,  # Standard coupling
        dimension=4,  # 4D spacetime
        max_level=10  # Sufficient for convergence
    )
    
    # T2K baseline and energy range
    L = 295.0  # km
    E = np.linspace(0.1, 2.0, 100)  # GeV
    
    P_mue = [field.compute_oscillation_probability(
        initial='muon',
        final='electron',
        L=L,
        E=e,
        config=config  # Pass config for mass calculations
    ).value for e in E]
    
    plt.figure()
    plt.plot(E, P_mue, linewidth=2)
    plt.xlabel('$E_\\nu$ (GeV)')
    plt.ylabel('$P(\\nu_\\mu \\to \\nu_e)$')
    plt.title('Neutrino Oscillation Probability')
    plt.grid(True, alpha=0.3)
    plt.savefig(str(figures_dir / 'neutrino_oscillations.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """Generate all figures."""
    # Create figures directory if it doesn't exist
    figures_dir = Path('../../../figures')  # Go up 3 levels from code/ to submission/
    figures_dir.mkdir(exist_ok=True)
    
    setup_plotting()
    
    print("Generating figures...")
    generate_coupling_evolution()
    generate_mass_spectrum()
    generate_dark_matter_profile()
    generate_neutrino_oscillations()
    print("Done generating figures.")

if __name__ == "__main__":
    main()