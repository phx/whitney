# Error Analysis Documentation

## Overview
This document describes the error analysis methodology used in the fractal field theory framework.

## 1. Coupling Evolution Uncertainties

### Sources of Uncertainty
1. **Statistical Uncertainties**
   - U(1) coupling: ±0.00020 at M_Z
   - SU(2) coupling: ±0.00035 at M_Z
   - SU(3) coupling: ±0.00110 at M_Z

2. **Scale Uncertainties**
   - Estimated by varying scale μ by factor of 2
   - Computed as: |g(2E) - g(E/2)|/2

3. **Truncation Uncertainties**
   - From series expansion in α
   - Estimated as α⁴|g| (fourth order)

### Combination Method
- Uncertainties combined in quadrature
- Total uncertainty = √(σ²_stat + σ²_scale + σ²_trunc)

## 2. Basis Function Uncertainties

### Sources of Uncertainty
1. **Normalization Errors**
   - Deviation from orthonormality condition
   - Measured via: |⟨ψₙ|ψₙ⟩ - 1|

2. **Truncation Errors**
   - From series expansion in basis functions
   - Estimated as: α^(n+1)|F(x)|

3. **Integration Errors**
   - From numerical quadrature
   - Conservative estimate: 10⁻⁸|ψ(x)|

### Error Propagation
- Total error computed as sum of individual contributions
- Conservative approach used for systematic effects 