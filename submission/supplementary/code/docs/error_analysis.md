# Error Analysis Guide

This document describes the error analysis and uncertainty quantification framework.

## 1. Statistical Uncertainties

### 1.1 Basic Error Propagation

For a function $f(x_1, ..., x_n)$ of multiple variables, the statistical uncertainty is calculated using:

```python
from core.errors import propagate_errors
from core.types import ErrorEstimate

def analyze_measurement(values, uncertainties):
    """
    Analyze measurement with error propagation.
    
    Args:
        values: Dictionary of measured values
        uncertainties: Dictionary of uncertainties
    
    Returns:
        ErrorEstimate: Result with propagated uncertainties
    """
    result = propagate_errors(
        function=compute_observable,
        values=values,
        uncertainties=uncertainties
    )
    return result
```

### 1.2 Covariance Matrix Handling

For correlated measurements:

```python
def analyze_correlated_data(data, covariance_matrix):
    """Handle correlated measurements."""
    return ErrorEstimate(
        value=compute_central_value(data),
        statistical=compute_stat_error(data, covariance_matrix),
        systematic=compute_syst_error(data),
        correlations=extract_correlations(covariance_matrix)
    )
```

## 2. Systematic Uncertainties

### 2.1 Sources of Systematics

1. **Detector Effects**
   - Resolution smearing
   - Efficiency corrections
   - Acceptance uncertainties

2. **Theory Uncertainties**
   - Scale variations
   - PDF uncertainties
   - Model dependencies

### 2.2 Systematic Error Estimation

```python
def estimate_systematics(measurement, variations):
    """
    Estimate systematic uncertainties.
    
    Args:
        measurement: Nominal measurement
        variations: List of systematic variations
    
    Returns:
        dict: Systematic uncertainties by source
    """
    systematics = {}
    for source, variation in variations.items():
        systematics[source] = compute_variation_impact(
            nominal=measurement,
            variation=variation
        )
    return systematics
```

## 3. Combined Uncertainty Analysis

### 3.1 Total Uncertainty Calculation

```python
def compute_total_uncertainty(statistical, systematic, correlation=0.0):
    """
    Compute total uncertainty including correlations.
    
    Args:
        statistical: Statistical uncertainty
        systematic: Systematic uncertainty
        correlation: Correlation coefficient
    
    Returns:
        float: Total uncertainty
    """
    return np.sqrt(
        statistical**2 + 
        systematic**2 + 
        2 * correlation * statistical * systematic
    )
```

### 3.2 Error Reporting

```python
def format_measurement(value, error):
    """
    Format measurement with uncertainty.
    
    Args:
        value: Measured value
        error: ErrorEstimate object
    
    Returns:
        str: Formatted result
    """
    return f"{value:.2f} ± {error.statistical:.2f} (stat) ± {error.systematic:.2f} (syst)"
```

## 4. Validation Methods

### 4.1 Cross-Validation

```python
def cross_validate(data, estimator, n_folds=5):
    """
    Perform cross-validation.
    
    Args:
        data: Input data
        estimator: Estimation function
        n_folds: Number of validation folds
    
    Returns:
        dict: Validation metrics
    """
    return {
        'bias': compute_bias(data, estimator, n_folds),
        'variance': compute_variance(data, estimator, n_folds),
        'stability': assess_stability(data, estimator, n_folds)
    }
```

### 4.2 Consistency Checks

```python
def check_consistency(results, reference, tolerance=0.05):
    """
    Check consistency with reference values.
    
    Args:
        results: Measured results
        reference: Reference values
        tolerance: Acceptance threshold
    
    Returns:
        bool: True if consistent
    """
    return all(
        abs(r - ref) < tolerance * ref
        for r, ref in zip(results, reference)
    )
```

## 5. Best Practices

1. **Documentation**
   - Document all uncertainty sources
   - Explain correlation assumptions
   - Justify systematic variations

2. **Validation**
   - Cross-check with independent methods
   - Verify error propagation
   - Test edge cases

3. **Reporting**
   - Include all uncertainty components
   - Specify correlation assumptions
   - Provide detailed breakdown

## 6. Common Pitfalls

1. **Statistical Limitations**
   - Small sample effects
   - Bias in uncertainty estimation
   - Correlation mismodeling

2. **Systematic Effects**
   - Double-counting correlations
   - Underestimating systematics
   - Missing uncertainty sources

3. **Technical Issues**
   - Numerical instabilities
   - Precision limitations
   - Implementation errors

# Error Propagation System

## Overview

The framework implements a comprehensive error propagation system that handles both statistical and systematic uncertainties in physics calculations.

## Uncertainty Types

### Statistical Uncertainties

Statistical uncertainties are propagated using standard error propagation rules:

1. For addition/subtraction (z = x ± y):
```python
σz = √(σx² + σy²)
```

2. For multiplication/division (z = x * y or z = x/y):
```python
(σz/z)² = (σx/x)² + (σy/y)²
```

### Systematic Uncertainties

Systematic uncertainties are tracked separately and categorized by source:

```python
class Measurement:
    value: float
    statistical_uncertainty: float
    systematics: Dict[str, float]  # By source
```

Common systematic sources include:
- Calibration uncertainties
- Resolution effects
- Acceptance corrections
- Background estimation
- Theoretical uncertainties

## Correlation Handling

The framework supports correlated uncertainties through:

1. Correlation matrices for statistical uncertainties:
```python
def propagate_with_correlation(values, uncertainties, correlation_matrix):
    """Propagate uncertainties considering correlations."""
    return np.sqrt(uncertainties @ correlation_matrix @ uncertainties)
```

2. Common systematic uncertainties:
```python
def combine_systematics(measurements):
    """Combine measurements with common systematics."""
    common_sources = set.intersection(*[set(m.systematics.keys()) 
                                     for m in measurements])
    # Correlations handled by source
    return {source: sum(m.systematics[source] for m in measurements)
            for source in common_sources}
```

## Usage Examples

1. Basic uncertainty propagation:
```python
# Energy measurement with uncertainty
e1 = Energy(100.0, 5.0)  # 100 ± 5 GeV
e2 = Energy(50.0, 2.0)   # 50 ± 2 GeV

# Automatic propagation in arithmetic
total = e1 + e2  # Uncertainties added in quadrature
print(f"Total: {total.value:.1f} ± {total.uncertainty:.1f} GeV")
```

2. Including systematic uncertainties:
```python
measurement = detector.simulate_measurement(
    energy,
    include_systematics=True
)

# Access different uncertainty components
stat_unc = measurement.uncertainty
syst_unc = np.sqrt(sum(s**2 for s in measurement.systematics.values()))
total_unc = np.sqrt(stat_unc**2 + syst_unc**2)
```

## Best Practices

1. Always specify uncertainty sources clearly
2. Track systematic uncertainties separately by source
3. Document correlation assumptions
4. Use proper significant figures in reporting
5. Validate uncertainty propagation with test cases

## Implementation Details

The error propagation system is implemented across several modules:

- `core/types.py`: Base classes for physical quantities with uncertainties
- `core/errors.py`: Error handling and validation
- `core/precision.py`: Numerical precision utilities
- `core/stability.py`: Numerical stability checks

See the API documentation for detailed interface descriptions.