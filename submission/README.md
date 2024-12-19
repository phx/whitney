# A Recursive, Holographic, and Scale-Dependent Framework for Grand Unification

## Submission Package

This package contains:
- main.tex: The main paper with complete mathematical framework
- figures/: Visualizations and diagrams
- supplementary/: Code implementation and numerical results

## Requirements
- LaTeX with AMS packages
- Python 3.8+ with numpy, sympy, matplotlib
- Arbitrary precision support via mpmath

## Compilation
```bash
cd submission
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Code Verification
```bash
cd supplementary/code
python -m pytest test_*.py
```

All theoretical predictions have been computationally validated:
- Coupling unification at predicted GUT scale
- Holographic entropy bounds
- Fractal recursion relations
- Dark matter density profiles
- Ward identities and unitarity

See test_field.py for detailed numerical validations.

## Authors
- James Robert Austin
- Keatron Leviticus Evans 

## Documentation

- [Installation Guide](supplementary/code/docs/installation.md)
- [API Reference](supplementary/code/docs/api.md)
- [Scientific Documentation](supplementary/code/docs/scientific.md)
- [Testing Guide](supplementary/code/docs/testing_guide.md)