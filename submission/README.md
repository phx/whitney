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

## Authors
- James Robert Austin
- Keatron Leviticus Evans 