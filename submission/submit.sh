#!/bin/bash

# Generate figures
cd supplementary/code
python generate_figures.py
python generate_data.py

# Compile paper
cd ../..
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Create submission tarball
tar czf submission.tar.gz main.tex figures/ supplementary/ manifest.yml README.md 