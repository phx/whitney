#!/bin/bash
"""
Submit paper to arXiv.
"""

set -euo pipefail

# Configuration
PAPER_DIR="submission"
MAIN_TEX="main.tex"
SUPPLEMENTARY_DIR="supplementary"

# Validate paper first
echo "Running validation checks..."
python3 validate_all.py
if [ $? -ne 0 ]; then
    echo "Validation failed! Fix errors before submitting."
    exit 1
fi

# Create submission tarball
echo "Creating submission package..."
cd "${PAPER_DIR}"

# Verify required files exist
required_files=(
    "${MAIN_TEX}"
    "bibliography.bib"
    "manifest.yml"
)

for file in "${required_files[@]}"; do
    if [ ! -f "${file}" ]; then
        echo "Error: Required file ${file} not found!"
        exit 1
    fi
done

# Create submission tarball
tar czf submission.tar.gz \
    "${MAIN_TEX}" \
    bibliography.bib \
    figures/* \
    "${SUPPLEMENTARY_DIR}" \
    manifest.yml

echo "Created submission.tar.gz"

# Optional: Upload to arXiv
if [ "${1:-}" = "--upload" ]; then
    echo "Uploading to arXiv..."
    # Add arXiv upload command here
    # arxiv-submit submission.tar.gz
fi

echo "Done!"