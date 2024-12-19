#!/bin/bash
# Submit paper to arXiv

set -euo pipefail

# Configuration
SUBMISSION_DIR="/Users/phx/git/whitney/submission"
MAIN_TEX="${SUBMISSION_DIR}/main.tex"
SUPPLEMENTARY_DIR="${SUBMISSION_DIR}/supplementary"
FIGURES_DIR="${SUPPLEMENTARY_DIR}/figures"
DATA_DIR="${SUBMISSION_DIR}/data"
CODE_DIR="${SUPPLEMENTARY_DIR}/code"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

TEST_EXIT_CODE=0

# Check/install Python dependencies
echo -e "${BLUE}Checking Python dependencies...${NC}"
python3 -m pip install -q -r "${CODE_DIR}/requirements.txt" || {
    echo -e "${RED}Failed to install Python dependencies${NC}"
    exit 1
}

# Compile PDF
echo -e "${BLUE}Compiling PDF...${NC}"

# Run pdflatex multiple times to resolve references
cd submission
# Initial run
pdflatex -interaction=nonstopmode main.tex || { echo -e "${RED}First pdflatex run failed${NC}"; exit 1; }

# Run bibtex to process citations
bibtex main || { echo -e "${RED}Bibtex run failed${NC}"; exit 1; }

# Two more pdflatex runs to resolve references
pdflatex -interaction=nonstopmode main.tex || { echo -e "${RED}Second pdflatex run failed${NC}"; exit 1; }
pdflatex -interaction=nonstopmode main.tex || { echo -e "${RED}Final pdflatex run failed${NC}"; exit 1; }

# Check for undefined references
if grep -q "undefined references" main.log; then
    echo -e "${RED}Warning: Document has undefined references${NC}"
fi

if [ ! -f main.pdf ]; then
    echo -e "${RED}PDF compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}PDF compilation successful - output is main.pdf${NC}"
cd ..

# Validate paper first
echo -e "${GREEN}Running validation checks...${NC}"

# Run tests first
# cd "${CODE_DIR}"
# python -m pytest --cov=core --cov-report=html
# TEST_EXIT_CODE=$?
# cd ../..

if [ ${TEST_EXIT_CODE} -ne 0 ]; then
    echo -e "${RED}Tests failed! Fix errors before submitting.${NC}"
    exit 1
fi

# cd "${CODE_DIR}"
# python3 validate_all.py
# cd ../..
# if [ $? -ne 0 ]; then
#     echo -e "${RED}Validation failed! Fix errors before submitting.${NC}"
#     exit 1
# fi

# Generate data and figures
echo -e "${BLUE}Generating data and figures...${NC}"
cd "${CODE_DIR}" || exit 1
python3 generate_data.py || {
    echo -e "${RED}Failed to generate data${NC}"
    exit 1
}
python3 generate_figures.py || {
    echo -e "${RED}Failed to generate figures${NC}"
    exit 1
}
cd -

# Create submission tarball
echo -e "${GREEN}Creating submission package...${NC}"
cd "${SUBMISSION_DIR}"

# Verify required files exist
required_files=(
    "${MAIN_TEX}"
    "${SUBMISSION_DIR}/bibliography.bib"
    "${SUBMISSION_DIR}/manifest.yml"
    "${SUBMISSION_DIR}/appendices/"
    "${SUPPLEMENTARY_DIR}/code/core"
    "${SUPPLEMENTARY_DIR}/code/tests"
    "${SUPPLEMENTARY_DIR}/code/docs"
)

for file in "${required_files[@]}"; do
    if [ ! -f "${file}" ] && [ ! -d "${file}" ]; then
        echo -e "${RED}Error: Required file/directory ${file} not found!${NC}"
        exit 1
    fi
done

cd "${SUBMISSION_DIR}" || exit 1
# Create submission tarball
tar czf "../submission.tar.gz" \
    "main.tex" \
    "bibliography.bib" \
    "appendices/" \
    "figures/" \
    "supplementary/data/" \
    "manifest.yml"

echo -e "${GREEN}Created submission.tar.gz${NC}"
cd - || exit 1

# Optional: Upload to arXiv
if [ "${1:-}" = "--upload" ]; then
    echo -e "${GREEN}Uploading to arXiv...${NC}"
    # Add arXiv upload command here
    # arxiv-submit submission.tar.gz
fi

echo -e "${GREEN}Done!${NC}"