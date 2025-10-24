#!/bin/bash
# Created in 2025.10.23
# Environment Setup for PyTorch Lightning on Aurora
# Source this file before using the examples: source setup_env.sh

# ============================================================================
# User Configuration - EDIT THESE
# ============================================================================

# Set your project allocation name
export MY_PROJECT="${MY_PROJECT:-NeuroX-MM}"

# Set base directory (where you cloned this repo)
export TUTORIAL_BASE="${TUTORIAL_BASE:-/flare/${MY_PROJECT}/${USER}/pytorch_lightning_aurora_example}"

# Set virtual environment path
export VENV_PATH="${VENV_PATH:-/flare/${MY_PROJECT}/PT_2.8.0}"
# Check if VENV_PATH is already set in ~/.bashrc to avoid duplicates
if ! grep -q "export VENV_PATH=" ~/.bashrc; then
    echo "export VENV_PATH=${VENV_PATH}" >> ~/.bashrc
    echo "Added VENV_PATH to ~/.bashrc"
fi

# ============================================================================
# Automatic Configuration - DO NOT EDIT
# ============================================================================

# Change to tutorial directory
cd "${TUTORIAL_BASE}" || { echo "Error: Cannot access ${TUTORIAL_BASE}"; return 1; }

# Load Aurora frameworks
module load frameworks

# Activate virtual environment if it exists
if [ -d "${VENV_PATH}/bin/activate}" ]; then
    source "${VENV_PATH}/bin/activate"
    echo "✓ Virtual environment activated: ${VENV_PATH}"
else
    echo "⚠ Virtual environment not found: ${VENV_PATH}"
    echo "  Creating virtual environment: ${VENV_PATH}"
    python3 -m venv $VENV_PATH --system-site-packages
    source "${VENV_PATH}/bin/activate"
    echo "  Insatlling pytorch lightning 1.9.5"
    yes | python -m pip install pytorch_lightning==1.9.5 --user
fi

# save your current environment configurations using pip freeze
pip freeze > ${TUTORIAL_BASE}/requirements.txt
module li 2>&1 | tee ${TUTORIAL_BASE}/module_list.txt
ds_report # check deepspeed information

# Create necessary directories
mkdir -p logs output

echo "=============================================="
echo "PyTorch Lightning on Aurora - Environment Ready"
echo "=============================================="
echo "Project: ${MY_PROJECT}"
echo "Tutorial: ${TUTORIAL_BASE}"
echo "Venv: ${VENV_PATH}"
echo "=============================================="
echo "Usage"
echo "module load frameworks
source ${VENV_PATH}/bin/activate"
echo "=============================================="
echo "Next steps:"
echo "1. Verify: python project/verify_environment.py"
echo "2. Test: qsub scripts/submit_ddp_simple.sh"
echo "=============================================="
