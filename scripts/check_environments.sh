#!/bin/bash -l
#PBS -A YourProject
#PBS -N ptl_check_env
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -o logs/${PBS_JOBNAME}_${PBS_JOBID}.txt

######################################################################
# PyTorch Lightning Venv Check on Aurora
#
# BEFORE RUNNING: Edit #PBS -A line above with your project name!
#
# Usage: qsub scripts/check_environments.sh
######################################################################

echo "=========================================="
echo "Checking venv and frameworks"
echo "Started: $(date)"
echo "=========================================="

######################################################################
# Environment Setup
######################################################################

# Set project and paths (EDIT THESE!)
export MY_PROJECT="${MY_PROJECT:-YourProject}"
export TUTORIAL_BASE="${TUTORIAL_BASE:-/flare/${MY_PROJECT}/${USER}/pytorch_lightning_aurora_example}"
export VENV_PATH="${VENV_PATH:-/flare/${MY_PROJECT}/PT_2.8.0}"

cd "${TUTORIAL_BASE}" || exit 1
mkdir -p logs output

### Load modules and activate environment ###
module load frameworks
source ${VENV_PATH}/bin/activate

mpiexec -n 1 -ppn 1 python3 project/verify_environment.py

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED - Check logs/${PBS_JOBNAME}_${PBS_JOBID}.txt"
fi

exit ${EXIT_CODE}