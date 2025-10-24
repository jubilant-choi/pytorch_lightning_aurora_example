#!/bin/bash -l
#PBS -A YourProject
#PBS -N ptl_ddp_simple
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -o logs/${PBS_JOBNAME}_${PBS_JOBID}.txt

######################################################################
# PyTorch Lightning DDP Example on Aurora
#
# BEFORE RUNNING: Edit #PBS -A line above with your project name!
#
# Usage: qsub scripts/submit_ddp_simple.sh
######################################################################

echo "=========================================="
echo "PyTorch Lightning DDP - $(wc -l < $PBS_NODEFILE) Nodes"
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
source "${VENV_PATH}/bin/activate"

### MPI setup ###
export MPI_PROVIDER=$FI_PROVIDER
export MPICH_GPU_SUPPORT_ENABLED=1
export FI_MR_CACHE_MONITOR=userfaultfd

NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS_PER_NODE=12
NTOTRANKS=$((NNODES * NRANKS_PER_NODE))

echo "Nodes: ${NNODES}, Ranks/node: ${NRANKS_PER_NODE}, Total ranks: ${NTOTRANKS}"

export CPU_BIND="verbose,list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
MPI_ARG="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind=${CPU_BIND} --genvall"

export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=29500
export WORLD_SIZE=$NTOTRANKS

export OMP_NUM_THREADS=1
export TMPDIR=/tmp

export LIBRARY_PATH=$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=$PWD/deepspeed
echo "PATH=$PATH" > .deepspeed_env
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> .deepspeed_env
echo "CPATH=$CPATH" >> .deepspeed_env
echo "TORCH_EXTENSIONS_DIR=$PWD/deepspeed" >> .deepspeed_env
export TRITON_CACHE_DIR="$PWD/.triton"

### proxy settings ###
# https://docs.alcf.anl.gov/aurora/getting-started-on-aurora/?h=#proxy
if [[ ! "${HOSTNAME}" =~ aurora-uan ]]; then
  export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
  export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
  export http_proxy="http://proxy.alcf.anl.gov:3128"
  export https_proxy="http://proxy.alcf.anl.gov:3128"
  export ftp_proxy="http://proxy.alcf.anl.gov:3128"
  export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"
fi

######################################################################
# Training Configuration
######################################################################
export MAIN_JOB_ID="${PBS_JOBID%%.*}"
OUTPUT_DIR="${TUTORIAL_BASE}/output/ddp_${MAIN_JOB_ID}"
mkdir -p ${OUTPUT_DIR}

TRAIN_ARGS="--num_nodes ${NNODES} \
--devices ${NRANKS_PER_NODE} \
--strategy ddp \
--precision bf16 \
--max_epochs 10 \
--batch_size 32 \
--learning_rate 1e-3 \
--input_dim 784 \
--hidden_dim 256 \
--output_dim 10 \
--data_size 10000 \
--num_workers 4 \
--output_dir ${OUTPUT_DIR} \
--seed 42"

echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

######################################################################
# Run Training
######################################################################

# Set logging config
export LOGLEVEL=INFO
export REDIRECT_OUTPUT_MODE="ALL_OE" # Options: ALL_OE, ALL_E, Only_RANK0
chmod +x scripts/set_rank_and_redirect_outerr.sh
mpiexec ${MPI_ARG} scripts/set_rank_and_redirect_outerr.sh python project/simple_example.py ${TRAIN_ARGS}

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "=========================================="

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ SUCCESS - Checkpoints: ${OUTPUT_DIR}"
    ls -lh ${OUTPUT_DIR}
else
    echo "✗ FAILED - Check logs/${PBS_JOBNAME}_${PBS_JOBID}.txt"
fi

exit ${EXIT_CODE}
