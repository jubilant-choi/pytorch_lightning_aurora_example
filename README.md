# PyTorch Lightning on Intel Aurora

Production-ready PyTorch Lightning implementation for Intel Aurora XPUs. Modified codes based on the SwiFT v2 project, which was tested up to 128 nodes (1,536 GPUs).

**Author**: Jubin Choi (wnqlszoq123@snu.ac.kr), PhD student at Connectome Lab, Seoul National University

---
## Table of Contents

- [Quick Start](#quick-start)
- [Key Aurora Adaptations](#key-aurora-adaptations)
- **[Strategy Selection Guide](#strategy-selection-guide)** ← **Start here to choose DDP or DeepSpeed**
  - [Option 1: XPUDeepSpeedStrategy (Recommended for Most Cases)](#option-1-xpudeepspeedstrategy-recommended-for-most-cases)
  - [Option 2: MPIDDPStrategy (For Smaller Models)](#option-2-mpiddpstrategy-for-smaller-models)
  - [Strategy Comparison Table](#strategy-comparison-table)
- [Training Strategies Overview](#training-strategies-overview)
- [Performance Optimization](#performance-optimization)
- [Adapting for Your Project](#adapting-for-your-project)
- [Troubleshooting](#troubleshooting)
---

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/jubilant-choi/pytorch_lightning_aurora_example.git pytorch_lightning_aurora_example
cd pytorch_lightning_aurora_example

# Edit setup_env.sh: Set MY_PROJECT="YourProject"
vim setup_env.sh

# Initialize environment
source setup_env.sh
```

### 2. Verify Installation

```bash
python project/verify_environment.py
```

Expected output: `✅ VERIFICATION SUCCESSFUL!`

### 3. Test Run (Interactive Mode)

```bash
# Get interactive node
qsub -A ${MY_PROJECT} -q debug -l select=1 -l walltime=01:00:00 -I

# Test training
bash scripts/submit_ddp_simple.sh 2>&1 | tee logs/test_run.log
```

### 4. Submit Production Job

```bash
# Edit project allocation in scripts/submit_ddp_simple.sh
vim scripts/submit_ddp_simple.sh  # Change #PBS -A YourProject

# Submit job
qsub scripts/submit_ddp_simple.sh

# Monitor logs
tail -f logs/ptl_ddp_simple_*.txt
```

---

## Key Aurora Adaptations

| Component | Standard (CUDA) | Aurora (XPU) |
|-----------|-----------------|--------------|
| Accelerator | `'gpu'` | `'xpu'` |
| Backend | `'nccl'` | `'xccl'` |
| Strategy | Built-in DDP | `MPIDDPStrategy`, `XPUDeepSpeedStrategy` |
| Device | `.to('cuda')` | `.to('xpu')` |
| Precision | `'fp16'` | `'bf16'` (recommended) |

---

## Strategy Selection Guide

### Option 1: XPUDeepSpeedStrategy (Recommended for large scale training)

**Best for:**
- Achieving maximum memory efficiency to enable larger models or batch sizes.
- Models of any size, from small to 100B+ parameters.
- **Stage 2 is the recommended starting point**, offering an excellent balance of memory savings and performance.

**Code Example:**
```python
import torch
import pytorch_lightning as pl
from aurora_utils.deepspeed_intel import XPUDeepSpeedStrategy
from aurora_utils.ddp_intel import MPIEnvironment

# Setup Aurora environment
env = MPIEnvironment()

# Create DeepSpeed strategy (Stage 1 or 2 is a great default)
from pytorch_lightning.plugins.precision import DeepSpeedPrecisionPlugin
strategy = XPUDeepSpeedStrategy(
    accelerator="xpu",
    cluster_environment=env,
    precision_plugin=DeepSpeedPrecisionPlugin(precision='bf16'),
    process_group_backend='xccl',
    stage=2,
    offload_optimizer=True,
    logging_batch_size_per_gpu="auto"
)

# Train
trainer = pl.Trainer(
    strategy=strategy,
    devices=12,
    num_nodes=4,
    precision='bf16'
)
trainer.fit(model, datamodule)
```

**Command-line usage:**
- Always test the best options for your project. 

```bash
NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS_PER_NODE=12
NTOTRANKS=$((NNODES * NRANKS_PER_NODE))
MPI_ARG="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} "
mpi_run_command="mpiexec ${MPI_ARG} scripts/set_rank_and_redirect_outerr.sh "

# Stage 1: Optimizer state partitioning
${mpi_run_command} python project/simple_example.py \
    --strategy deepspeed_stage_1 \
    --num_nodes ${NNODES} \
    --devices ${NRANKS_PER_NODE} \
    --precision bf16

# Stage 2: Optimizer + gradient partitioning (recommended)
${mpi_run_command} python project/simple_example.py \
    --strategy deepspeed_stage_2 \
    --num_nodes ${NNODES} \
    --devices ${NRANKS_PER_NODE} \
    --precision bf16

# Stage 3: Full parameter sharding (for very large models)
${mpi_run_command} python project/simple_example.py \
    --strategy deepspeed_stage_3 \
    --num_nodes ${NNODES} \
    --devices ${NRANKS_PER_NODE} \
    --precision bf16
```

### Option 2: MPIDDPStrategy (For Smaller Models)

**Best for:**
- Smaller models where memory is not a concern.
- Scenarios where maximum throughput is critical and the model + batch size easily fit in GPU memory.
- Simpler checkpointing and debugging workflows.

**Code Example:**
```python
import torch
import pytorch_lightning as pl
from aurora_utils.ddp_intel import MPIDDPStrategy, MPIEnvironment

# Setup Aurora environment
env = MPIEnvironment()
torch.distributed.init_process_group(
    'xccl',
    init_method="env://",
    world_size=env.world_size(),
    rank=env.global_rank()
)

# Create DDP strategy
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
strategy = MPIDDPStrategy(
    accelerator="xpu",
    cluster_environment=env,
    precision_plugin=MixedPrecisionPlugin(
        precision='bf16',
        device='xpu',
        scaler=None
    )
)

# Train
trainer = pl.Trainer(
    strategy=strategy,
    devices=12,      # 12 GPU tiles per Aurora node
    num_nodes=2,
    precision='bf16'
)
trainer.fit(model, datamodule)
```

**Command-line usage:**
```bash
${mpi_run_command} python project/simple_example.py \
    --strategy ddp \
    --num_nodes ${NNODES} \
    --devices ${NRANKS_PER_NODE} \\
    --precision bf16
```

### Strategy Comparison Table

| Feature | MPIDDPStrategy | XPUDeepSpeedStrategy |
|---------|----------------|---------------------|
| **Speed** | Fast for small models | Slow for small models |
| **Memory Usage** | Baseline | Reduced (4-16x savings) |
| **Complexity** | Simple | Moderate |
| **Checkpointing** | Standard | DeepSpeed format |

---

## Project Structure

```
pytorch_lightning_aurora_example/
├── setup_env.sh                    # Environment configuration
├── project/
│   ├── simple_example.py          # Complete working example
│   ├── verify_environment.py      # Environment checker
│   └── aurora_utils/              # Aurora-specific utilities (DO NOT EDIT)
│       ├── ddp_intel.py           # Custom MPI-based DDP strategy
│       ├── deepspeed_intel.py     # DeepSpeed for XPU
│       └── xpu_intel.py           # XPU accelerator
└── scripts/
    ├── submit_ddp_simple.sh       # DDP job (2 nodes)
    └── submit_deepspeed_stage2.sh # DeepSpeed Stage 2 (2 nodes)
```

---

## Training Strategies Overview

This project supports two main distributed training strategies for Intel Aurora XPUs. See [Strategy Selection Guide](#strategy-selection-guide) above for detailed code examples.

### 1. MPIDDPStrategy (DDP)

**Quick Start:**
```bash
qsub scripts/submit_ddp_simple.sh
```

| Aspect | Details |
|--------|---------|
| **Use Case** | Standard models fitting in GPU memory |
| **Memory** | Baseline (no savings) |
| **Speed** | Fastest training speed |
| **Nodes** | 1-32 nodes recommended |
| **Script** | `scripts/submit_ddp_simple.sh` |

### 2. XPUDeepSpeedStrategy

**Quick Start:**
```bash
# Stage 2 (recommended for the start point)
qsub scripts/submit_deepspeed_stage2.sh
```

| Stage | Memory Savings | Example Use Case | Nodes |
|-------|----------------|----------|-------|
| **Stage 1** | 4x (optimizer partitioned) | Medium models | 2-16 |
| **Stage 2** | 8x (optimizer + gradients) | Large models | 4-64 |
| **Stage 3** | 16x (all parameters) | 15B+ params | 8+ |

**DeepSpeed Stage Details:**
- **Stage 1**: Optimizer states partitioned across GPUs
- **Stage 2**: Optimizer + gradients partitioned (recommended)
- **Stage 3**: Full ZeRO - all parameters, gradients, and optimizer states sharded

### How to Choose?

```
Model fits in memory (< 40GB/tile)? → Use MPIDDPStrategy
    ├─ Fast training speed needed? → MPIDDPStrategy ✓
    └─ Standard checkpointing? → MPIDDPStrategy ✓

Model doesn't fit in memory or each data is big ?  → Use XPUDeepSpeedStrategy
    ├─ Medium model (~3B params)? → DeepSpeed Stage 1 or 2
    ├─ Large model (3-15B params)? → DeepSpeed Stage 2 or 3
    └─ Very large model (15B+ params)? → DeepSpeed Stage 3 + CPU Offload
```
---

## Performance Optimization

### DataLoader Configuration
```python
DataLoader(
    dataset,
    num_workers=4,              # 2-4 recommended
    pin_memory=True,            # Faster host→device transfer
    persistent_workers=True,    # Reuse workers across epochs
    prefetch_factor=2           # Prefetch 2 batches/worker
)
```

### Precision Selection
```python
# Recommended for Intel Aurora
trainer = pl.Trainer(precision='bf16')  # 2x faster than fp32
```

---

## Adapting for Your Project

### Step 1: Keep Aurora Components Unchanged
- `project/aurora_utils/` - Custom strategies for XPU
- `setup_aurora_environment()` in `simple_example.py`

### Step 2: Replace Model and Data
```python
# Replace SimpleClassifier with your model
class YourModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        # Your architecture

    def training_step(self, batch, batch_idx):
        # Your training logic
        return loss

# Replace SyntheticDataModule with your data
class YourDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(your_dataset, ...)
```

### Step 3: Configure Resources
Edit `scripts/submit_ddp_simple.sh`:
- `#PBS -A YourProject` - Your project allocation
- `#PBS -l select=N` - Number of nodes
- Training hyperparameters (epochs, batch size, learning rate)

---

## Environment Variables

Configured in `setup_env.sh`:

| Variable | Description | Example |
|----------|-------------|---------|
| `MY_PROJECT` | Aurora project allocation | `"NeuroX-MM"` |
| `TUTORIAL_BASE` | Repository location | `/flare/${MY_PROJECT}/${USER}/...` |
| `VENV_PATH` | Virtual environment path | `/flare/${MY_PROJECT}/PT_2.8.0` |

---

## Troubleshooting

**"XPU not available"**
→ XPUs only available on compute nodes, not login nodes. Submit PBS job.

**"XCCL backend not available"**
→ Load frameworks module: `module load frameworks`

**Job hangs during initialization**
→ Check MASTER_ADDR and MASTER_PORT in PBS script

---

## Resources

- **Aurora Documentation**: https://docs.alcf.anl.gov/aurora/
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/stable/
- **Support**: support@alcf.anl.gov

---

## Citation

```bibtex
@misc{pytorch_lightning_aurora_2025,
  title={PyTorch Lightning on Intel Aurora},
  author={Connectome Lab, Seoul National University},
  year={2025},
  howpublished={\url{https://github.com/jubilant-choi/pytorch_lightning_aurora_example}}
}
```

**License**: MIT | **Acknowledgments**: SwiFT v2 project of Connectome Lab, Seoul National University.