# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import contextlib
import json
import logging
import os
import platform
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Generator, List, Mapping, Optional, Tuple, TYPE_CHECKING, Union

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from lightning_fabric.plugins import ClusterEnvironment
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH, LRScheduler, ReduceLROnPlateau
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.core.optimizer import _init_optimizers_and_lr_schedulers
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies import StrategyRegistry
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy, remove_module_hooks
from pytorch_lightning.strategies.utils import _fp_to_half
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn, WarningCache
from pytorch_lightning.utilities.types import LRSchedulerConfig, STEP_OUTPUT
from .xpu_intel import XPUAccelerator

log = logging.getLogger(__name__)
warning_cache = WarningCache()

_DEEPSPEED_AVAILABLE = RequirementCache("deepspeed")
if TYPE_CHECKING and _DEEPSPEED_AVAILABLE:
    import deepspeed

__all__ = ["XPUDeepSpeedStrategy"]

class XPUDeepSpeedStrategy(DeepSpeedStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_deepspeed(self) -> None:
        assert self.lightning_module is not None
        # deepspeed handles gradient clipping internally
        if is_overridden("configure_gradient_clipping", self.lightning_module, pl.LightningModule):
            rank_zero_warn(
                "Since DeepSpeed handles gradient clipping internally, the default"
                " `LightningModule.configure_gradient_clipping` implementation will not actually clip gradients."
                " The hook will still be called. Consider setting"
                " `Trainer(gradient_clip_val=..., gradient_clip_algorithm='norm')`"
                " which will use the internal mechanism."
            )

        if self.lightning_module.trainer.gradient_clip_algorithm == GradClipAlgorithmType.VALUE:
            raise MisconfigurationException("DeepSpeed does not support clipping gradients by value.")

        if not isinstance(self.accelerator, (CUDAAccelerator, XPUAccelerator)):
            raise MisconfigurationException(
                f"DeepSpeed strategy is only supported on GPU but `{self.accelerator.__class__.__name__}` is used."
            )

        accumulation_scheduler = self.lightning_module.trainer.accumulation_scheduler

        if accumulation_scheduler.epochs != [0]:
            raise MisconfigurationException(
                "DeepSpeed currently does not support different `accumulate_grad_batches` at different epochs."
            )

        assert isinstance(self.model, (pl.LightningModule, _LightningPrecisionModuleWrapperBase))
        model = _LightningModuleWrapperBase(forward_module=self.model)

        if self.lightning_module.trainer and self.lightning_module.trainer.training:
            self._initialize_deepspeed_train(model)
        else:
            self._initialize_deepspeed_inference(model)

StrategyRegistry.register(
    "deepspeed_xpu",
    XPUDeepSpeedStrategy,
    description="Run distributed data parallel on Intel XPUs.",
    process_group_backend="xccl",
    accelerator="xpu"
)